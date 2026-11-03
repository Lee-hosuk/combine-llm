#!/usr/bin/env python3
"""spec_bench.py — vLLM 추측 디코딩 α·c 예비 측정 (역할 ② 드래프트·타겟)

모드
  target_only : 타겟 모델 단독 (기준선, c의 분모)
  draft_only  : 드래프트 모델 단독 (c의 분자)
  spec        : 드래프트 + 타겟 추측 디코딩 (α, 수락 길이, 실측 속도)
  summarize   : results/*.json 을 모아 표와 summary.csv/json 생성 (GPU 불필요)

측정 원칙 (제안서 6절 준용)
  - greedy(temperature 0), Qwen3 non-thinking(enable_thinking=False), 배치 1
  - 프롬프트마다 reps회 반복, 앞 warmup회는 버림 (기본 12/4, Kummer et al. Wild 준용)
  - 프리픽스 캐싱 기본 끔 (같은 프롬프트 반복 시 prefill이 캐시돼 TTFT가 왜곡되지 않게)
  - TPOT(토큰당 디코딩 시간) = (전체 생성 시간 - 첫 토큰만 생성 시간) / (출력 토큰 수 - 1)

입력 프롬프트
  --prompts 없이 실행하면 내장 예시 6개(코딩 에이전트 형식)를 쓴다.
  실제 기록은 JSONL로 넘긴다. 한 줄 = {"id": "...", "messages": [...], "tools": [...](선택)}
  messages 는 OpenAI 채팅 형식(role/content). ③번 팀원의 재생 도구 출력 형식과 맞출 것.
"""
import argparse
import glob
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 내장 예시 프롬프트 (연습용, 영어 — 실제 평가 데이터가 영어이므로)
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are a software engineering agent working in a Python repository. "
    "At each step, think briefly, then reply with exactly one action: either a "
    "bash command inside a ```bash``` block, or a code edit as a unified diff "
    "inside a ```diff``` block. Do not ask the user questions."
)

SAMPLE_PROMPTS = [
    {
        "id": "sample_read_file",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Issue: `parse_date('2024-03-01T10:00:00+09:00')` raises TypeError: can't compare offset-naive and offset-aware datetimes. Expected it to return an aware datetime."},
            {"role": "assistant", "content": "Let me look at the implementation.\n```bash\ncat -n src/utils/dates.py\n```"},
            {"role": "user", "content": "[tool output]\n"
             "     1\tfrom datetime import datetime, timezone\n"
             "     2\t\n"
             "     3\tEPOCH = datetime(1970, 1, 1)\n"
             "     4\t\n"
             "     5\t\n"
             "     6\tdef parse_date(value: str) -> datetime:\n"
             "     7\t    \"\"\"Parse an ISO-8601 string.\"\"\"\n"
             "     8\t    dt = datetime.fromisoformat(value)\n"
             "     9\t    if dt < EPOCH:\n"
             "    10\t        raise ValueError(f\"date before epoch: {value}\")\n"
             "    11\t    return dt\n"
             "    12\t\n"
             "    13\t\n"
             "    14\tdef to_timestamp(dt: datetime) -> float:\n"
             "    15\t    return (dt - EPOCH).total_seconds()\n"
             "[exit code 0]"},
        ],
    },
    {
        "id": "sample_pytest_fail",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Issue: `Cart.total()` ignores the discount when quantity is greater than 10."},
            {"role": "assistant", "content": "```bash\npython -m pytest tests/test_cart.py -x -q\n```"},
            {"role": "user", "content": "[tool output]\n"
             ".....F\n"
             "=================================== FAILURES ===================================\n"
             "__________________________ test_bulk_discount_applied ___________________________\n"
             "\n"
             "    def test_bulk_discount_applied():\n"
             "        cart = Cart()\n"
             "        cart.add(Item('pen', price=2.0), quantity=12)\n"
             ">       assert cart.total() == pytest.approx(21.6)\n"
             "E       assert 24.0 == 21.6 \u00b1 2.2e-05\n"
             "\n"
             "tests/test_cart.py:41: AssertionError\n"
             "=========================== short test summary info ============================\n"
             "FAILED tests/test_cart.py::test_bulk_discount_applied - assert 24.0 == 21.6\n"
             "1 failed, 5 passed in 0.31s\n"
             "[exit code 1]"},
        ],
    },
    {
        "id": "sample_grep",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Issue: the `--verbose` flag of the CLI has no effect since v2.3."},
            {"role": "assistant", "content": "```bash\ngrep -rn \"verbose\" src/ | head -n 20\n```"},
            {"role": "user", "content": "[tool output]\n"
             "src/cli/main.py:18:    parser.add_argument('--verbose', '-v', action='store_true')\n"
             "src/cli/main.py:44:    configure_logging(level=args.log_level)\n"
             "src/cli/logging_setup.py:7:def configure_logging(level='INFO', verbose=False):\n"
             "src/cli/logging_setup.py:9:    if verbose:\n"
             "src/cli/logging_setup.py:10:        level = 'DEBUG'\n"
             "src/core/runner.py:112:        if self.options.get('verbose'):\n"
             "src/core/runner.py:113:            logger.debug('task %s started', task.name)\n"
             "[exit code 0]"},
        ],
    },
    {
        "id": "sample_import_error",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Issue: importing `pkg.io.readers` fails on a fresh install."},
            {"role": "assistant", "content": "```bash\npython -c \"import pkg.io.readers\"\n```"},
            {"role": "user", "content": "[tool output]\n"
             "Traceback (most recent call last):\n"
             "  File \"<string>\", line 1, in <module>\n"
             "  File \"/workspace/repo/pkg/io/readers.py\", line 3, in <module>\n"
             "    from pkg.io.formats import detect_format\n"
             "  File \"/workspace/repo/pkg/io/formats.py\", line 5, in <module>\n"
             "    from pkg.io.readers import BaseReader\n"
             "ImportError: cannot import name 'BaseReader' from partially initialized module "
             "'pkg.io.readers' (most likely due to a circular import) (/workspace/repo/pkg/io/readers.py)\n"
             "[exit code 1]"},
        ],
    },
    {
        "id": "sample_ls",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Issue: `Config.load()` should read YAML files with the `.yml` extension, not only `.yaml`."},
            {"role": "assistant", "content": "```bash\nfind . -name \"*.py\" -path \"*config*\" | sort\n```"},
            {"role": "user", "content": "[tool output]\n"
             "./src/app/config/__init__.py\n"
             "./src/app/config/loader.py\n"
             "./src/app/config/schema.py\n"
             "./src/app/config/sources.py\n"
             "./tests/config/test_loader.py\n"
             "./tests/config/test_schema.py\n"
             "[exit code 0]"},
        ],
    },
    {
        "id": "sample_git_diff",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Issue: `slugify('Hello  World')` returns 'hello--world' but should collapse repeated separators."},
            {"role": "assistant", "content": "```bash\ncat -n src/text/slug.py\n```"},
            {"role": "user", "content": "[tool output]\n"
             "     1\timport re\n"
             "     2\timport unicodedata\n"
             "     3\t\n"
             "     4\t\n"
             "     5\tdef slugify(text: str, sep: str = '-') -> str:\n"
             "     6\t    text = unicodedata.normalize('NFKD', text)\n"
             "     7\t    text = text.encode('ascii', 'ignore').decode('ascii')\n"
             "     8\t    text = re.sub(r'[^a-zA-Z0-9]', sep, text)\n"
             "     9\t    return text.strip(sep).lower()\n"
             "[exit code 0]"},
        ],
    },
]


# ---------------------------------------------------------------------------
# 공통 함수
# ---------------------------------------------------------------------------
def load_prompts(path, limit):
    if path:
        items = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj.setdefault("id", f"row{i}")
                items.append(obj)
    else:
        items = list(SAMPLE_PROMPTS)
    return items[:limit] if limit else items


def to_token_ids(tokenizer, item):
    """채팅 템플릿(non-thinking)을 적용한 뒤 토큰 id 목록으로 변환."""
    kwargs = {"add_generation_prompt": True, "tokenize": False, "enable_thinking": False}
    if item.get("tools"):
        kwargs["tools"] = item["tools"]
    text = tokenizer.apply_chat_template(item["messages"], **kwargs)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def prev_turn_prefix(tokenizer, item, ids):
    """C0b용: 직전 모델 호출(마지막 assistant 메시지 직전까지의 대화)의 입력과
    현재 입력이 토큰 단위로 공유하는 앞부분. 에이전트가 직전 턴에서 이미 계산해
    캐시에 남아 있었을 부분이다. assistant 메시지가 없으면 빈 목록."""
    msgs = item["messages"]
    last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
    if last is None:
        return []
    prev_ids = to_token_ids(tokenizer, {**item, "messages": msgs[:last]})
    n = 0
    for a, b in zip(prev_ids, ids):
        if a != b:
            break
        n += 1
    return ids[:n]


SPEC_KEYS = {
    "vllm:spec_decode_num_drafts": "drafts",
    "vllm:spec_decode_num_draft_tokens": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "accepted",
}
PER_POS_KEY = "vllm:spec_decode_num_accepted_tokens_per_pos"


def read_spec_counters(llm):
    """vLLM 누적 카운터를 읽는다 (disable_log_stats=False 필요)."""
    out = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "per_pos": []}
    for m in llm.get_metrics():
        name = m.name[:-6] if m.name.endswith("_total") else m.name
        if name in SPEC_KEYS:
            out[SPEC_KEYS[name]] += float(getattr(m, "value", 0) or 0)
        elif name == PER_POS_KEY:
            vals = [float(v) for v in getattr(m, "values", [])]
            if len(out["per_pos"]) < len(vals):
                out["per_pos"] += [0.0] * (len(vals) - len(out["per_pos"]))
            for i, v in enumerate(vals):
                out["per_pos"][i] += v
    return out


def diff_counters(after, before):
    d = {k: after[k] - before[k] for k in ("drafts", "draft_tokens", "accepted")}
    n = max(len(after["per_pos"]), len(before["per_pos"]))
    a = after["per_pos"] + [0.0] * (n - len(after["per_pos"]))
    b = before["per_pos"] + [0.0] * (n - len(before["per_pos"]))
    d["per_pos"] = [x - y for x, y in zip(a, b)]
    return d


def expected_tokens(alpha, k):
    """Leviathan et al. 식: 검증 1회당 기대 생성 토큰 = (1 - α^(k+1)) / (1 - α)."""
    return sum(alpha ** i for i in range(k + 1))


def alpha_from_acceptance_length(L, k):
    """평균 수락 길이 L(보너스 토큰 포함)에서 Leviathan α를 역산 (이분법)."""
    if L is None:
        return None
    if L <= 1.0:
        return 0.0
    if L >= k + 1 - 1e-9:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if expected_tokens(mid, k) < L:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def predicted_speedup(alpha, k, c):
    """Leviathan Theorem 3.8: 기대 속도 향상 = (1-α^(k+1)) / ((1-α)(kc+1))."""
    if alpha is None or c is None:
        return None
    return expected_tokens(alpha, k) / (k * c + 1)


def degenerate(toks, min_len=32, frac=0.5):
    """fp16 오버플로(inf/NaN) 의심: 출력이 비었거나 한 토큰이 절반 이상을 차지하면 True"""
    if not toks:
        return True
    if len(toks) < min_len:
        return False
    top = max(toks.count(t) for t in set(toks))
    return top / len(toks) > frac


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def env_info():
    info = {"python": platform.python_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["gpu_count"] = torch.cuda.device_count()
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = str(e)
    try:
        import vllm
        info["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        info["vllm_error"] = str(e)
    return info


# ---------------------------------------------------------------------------
# 측정
# ---------------------------------------------------------------------------
def run_measure(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    is_spec = args.mode == "spec"
    model = args.draft if args.mode == "draft_only" else args.target
    k = args.k if is_spec else 0

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    items = load_prompts(args.prompts, args.limit)
    max_input = args.max_model_len - args.max_tokens

    llm_kwargs = dict(
        model=model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        tensor_parallel_size=args.tp,
        enable_prefix_caching=args.prefix_caching,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
        seed=0,
    )
    hf_overrides = json.loads(args.hf_overrides) if args.hf_overrides else {}
    if args.yarn:
        # Qwen3 모델 카드의 131,072 문맥 설정 (factor 4 × 원래 32,768). vLLM 버전에 따라 키 이름이 다를 수 있어 빠른 점검 필수
        hf_overrides.setdefault("rope_scaling", {"rope_type": "yarn", "factor": 4.0,
                                                 "original_max_position_embeddings": 32768})
    if hf_overrides:
        llm_kwargs["hf_overrides"] = hf_overrides
    if is_spec:
        spec_cfg = {
            "method": "draft_model",
            "model": args.draft,
            "num_speculative_tokens": k,
            "max_model_len": args.max_model_len,
        }
        if args.draft_tp:
            spec_cfg["draft_tensor_parallel_size"] = args.draft_tp
        llm_kwargs["speculative_config"] = spec_cfg

    print(f"[load] mode={args.mode} model={model}" + (f" draft={args.draft} k={k}" if is_spec else ""))
    llm = LLM(**llm_kwargs)

    sp_first = SamplingParams(temperature=0.0, max_tokens=1)
    sp_warm = SamplingParams(temperature=0.0, max_tokens=1)

    def prime(warm_ids):
        """캐싱 켠 경우: 캐시를 비운 뒤 직전 턴 입력만 미리 계산해 둔다.
        TTFT 측정과 전체 생성 측정 앞에서 각각 호출해, 둘 다 '직전 턴까지만 캐시된' 같은 상태에서 잰다."""
        if not args.prefix_caching:
            return
        if not llm.reset_prefix_cache():
            print("[경고] reset_prefix_cache 실패 (실행 중 요청이 남아 있음). 캐시 상태가 섞일 수 있습니다.")
        if warm_ids:
            llm.generate([{"prompt_token_ids": warm_ids}], sp_warm, use_tqdm=False)
    sp_full = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    total_before = read_spec_counters(llm) if is_spec else None
    rows = []
    n_degenerate = 0
    for idx, item in enumerate(items):
        ids = to_token_ids(tokenizer, item)
        if len(ids) > max_input:
            print(f"[skip] {item['id']}: 입력 {len(ids)}토큰 > 허용 {max_input}")
            rows.append({"id": item["id"], "prompt_tokens": len(ids), "skipped": True, "meta": item.get("meta")})
            continue
        inp = {"prompt_token_ids": ids}
        warm_ids = prev_turn_prefix(tokenizer, item, ids) if args.prefix_caching else []
        runs = []
        out_ids = None
        for r in range(args.reps):
            prime(warm_ids)
            t0 = time.perf_counter()
            llm.generate([inp], sp_first, use_tqdm=False)
            t_first = time.perf_counter() - t0

            prime(warm_ids)
            c0 = read_spec_counters(llm) if is_spec else None
            t0 = time.perf_counter()
            out = llm.generate([inp], sp_full, use_tqdm=False)
            t_full = time.perf_counter() - t0
            c1 = read_spec_counters(llm) if is_spec else None

            toks = list(out[0].outputs[0].token_ids)
            n_out = len(toks)
            if out_ids is None:
                out_ids = toks
            run = {
                "rep": r,
                "warmup": r < args.warmup,
                "t_first_s": t_first,
                "t_full_s": t_full,
                "n_out": n_out,
                "tpot_s": (t_full - t_first) / (n_out - 1) if n_out > 1 else None,
                "same_output_as_rep0": toks == out_ids,
            }
            if is_spec:
                run["spec"] = diff_counters(c1, c0)
            runs.append(run)

        measured = [x for x in runs if not x["warmup"]] or runs
        row = {
            "id": item["id"],
            "prompt_tokens": len(ids),
            "n_out": measured[-1]["n_out"],
            "ttft_s_median": median([x["t_first_s"] for x in measured]),
            "e2e_s_median": median([x["t_full_s"] for x in measured]),
            "tpot_s_median": median([x["tpot_s"] for x in measured]),
            "deterministic": all(x["same_output_as_rep0"] for x in runs),
            "output_token_ids": out_ids,
            "output_text": tokenizer.decode(out_ids or []),
            "runs": runs,
            "meta": item.get("meta"),
            "degenerate": degenerate(out_ids),
            "cached_prefix_tokens": len(warm_ids),
            "new_prefill_tokens": len(ids) - len(warm_ids),
        }
        if row["degenerate"]:
            n_degenerate += 1
            print(f"[경고] {item['id']}: 출력이 비었거나 같은 토큰 반복 (dtype={args.dtype} 수치 오버플로 의심)")
        if is_spec:
            s = measured[-1]["spec"]
            row["spec_last_run"] = s
            row["acceptance_length"] = 1 + s["accepted"] / s["drafts"] if s["drafts"] else None
            row["draft_acceptance_rate"] = s["accepted"] / s["draft_tokens"] if s["draft_tokens"] else None
        rows.append(row)
        msg = f"[{idx + 1}/{len(items)}] {item['id']}: 입력 {len(ids)}토큰" + (f"(캐시 {len(warm_ids)})" if args.prefix_caching else "") + ", " + f"출력 {row['n_out']}토큰, TPOT {row['tpot_s_median'] * 1000:.1f} ms" if row["tpot_s_median"] else f"[{idx + 1}/{len(items)}] {item['id']}"
        if is_spec and row.get("acceptance_length"):
            msg += f", 수락 길이 {row['acceptance_length']:.2f}"
        print(msg)

    result = {
        "mode": args.mode,
        "target": args.target,
        "draft": args.draft if args.mode != "target_only" else None,
        "model_run": model,
        "k": k,
        "config": {key: getattr(args, key) for key in (
            "dtype", "max_model_len", "max_tokens", "gpu_mem", "tp", "draft_tp",
            "prefix_caching", "enforce_eager", "reps", "warmup", "prompts", "limit", "yarn", "hf_overrides")},
        "env": env_info(),
        "n_degenerate": n_degenerate,
        "rows": rows,
    }
    if is_spec:
        tot = diff_counters(read_spec_counters(llm), total_before)
        result["spec_total"] = tot
        result["acceptance_length"] = 1 + tot["accepted"] / tot["drafts"] if tot["drafts"] else None
        result["draft_acceptance_rate"] = tot["accepted"] / tot["draft_tokens"] if tot["draft_tokens"] else None
        result["per_pos_acceptance"] = [p / tot["drafts"] for p in tot["per_pos"]] if tot["drafts"] else []
        if not tot["drafts"]:
            print("[경고] 추측 디코딩 카운터가 0입니다. vLLM 버전에서 지표 이름이 바뀌었을 수 있습니다.")

    n_ok = sum(1 for r in rows if not r.get("skipped"))
    if n_degenerate:
        print(f"[경고] 이상 출력 {n_degenerate}/{n_ok}턴. fp16 오버플로 가능성이 있으니 로그와 함께 알려 주세요.")

    label = args.label or default_label(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {path}")
    if is_spec and result["acceptance_length"]:
        print(f"[spec] 평균 수락 길이 {result['acceptance_length']:.3f}, "
              f"드래프트 토큰 채택 비율 {result['draft_acceptance_rate']:.3f}, "
              f"Leviathan α 추정 {alpha_from_acceptance_length(result['acceptance_length'], k):.3f}")


def short(name):
    return name.rstrip("/").split("/")[-1] if name else "none"


def default_label(args):
    if args.mode == "target_only":
        return f"target_only__{short(args.target)}"
    if args.mode == "draft_only":
        return f"draft_only__{short(args.draft)}"
    return f"spec__{short(args.draft)}__{short(args.target)}__k{args.k}"


# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
def by_id(result):
    return {r["id"]: r for r in result["rows"] if not r.get("skipped")}


def summarize(args):
    files = sorted(glob.glob(os.path.join(args.out_dir, "*.json")))
    files = [f for f in files if not os.path.basename(f).startswith("summary")]
    results = [json.load(open(f, encoding="utf-8")) for f in files]
    targets = {r["model_run"]: r for r in results if r["mode"] == "target_only"}
    drafts = {r["model_run"]: r for r in results if r["mode"] == "draft_only"}
    specs = [r for r in results if r["mode"] == "spec"]
    if not specs:
        print("spec 결과가 없습니다. 먼저 --mode spec 을 실행하세요.")
        return

    table = []
    for s in specs:
        tgt, drf = targets.get(s["target"]), drafts.get(s["draft"])
        k = s["k"]
        sid = by_id(s)
        tid = by_id(tgt) if tgt else {}
        did = by_id(drf) if drf else {}
        common = [i for i in sid if i in tid]

        c_vals = [did[i]["tpot_s_median"] / tid[i]["tpot_s_median"]
                  for i in common if i in did and did[i]["tpot_s_median"] and tid[i]["tpot_s_median"]]
        c = median(c_vals)
        tpot_speed = median([tid[i]["tpot_s_median"] / sid[i]["tpot_s_median"]
                             for i in common if sid[i]["tpot_s_median"] and tid[i]["tpot_s_median"]])
        e2e_speed = median([tid[i]["e2e_s_median"] / sid[i]["e2e_s_median"]
                            for i in common if sid[i]["e2e_s_median"]])
        same = [sid[i]["output_token_ids"] == tid[i]["output_token_ids"] for i in common]

        L = s.get("acceptance_length")
        alpha = alpha_from_acceptance_length(L, k)
        table.append({
            "draft": short(s["draft"]),
            "target": short(s["target"]),
            "k": k,
            "n_prompts": len(sid),
            "acceptance_length": L,
            "draft_acceptance_rate": s.get("draft_acceptance_rate"),
            "alpha_leviathan": alpha,
            "per_pos_acceptance": s.get("per_pos_acceptance"),
            "c_measured": c,
            "c_n_prompts": len(c_vals),
            "speedup_predicted": predicted_speedup(alpha, k, c),
            "speedup_tpot_measured": tpot_speed,
            "speedup_e2e_measured": e2e_speed,
            "output_match_vs_target": (sum(same) / len(same)) if same else None,
            "gpu": s["env"].get("gpu"),
            "vllm": s["env"].get("vllm"),
        })

    def f(x, nd=3):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    hdr = ["draft", "target", "k", "L", "rate", "α", "c", "예측", "실측TPOT", "실측E2E", "출력일치"]
    print(" | ".join(hdr))
    for t in table:
        print(" | ".join([t["draft"], t["target"], str(t["k"]), f(t["acceptance_length"], 2),
                          f(t["draft_acceptance_rate"]), f(t["alpha_leviathan"]), f(t["c_measured"]),
                          f(t["speedup_predicted"], 2) + "x" if t["speedup_predicted"] else "-",
                          f(t["speedup_tpot_measured"], 2) + "x" if t["speedup_tpot_measured"] else "-",
                          f(t["speedup_e2e_measured"], 2) + "x" if t["speedup_e2e_measured"] else "-",
                          f(t["output_match_vs_target"], 2)]))
        if t["c_measured"] is None:
            print("   (c 없음: 같은 프롬프트로 draft_only 와 target_only 를 먼저 실행하세요)")

    out = Path(args.out_dir)
    (out / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = [k for k in table[0] if k != "per_pos_acceptance"]
    with open(out / "summary.csv", "w", encoding="utf-8") as fp:
        fp.write(",".join(cols) + "\n")
        for t in table:
            fp.write(",".join("" if t[c] is None else str(t[c]) for c in cols) + "\n")
    print(f"[saved] {out / 'summary.json'}, {out / 'summary.csv'}")


# ---------------------------------------------------------------------------
# 원본 대 압축 비교 (C0a / C1 / C2 / C3 미니판)
# ---------------------------------------------------------------------------
def load_results(d):
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    files = [f for f in files if not os.path.basename(f).startswith("summary")]
    return [json.load(open(f, encoding="utf-8")) for f in files]


def pick(results, mode, k=None):
    found = [r for r in results if r["mode"] == mode and (k is None or r["k"] == k)]
    return found[-1] if found else None


def row_alpha(row, k):
    L = row.get("acceptance_length")
    return alpha_from_acceptance_length(L, k) if L else None


def compare(args):
    """--out-dir(원본)과 --other-dir(압축)의 결과를 같은 샘플끼리 짝지어 비교."""
    A, B = load_results(args.out_dir), load_results(args.other_dir)
    tA, tB = pick(A, "target_only"), pick(B, "target_only")
    if not (tA and tB):
        print("두 폴더 모두 target_only 결과가 있어야 합니다.")
        return
    a_rows, b_rows = by_id(tA), by_id(tB)
    ids = [i for i in a_rows if i in b_rows]

    def med(f):
        return median([f(i) for i in ids])

    def ratio(x, y):
        return x / y if (x and y) else None

    report = {"n_pairs": len(ids), "orig_dir": args.out_dir, "comp_dir": args.other_dir}
    report["prompt_tokens_median"] = {"orig": med(lambda i: a_rows[i]["prompt_tokens"]),
                                      "comp": med(lambda i: b_rows[i]["prompt_tokens"])}
    report["prompt_token_ratio_median"] = med(lambda i: ratio(b_rows[i]["prompt_tokens"], a_rows[i]["prompt_tokens"]))
    report["target_output_same_rate"] = (sum(a_rows[i]["output_token_ids"] == b_rows[i]["output_token_ids"]
                                             for i in ids) / len(ids)) if ids else None
    report["output_tokens_median"] = {"orig": med(lambda i: a_rows[i]["n_out"]), "comp": med(lambda i: b_rows[i]["n_out"])}

    # 모든 속도 향상은 C0a(원본, 추측 없음) 대비, 같은 샘플끼리의 비의 중앙값
    def speed(base_rows, rows, key):
        return median([ratio(base_rows[i][key], rows[i][key]) for i in ids if i in rows])

    conds = [{"cond": "C0a 원본", "ttft": 1.0, "tpot": 1.0, "e2e": 1.0}]
    conds.append({"cond": "C1 압축만", "ttft": speed(a_rows, b_rows, "ttft_s_median"),
                  "tpot": speed(a_rows, b_rows, "tpot_s_median"), "e2e": speed(a_rows, b_rows, "e2e_s_median")})
    ks = sorted({r["k"] for r in A if r["mode"] == "spec"} & {r["k"] for r in B if r["mode"] == "spec"})
    alpha_rows = []
    for k in ks:
        sA, sB = by_id(pick(A, "spec", k)), by_id(pick(B, "spec", k))
        c2 = {"cond": f"C2 추측만 γ={k}", "ttft": speed(a_rows, sA, "ttft_s_median"),
              "tpot": speed(a_rows, sA, "tpot_s_median"), "e2e": speed(a_rows, sA, "e2e_s_median")}
        c3 = {"cond": f"C3 결합 γ={k}", "ttft": speed(a_rows, sB, "ttft_s_median"),
              "tpot": speed(a_rows, sB, "tpot_s_median"), "e2e": speed(a_rows, sB, "e2e_s_median")}
        c1 = conds[1]
        c3["e2e_if_independent"] = (c1["e2e"] * c2["e2e"]) if (c1["e2e"] and c2["e2e"]) else None
        conds += [c2, c3]

        pA, pB = pick(A, "spec", k), pick(B, "spec", k)
        aA = alpha_from_acceptance_length(pA.get("acceptance_length"), k)
        aB = alpha_from_acceptance_length(pB.get("acceptance_length"), k)
        diffs = [row_alpha(sB[i], k) - row_alpha(sA[i], k) for i in ids
                 if i in sA and i in sB and row_alpha(sA[i], k) is not None and row_alpha(sB[i], k) is not None]
        alpha_rows.append({
            "k": k, "alpha_orig": aA, "alpha_comp": aB,
            "L_orig": pA.get("acceptance_length"), "L_comp": pB.get("acceptance_length"),
            "per_sample_alpha_diff_median": median(diffs),
            "share_alpha_down": (sum(d < 0 for d in diffs) / len(diffs)) if diffs else None,
            "n_alpha_pairs": len(diffs),
            "spec_output_match_orig": (sum(sA[i]["output_token_ids"] == a_rows[i]["output_token_ids"] for i in ids if i in sA) / len(ids)) if ids else None,
            "spec_output_match_comp": (sum(sB[i]["output_token_ids"] == b_rows[i]["output_token_ids"] for i in ids if i in sB) / len(ids)) if ids else None,
        })
    report["conditions"] = conds
    report["alpha"] = alpha_rows

    def f(x, nd=2):
        return "-" if x is None else f"{x:.{nd}f}"

    print(f"짝지은 샘플 {len(ids)}개 | 프롬프트 토큰 중앙값 원본 {f(report['prompt_tokens_median']['orig'], 0)} → 압축 {f(report['prompt_tokens_median']['comp'], 0)} (비율 {f(report['prompt_token_ratio_median'])})")
    print(f"타겟 출력이 원본과 같은 비율(압축 후) {f(report['target_output_same_rate'])} | 출력 토큰 중앙값 원본 {f(report['output_tokens_median']['orig'], 0)} → 압축 {f(report['output_tokens_median']['comp'], 0)}")
    print("\n[속도 향상, C0a 대비, 샘플별 비의 중앙값]")
    print("조건 | TTFT | TPOT | E2E | (C1×C2 곱)")
    for c in conds:
        print(" | ".join([c["cond"], f(c["ttft"]) + "x", f(c["tpot"]) + "x", f(c["e2e"]) + "x",
                          (f(c["e2e_if_independent"]) + "x") if c.get("e2e_if_independent") else ""]))
    print("\n[α: 원본 → 압축]")
    print("γ | α 원본 | α 압축 | L 원본 | L 압축 | 샘플별 α 변화 중앙값 | α 하락 비율 | 출력일치(원본/압축)")
    for a in alpha_rows:
        print(" | ".join([str(a["k"]), f(a["alpha_orig"], 3), f(a["alpha_comp"], 3), f(a["L_orig"]), f(a["L_comp"]),
                          f(a["per_sample_alpha_diff_median"], 3), f(a["share_alpha_down"]),
                          f"{f(a['spec_output_match_orig'])}/{f(a['spec_output_match_comp'])}"]))
    out = Path(args.out_dir) / "summary_compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {out}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 캐싱 끔(C0a) 대 캐싱 켬(C0b) 속도 비교
# ---------------------------------------------------------------------------
def _ratio_median(pairs):
    vals = [a / b for a, b in pairs if a is not None and b]
    return median(vals) if vals else None


def cache_compare(args):
    """두 결과 폴더(캐싱 끔 / 켬)를 턴 ID로 짝지어 속도 차이를 계산.
    속도향상 = 끔 시간 / 켬 시간 (턴별 비율의 중앙값). 1보다 크면 캐싱이 빠름."""
    off_res, on_res = load_results(args.off_dir), load_results(args.on_dir)
    if not off_res or not on_res:
        print(f"[cachecmp] 결과가 없습니다: 끔={args.off_dir} ({len(off_res)}개), 켬={args.on_dir} ({len(on_res)}개)")
        return
    for name, res, want in (("끔", off_res, False), ("켬", on_res, True)):
        bad = [r["mode"] for r in res if bool(r["config"].get("prefix_caching")) != want]
        if bad:
            print(f"[경고] '{name}' 폴더에 prefix_caching 설정이 다른 결과가 있습니다: {bad}")
    dts = {r["config"].get("dtype") for r in off_res + on_res}
    if len(dts) > 1:
        print(f"[경고] 두 폴더의 dtype 이 다릅니다: {dts}. 캐싱 외 차이가 섞입니다.")

    keys = []
    for r in off_res:
        if r["mode"] in ("target_only", "draft_only", "spec"):
            keys.append((r["mode"], r["k"]))
    keys = sorted(set(keys), key=lambda x: (["target_only", "draft_only", "spec"].index(x[0]), x[1]))

    table, turns = [], []
    for mode, k in keys:
        a, b = pick(off_res, mode, k), pick(on_res, mode, k)
        if not b:
            print(f"[cachecmp] 켬 폴더에 {mode} k={k} 결과가 없어 건너뜀")
            continue
        A, B = by_id(a), by_id(b)
        ids = [i for i in A if i in B]
        if not ids:
            continue

        def stats(sub):
            return {
                "n": len(sub),
                "ttft_off_s": median([A[i]["ttft_s_median"] for i in sub]),
                "ttft_on_s": median([B[i]["ttft_s_median"] for i in sub]),
                "ttft_speedup": _ratio_median([(A[i]["ttft_s_median"], B[i]["ttft_s_median"]) for i in sub]),
                "tpot_off_ms": 1000 * median([A[i]["tpot_s_median"] for i in sub if A[i]["tpot_s_median"]] or [0]),
                "tpot_on_ms": 1000 * median([B[i]["tpot_s_median"] for i in sub if B[i]["tpot_s_median"]] or [0]),
                "tpot_speedup": _ratio_median([(A[i]["tpot_s_median"], B[i]["tpot_s_median"]) for i in sub]),
                "e2e_off_s": median([A[i]["e2e_s_median"] for i in sub]),
                "e2e_on_s": median([B[i]["e2e_s_median"] for i in sub]),
                "e2e_speedup": _ratio_median([(A[i]["e2e_s_median"], B[i]["e2e_s_median"]) for i in sub]),
                "e2e_total_off_s": sum(A[i]["e2e_s_median"] for i in sub),
                "e2e_total_on_s": sum(B[i]["e2e_s_median"] for i in sub),
            }

        st = stats(ids)
        cached = [B[i].get("cached_prefix_tokens") for i in ids if B[i].get("cached_prefix_tokens") is not None]
        frac = [B[i]["cached_prefix_tokens"] / B[i]["prompt_tokens"] for i in ids
                if B[i].get("cached_prefix_tokens") is not None and B[i]["prompt_tokens"]]
        newp = [B[i].get("new_prefill_tokens") for i in ids if B[i].get("new_prefill_tokens") is not None]
        same = [A[i]["output_token_ids"] == B[i]["output_token_ids"] for i in ids]
        row = {"mode": mode, "k": k, **st,
               "cached_prefix_tokens_median": median(cached) if cached else None,
               "cached_fraction_median": median(frac) if frac else None,
               "new_prefill_tokens_median": median(newp) if newp else None,
               "output_same_off_vs_on": sum(same) / len(same)}
        buckets = {}
        for i in ids:
            buckets.setdefault((A[i].get("meta") or {}).get("length_bucket", "?"), []).append(i)
        row["by_length_bucket"] = {bk: {x: stats(sub)[x] for x in ("n", "ttft_speedup", "e2e_speedup")}
                                   for bk, sub in sorted(buckets.items())}
        table.append(row)
        for i in ids:
            turns.append({
                "mode": mode, "k": k, "id": i,
                "length_bucket": (A[i].get("meta") or {}).get("length_bucket"),
                "prompt_tokens": A[i]["prompt_tokens"],
                "cached_prefix_tokens": B[i].get("cached_prefix_tokens"),
                "new_prefill_tokens": B[i].get("new_prefill_tokens"),
                "ttft_off_s": A[i]["ttft_s_median"], "ttft_on_s": B[i]["ttft_s_median"],
                "tpot_off_s": A[i]["tpot_s_median"], "tpot_on_s": B[i]["tpot_s_median"],
                "e2e_off_s": A[i]["e2e_s_median"], "e2e_on_s": B[i]["e2e_s_median"],
                "n_out_off": A[i]["n_out"], "n_out_on": B[i]["n_out"],
                "same_output": A[i]["output_token_ids"] == B[i]["output_token_ids"],
            })

    # 같은 캐싱 조건 안에서 spec 이 타겟 단독보다 빠른가 (캐싱이 드래프트 프리필 손해를 없애는지)
    spec_gain = []
    for name, res in (("off", off_res), ("on", on_res)):
        t = pick(res, "target_only")
        if not t:
            continue
        T = by_id(t)
        for s_ in [r for r in res if r["mode"] == "spec"]:
            S = by_id(s_)
            ids = [i for i in S if i in T]
            spec_gain.append({
                "cache": name, "k": s_["k"], "n": len(ids),
                "spec_ttft_overhead_s": median([S[i]["ttft_s_median"] - T[i]["ttft_s_median"] for i in ids]) if ids else None,
                "spec_tpot_speedup": _ratio_median([(T[i]["tpot_s_median"], S[i]["tpot_s_median"]) for i in ids]),
                "spec_e2e_speedup": _ratio_median([(T[i]["e2e_s_median"], S[i]["e2e_s_median"]) for i in ids]),
            })

    def f(x, nd=2):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    print("\n[캐싱 끔 → 켬 속도향상]  (끔 시간 ÷ 켬 시간, 턴별 비율의 중앙값. 1보다 크면 캐싱이 빠름)")
    print(" | ".join(["모드", "k", "턴수", "TTFT 끔→켬(s)", "TTFT배", "TPOT 끔→켬(ms)", "TPOT배",
                      "E2E 끔→켬(s)", "E2E배", "캐시비율", "출력같음"]))
    for t in table:
        print(" | ".join([t["mode"], str(t["k"]), str(t["n"]),
                          f"{f(t['ttft_off_s'])}→{f(t['ttft_on_s'])}", f(t["ttft_speedup"]) + "x",
                          f"{f(t['tpot_off_ms'], 1)}→{f(t['tpot_on_ms'], 1)}", f(t["tpot_speedup"]) + "x",
                          f"{f(t['e2e_off_s'])}→{f(t['e2e_on_s'])}", f(t["e2e_speedup"]) + "x",
                          f(t["cached_fraction_median"]), f(t["output_same_off_vs_on"])]))
        for bk, v in t["by_length_bucket"].items():
            print(f"     └ {bk}: {v['n']}턴, TTFT {f(v['ttft_speedup'])}x, E2E {f(v['e2e_speedup'])}x")
    if spec_gain:
        print("\n[같은 캐싱 조건에서 추측 디코딩 ÷ 타겟 단독]")
        print(" | ".join(["캐싱", "k", "턴수", "TTFT 증가(s)", "TPOT배", "E2E배"]))
        for g in spec_gain:
            print(" | ".join([g["cache"], str(g["k"]), str(g["n"]), f(g["spec_ttft_overhead_s"]),
                              f(g["spec_tpot_speedup"]) + "x", f(g["spec_e2e_speedup"]) + "x"]))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_cache.json").write_text(json.dumps(
        {"off_dir": args.off_dir, "on_dir": args.on_dir, "off_vs_on": table, "spec_vs_target": spec_gain},
        ensure_ascii=False, indent=1), encoding="utf-8")
    flat = [{k_: v for k_, v in t.items() if k_ != "by_length_bucket"} for t in table]
    for name, rows in (("summary_cache.csv", flat), ("summary_cache_turns.csv", turns)):
        if not rows:
            continue
        cols = list(rows[0])
        with open(out / name, "w", encoding="utf-8") as fp:
            fp.write(",".join(cols) + "\n")
            for r in rows:
                fp.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    print(f"[saved] {out / 'summary_cache.json'}, summary_cache.csv, summary_cache_turns.csv")


def main(argv=None):
    p = argparse.ArgumentParser(description="vLLM 추측 디코딩 α·c 예비 측정")
    p.add_argument("--mode", required=True, choices=["target_only", "draft_only", "spec", "summarize", "compare", "cache_compare"])
    p.add_argument("--target", default="Qwen/Qwen3-4B")
    p.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    p.add_argument("--k", type=int, default=5, help="γ: 한 번에 제안하는 드래프트 토큰 수")
    p.add_argument("--prompts", default=None, help="JSONL 경로 (없으면 내장 예시)")
    p.add_argument("--limit", type=int, default=None, help="프롬프트 개수 제한")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=128)
    p.add_argument("--max-model-len", dest="max_model_len", type=int, default=4096)
    p.add_argument("--dtype", default="auto", help="T4는 half, A100·H100은 bfloat16")
    p.add_argument("--gpu-mem", dest="gpu_mem", type=float, default=0.85)
    p.add_argument("--tp", type=int, default=1, help="타겟 tensor parallel (GPU 2장이면 2)")
    p.add_argument("--draft-tp", dest="draft_tp", type=int, default=None)
    p.add_argument("--prefix-caching", dest="prefix_caching", action="store_true", help="C0b용: 프리픽스 캐싱 켜기. 턴마다 직전 턴 입력까지만 캐시된 상태에서 측정")
    p.add_argument("--enforce-eager", dest="enforce_eager", action="store_true", help="CUDA graph 끄기 (오류 날 때만)")
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--out-dir", dest="out_dir", default="results")
    p.add_argument("--label", default=None)
    p.add_argument("--other-dir", dest="other_dir", default=None, help="compare: 압축판 결과 폴더 (--out-dir 은 원본판)")
    p.add_argument("--off-dir", dest="off_dir", default=None, help="cache_compare: 캐싱 끔 결과 폴더")
    p.add_argument("--on-dir", dest="on_dir", default=None, help="cache_compare: 캐싱 켬 결과 폴더")
    p.add_argument("--yarn", action="store_true", help="Qwen3 YaRN 131K 문맥 켜기 (--max-model-len 131072 와 함께)")
    p.add_argument("--hf-overrides", dest="hf_overrides", default=None, help="모델 설정 덮어쓰기 JSON (고급)")
    args = p.parse_args(argv)

    if args.warmup >= args.reps:
        p.error("--warmup 은 --reps 보다 작아야 합니다")
    if args.mode == "summarize":
        summarize(args)
    elif args.mode == "cache_compare":
        if not (args.off_dir and args.on_dir):
            p.error("cache_compare 에는 --off-dir 과 --on-dir 이 필요합니다")
        cache_compare(args)
    elif args.mode == "compare":
        if not args.other_dir:
            p.error("compare 에는 --other-dir 이 필요합니다")
        compare(args)
    else:
        run_measure(args)


if __name__ == "__main__":
    main()
