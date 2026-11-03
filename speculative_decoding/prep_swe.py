#!/usr/bin/env python3
"""prep_swe.py — SWE-rebench OpenHands 기록 → 턴 단위 프롬프트 JSONL (압축 없음)

입력: nebius/SWE-rebench-openhands-trajectories 의 trajectories.parquet (약 2GB), tools.json
출력: swe_turns.jsonl (spec_bench.py --prompts 로 바로 사용), swe_meta.json (통계·재현 정보)

처리 순서
  1) resolved == 1 (성공 기록)만 대상. trajectory_id 만 먼저 읽어 seed 고정으로 기록 N개 추출
     (2GB 파일을 한 번에 메모리에 올리지 않도록 2단계로 읽음)
  2) 기록마다 assistant 턴을 앞→뒤 구간 K개로 나누고 구간마다 1개씩 무작위 선택
  3) 선택한 assistant 턴 t 에 대해 입력 = trajectory[:t] (그 턴 직전까지). t 이후 정보는 넣지 않음(누출 금지)
  4) tool_calls.arguments 문자열 → JSON 객체 (데이터 카드 권장), content None → ""
  5) tools.json 을 모든 샘플의 tools 로 넣음 (채팅 템플릿에 도구 정의 포함)
  6) Qwen3 토크나이저 + 채팅 템플릿(non-thinking)으로 실제 입력 토큰 수 계산, --max-input 초과 턴은 제외하고 기록
  OpenHands 출력 형식은 손대지 않음 (압축 없는 원본 조건)
"""
import argparse
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

REPO = "nebius/SWE-rebench-openhands-trajectories"
ROLE_FIELDS = {
    "system": ["role", "content"],
    "assistant": ["role", "content", "tool_calls"],
    "user": ["role", "content"],
    "tool": ["role", "content", "name", "tool_call_id"],
}
BUCKETS = [(0, 16384, "<16K"), (16384, 32768, "16K-32K"), (32768, 65536, "32K-64K"), (65536, 10 ** 9, ">=64K")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def as_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # [{"type": "text", "text": ...}, ...]
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def normalize_message(msg):
    role = msg.get("role")
    m = {k: msg.get(k) for k in ROLE_FIELDS.get(role, ["role", "content"])}
    m["content"] = as_text(m.get("content"))
    if role == "assistant":
        calls = m.get("tool_calls") or None
        if calls:
            fixed = []
            for c in calls:
                c = dict(c)
                fn = dict(c.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        pass  # 그대로 문자열 유지
                c["function"] = fn
                fixed.append(c)
            m["tool_calls"] = fixed
        else:
            m.pop("tool_calls", None)
    if role == "tool":
        for k in ("name", "tool_call_id"):
            if m.get(k) is None:
                m.pop(k, None)
    return m


def choose_turns(n_messages_roles, k, rng):
    """assistant 위치들을 앞→뒤 k 구간으로 나눠 구간마다 하나씩 고른다."""
    idx = [i for i, r in enumerate(n_messages_roles) if r == "assistant" and i > 0]
    if not idx:
        return []
    k = min(k, len(idx))
    chosen = []
    for s in range(k):
        seg = idx[len(idx) * s // k: len(idx) * (s + 1) // k]
        if seg:
            chosen.append((s, rng.choice(seg)))
    return chosen


def bucket(n):
    for lo, hi, name in BUCKETS:
        if lo <= n < hi:
            return name
    return BUCKETS[-1][2]


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs):
    return {} if not xs else {"n": len(xs), "min": min(xs), "p50": pct(xs, .5), "p90": pct(xs, .9),
                              "max": max(xs), "mean": round(statistics.mean(xs), 1)}


def build_samples(record, tools, k, seed, tokenizer, to_token_ids, max_input):
    traj = [normalize_message(m) for m in record["trajectory"]]
    roles = [m["role"] for m in traj]
    rng = random.Random(f"{seed}:{record['trajectory_id']}")
    kept, skipped = [], []
    n_assistant = sum(1 for r in roles if r == "assistant")
    for seg, t in choose_turns(roles, k, rng):
        prompt = traj[:t]
        assert all(i < t for i in range(len(prompt)))  # 누출 방지: t 이후 메시지 없음
        item = {"id": f"{record['trajectory_id']}__t{t}", "messages": prompt, "tools": tools}
        n_tok = len(to_token_ids(tokenizer, item))
        ref = traj[t]
        item["meta"] = {
            "trajectory_id": record["trajectory_id"], "instance_id": record.get("instance_id"),
            "repo": record.get("repo"), "turn_index": t, "segment": seg, "n_segments": k,
            "assistant_turn_no": sum(1 for r in roles[:t] if r == "assistant") + 1,
            "n_assistant_turns": n_assistant, "prompt_tokens": n_tok, "length_bucket": bucket(n_tok),
            "reference_action": {"content": ref.get("content", ""), "tool_calls": ref.get("tool_calls")},
        }
        (kept if n_tok <= max_input else skipped).append(item)
    return kept, skipped


def iter_records(parquet_path, chosen_ids, batch_size=32):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    cols = [c for c in ["trajectory_id", "instance_id", "repo", "resolved", "trajectory"] if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
        ids = batch.column("trajectory_id").to_pylist()
        hit = [i for i, x in enumerate(ids) if x in chosen_ids]
        if not hit:
            continue
        sub = batch.take(hit).to_pylist()
        for row in sub:
            yield row


def pick_records(parquet_path, n, seed):
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path, columns=["trajectory_id", "resolved"])
    ids, res = t.column("trajectory_id").to_pylist(), t.column("resolved").to_pylist()
    ok = sorted(i for i, r in zip(ids, res) if r == 1)
    chosen = sorted(random.Random(seed).sample(ok, min(n, len(ok))))
    return chosen, {"total": len(ids), "resolved": len(ok)}


def main(argv=None):
    p = argparse.ArgumentParser(description="SWE-rebench OpenHands 기록 → 턴 단위 프롬프트 JSONL")
    p.add_argument("--parquet", default=None, help="trajectories.parquet 경로 (없으면 Hugging Face 에서 받음)")
    p.add_argument("--tools", default=None, help="tools.json 경로 (없으면 받음)")
    p.add_argument("--out-dir", dest="out_dir", default="swe_prompts")
    p.add_argument("--n-records", dest="n_records", type=int, default=50)
    p.add_argument("--turns-per-record", dest="k", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B", help="Qwen3 계열은 어휘가 같아 4B 토크나이저로 32B 길이를 셈")
    p.add_argument("--max-input", dest="max_input", type=int, default=131072 - 256,
                   help="입력 토큰 상한 (기본 131,072-256: YaRN 최대. 실제 측정 상한은 run.sh 의 MAXLEN 이 결정)")
    p.add_argument("--no-hash", dest="no_hash", action="store_true", help="parquet sha256 계산 생략")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    import spec_bench as sb

    if not args.parquet or not args.tools:
        from huggingface_hub import hf_hub_download
        args.parquet = args.parquet or hf_hub_download(REPO, "trajectories.parquet", repo_type="dataset")
        args.tools = args.tools or hf_hub_download(REPO, "tools.json", repo_type="dataset")
    tools = json.load(open(args.tools, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    chosen, counts = pick_records(args.parquet, args.n_records, args.seed)
    print(f"기록 {counts['total']}개 중 성공 {counts['resolved']}개 → {len(chosen)}개 선택 (seed {args.seed})")
    chosen_set = set(chosen)

    kept, skipped, seen = [], [], 0
    for rec in iter_records(args.parquet, chosen_set):
        if rec.get("resolved", 1) != 1:
            continue
        a, b = build_samples(rec, tools, args.k, args.seed, tok, sb.to_token_ids, args.max_input)
        kept += a
        skipped += b
        seen += 1
        if seen % 10 == 0:
            print(f"  {seen}/{len(chosen)} 기록 처리, 턴 {len(kept)}개")
    kept.sort(key=lambda it: it["id"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "swe_turns.jsonl", "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    toks = [it["meta"]["prompt_tokens"] for it in kept]
    meta = {
        "source": REPO, "license": "CC-BY-4.0", "filter": "resolved == 1",
        "records_total": counts["total"], "records_resolved": counts["resolved"],
        "records_chosen": len(chosen), "records_processed": seen,
        "turns_per_record": args.k, "seed": args.seed, "tokenizer": args.tokenizer,
        "chat_template": "Qwen3, enable_thinking=False, tools=tools.json",
        "max_input_tokens": args.max_input, "turns_kept": len(kept), "turns_over_max_input": len(skipped),
        "prompt_tokens": dist(toks),
        "length_buckets": dict(Counter(it["meta"]["length_bucket"] for it in kept)),
        "over_32768_minus_256": sum(t > 32768 - 256 for t in toks),
        "segment_counts": dict(Counter(it["meta"]["segment"] for it in kept)),
        "parquet_sha256": None if args.no_hash else sha256(args.parquet),
        "chosen_trajectory_ids": chosen,
    }
    (out / "swe_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"턴 {len(kept)}개 저장 (상한 {args.max_input} 초과 {len(skipped)}개 제외)")
    if toks:
        d = meta["prompt_tokens"]
        print(f"입력 토큰: 중앙값 {d['p50']:.0f}, p90 {d['p90']:.0f}, 최대 {d['max']}")
        print(f"길이 구간: {meta['length_buckets']} | 32K 설정에서 건너뛸 턴 {meta['over_32768_minus_256']}개")
    print(f"[saved] {out / 'swe_turns.jsonl'}, {out / 'swe_meta.json'}")


if __name__ == "__main__":
    main()
