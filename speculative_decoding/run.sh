#!/usr/bin/env bash
# run.sh — SWE-rebench 턴 단위 추측 디코딩 측정 (압축 없음, fp16, 프리픽스 캐싱 켬 = C0b)
#   fp16 캐싱 끔 꾸러미와 캐싱 여부만 다릅니다. 턴마다 '직전 턴 입력까지 캐시된' 상태에서 잽니다.
#
# 사용법
#   bash run.sh smoke     # 빠른 점검: 2턴, 1회. 환경·메모리 확인과 1턴당 시간 측정 (먼저 실행)
#   bash run.sh all       # 전체: 타겟 단독 → 드래프트 단독 → 추측 디코딩(γ별) → 요약 → 결과 압축
#   bash run.sh target | draft | spec | summary | pack   # 단계별 실행
#   bash run.sh cachecmp  # 캐싱 끔/켬 두 꾸러미 결과를 턴별로 짝지어 속도 차이 계산 (all 끝에 자동 시도)
#
# 설정은 아래 변수를 고치거나 실행할 때 앞에 붙여서 바꿉니다.  예) TP=2 LIMIT=40 bash run.sh all
set -euo pipefail
cd "$(dirname "$0")"

# ===== 설정 =====
TARGET=${TARGET:-Qwen/Qwen3-32B}     # 타겟 모델
DRAFT=${DRAFT:-Qwen/Qwen3-4B}        # 드래프트 모델 (타겟과 같은 Qwen3 어휘)
TP=${TP:-2}                          # 타겟 GPU 수 (bf16 실행과 같게 2)
DTYPE=${DTYPE:-float16}              # fp16 (bf16 실행과의 유일한 차이)
MAXLEN=${MAXLEN:-32768}              # 최대 문맥. 이를 넘는 턴은 자동으로 건너뛰고 기록됨
YARN=${YARN:-0}                      # 1 이면 YaRN 131K (MAXLEN=131072 와 함께, 빠른 점검으로 동작 확인 필수)
MAXTOK=${MAXTOK:-256}                # 턴당 최대 생성 토큰
REPS=${REPS:-3}                      # 프롬프트당 반복 횟수
WARMUP=${WARMUP:-1}                  # 앞에서 버리는 반복 횟수 (REPS 보다 작아야 함)
GAMMAS=${GAMMAS:-"3 5"}              # 추측 디코딩 γ (한 번에 제안하는 드래프트 토큰 수)
GPU_MEM=${GPU_MEM:-0.95}             # vLLM GPU 메모리 사용 비율 (bf16 실행과 같게)
LIMIT=${LIMIT:-}                     # 비우면 전체 턴, 숫자면 앞에서 N턴만
PROMPTS=${PROMPTS:-swe_turns.jsonl}
PREFIX=${PREFIX:-1}                  # 1 이면 프리픽스 캐싱 켬 (C0b), 0 이면 끔 (C0a)
OUT=${OUT:-results_fp16_cache}
PY=${PY:-python3}
# 캐싱 끔/켬 비교 (두 꾸러미를 같은 상위 폴더에 풀었을 때 기본 경로)
OFF_DIR=${OFF_DIR:-../swe_package_fp16/results_fp16}             # 캐싱 끔 결과 (옆 꾸러미)
ON_DIR=${ON_DIR:-$OUT}                                           # 캐싱 켬 결과 (이 꾸러미)

COMMON=(--target "$TARGET" --dtype "$DTYPE" --max-model-len "$MAXLEN" --max-tokens "$MAXTOK"
        --gpu-mem "$GPU_MEM" --prompts "$PROMPTS")
[[ "$YARN" == "1" ]] && COMMON+=(--yarn)
[[ "$PREFIX" == "1" ]] && COMMON+=(--prefix-caching)
[[ -n "$LIMIT" ]] && COMMON+=(--limit "$LIMIT")

mkdir -p "$OUT/logs"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/logs/run.log"; }

record_env() {
  {
    echo "date: $(date -Is)"; echo "host: $(hostname)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi 없음"
    $PY -c "import vllm, torch, transformers; print('vllm', vllm.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__)"
    echo "settings: TARGET=$TARGET DRAFT=$DRAFT TP=$TP DTYPE=$DTYPE MAXLEN=$MAXLEN YARN=$YARN MAXTOK=$MAXTOK REPS=$REPS WARMUP=$WARMUP GAMMAS='$GAMMAS' GPU_MEM=$GPU_MEM PREFIX=$PREFIX LIMIT=${LIMIT:-all}"
    echo "prompts: $(wc -l < "$PROMPTS") lines, sha256 $(sha256sum "$PROMPTS" | cut -d' ' -f1)"
  } > "$OUT/env.txt" 2>&1
  cat "$OUT/env.txt"
}

run_target() {
  log "타겟 단독 ($TARGET)"
  $PY spec_bench.py --mode target_only "${COMMON[@]}" --tp "$TP" --reps "$REPS" --warmup "$WARMUP" \
      --out-dir "$OUT" 2>&1 | tee "$OUT/logs/target_only.log"
}

run_draft() {
  log "드래프트 단독 ($DRAFT)"
  $PY spec_bench.py --mode draft_only "${COMMON[@]}" --draft "$DRAFT" --reps "$REPS" --warmup "$WARMUP" \
      --out-dir "$OUT" 2>&1 | tee "$OUT/logs/draft_only.log"
}

run_spec() {
  for k in $GAMMAS; do
    log "추측 디코딩 γ=$k ($DRAFT → $TARGET)"
    $PY spec_bench.py --mode spec "${COMMON[@]}" --draft "$DRAFT" --k "$k" --tp "$TP" --reps "$REPS" \
        --warmup "$WARMUP" --out-dir "$OUT" 2>&1 | tee "$OUT/logs/spec_k$k.log"
  done
}

run_summary() {
  log "요약"
  $PY spec_bench.py --mode summarize --out-dir "$OUT" 2>&1 | tee "$OUT/logs/summary.log"
}

run_cachecmp() {
  if ls "$OFF_DIR"/target_only__*.json >/dev/null 2>&1 && ls "$ON_DIR"/target_only__*.json >/dev/null 2>&1; then
    log "캐싱 끔/켬 속도 비교 (끔: $OFF_DIR, 켬: $ON_DIR)"
    $PY spec_bench.py --mode cache_compare --off-dir "$OFF_DIR" --on-dir "$ON_DIR" --out-dir "$OUT" \
        2>&1 | tee "$OUT/logs/cache_compare.log"
  else
    log "캐싱 끔/켬 비교 건너뜀: 다른 꾸러미 결과가 아직 없음 (끔: $OFF_DIR, 켬: $ON_DIR). 둘 다 끝난 뒤 'bash run.sh cachecmp' 실행"
  fi
}

run_pack() {
  local name="results_${DTYPE}_prefix${PREFIX}_$(date '+%Y%m%d_%H%M').tar.gz"
  tar czf "$name" "$OUT"
  log "결과 압축 완료: $name  ← 이 파일을 보내 주세요"
}

run_smoke() {
  local S="$OUT/smoke"; mkdir -p "$S"
  log "빠른 점검: 2턴, 1회 (타겟 단독 + 추측 디코딩 γ=3)"
  $PY spec_bench.py --mode target_only "${COMMON[@]}" --tp "$TP" --limit 2 --reps 1 --warmup 0 \
      --out-dir "$S" 2>&1 | tee "$OUT/logs/smoke_target.log"
  $PY spec_bench.py --mode spec "${COMMON[@]}" --draft "$DRAFT" --k 3 --tp "$TP" --limit 2 --reps 1 \
      --warmup 0 --out-dir "$S" 2>&1 | tee "$OUT/logs/smoke_spec.log"
  log "빠른 점검 끝. 위의 TPOT 와 각 턴 소요 시간으로 전체 시간을 가늠해 주세요 (README '예상 시간' 참고)"
}

case "${1:-}" in
  smoke)   record_env; run_smoke ;;
  all)     record_env; run_target; run_draft; run_spec; run_summary; run_cachecmp; run_pack ;;
  cachecmp) mkdir -p "$OUT/logs"; run_cachecmp; run_pack ;;
  target)  record_env; run_target ;;
  draft)   record_env; run_draft ;;
  spec)    record_env; run_spec ;;
  summary) run_summary ;;
  pack)    run_pack ;;
  *) echo "사용법: bash run.sh {smoke|all|target|draft|spec|summary|cachecmp|pack}"; exit 1 ;;
esac
