# SWE-rebench 턴 단위 추측 디코딩 측정 (압축 없음) — **fp16 · 프리픽스 캐싱 켬 (C0b)**

> **fp16 캐싱 끔 꾸러미와 캐싱 여부만 다릅니다.** 프롬프트(sha256 `374f3986…`), 모델, fp16, TP=2, MAXLEN 32768, 최대 출력 256, REPS 3 / WARMUP 1, γ 3·5 는 모두 같습니다.
> 결과는 `results_fp16_cache/`, 압축 파일은 `results_float16_prefix1_날짜.tar.gz` 입니다.
>
> **캐시 상태를 어떻게 맞추나요?** 실제 에이전트에서는 직전 턴에 모델을 부를 때 계산한 입력이 캐시에 남아 있고, 이번 턴에는 그 뒤에 붙은 부분(직전 응답 + 도구 출력)만 새로 계산합니다. 이를 재현하려고 측정마다
> ① 캐시를 비우고(`reset_prefix_cache`) → ② 직전 턴 입력(마지막 assistant 메시지 직전까지의 대화)을 1토큰 생성으로 미리 계산해 두고(시간 측정 안 함) → ③ 현재 턴을 측정합니다.
> 첫 토큰 측정과 전체 생성 측정 앞에서 각각 ①②를 다시 하므로, 같은 프롬프트를 반복해도 캐시가 통째로 맞는 일이 없습니다.
> 턴마다 캐시된 토큰 수(`cached_prefix_tokens`)와 새로 계산한 토큰 수(`new_prefill_tokens`)가 결과에 기록됩니다.

종합설계프로젝트 「압축–드래프트 파이프라인 기반 경량 LLM 추론 가속」


이 꾸러미는 **명령 두 줄로 실행**되도록 만들었습니다. 아래 "실행 순서"만 따라 주시고, 끝나면 생성되는 `results_날짜.tar.gz` 파일 하나를 보내 주시면 됩니다.

---

## 1. 무엇을 측정하나요

코딩 에이전트가 GitHub 이슈를 해결한 실제 기록을 **턴 단위로 잘라**, 각 턴에서 모델이 다음 행동을 생성하는 시간을 잽니다. 압축은 하지 않은 원본 입력입니다.

| 단계 | 내용 | 얻는 값 |
|---|---|---|
| 타겟 단독 | Qwen3-32B 만으로 생성 | 기준 TTFT, TPOT, 전체 응답 시간 |
| 드래프트 단독 | Qwen3-4B 만으로 생성 | 드래프트 TPOT (시간 비 c 계산용) |
| 추측 디코딩 γ=3, 5 | 4B 가 γ개 제안 → 32B 가 한 번에 검증 (vLLM `draft_model`) | 채택률 α, 수락 길이, 속도 향상, 출력 일치 |

측정 원칙: greedy(temperature 0), Qwen3 non-thinking, 배치 1, **프리픽스 캐싱 켬(직전 턴까지 캐시)**, **fp16**. greedy 이므로 추측 디코딩 출력은 타겟 단독 출력과 같아야 하며, 스크립트가 이를 자동으로 확인합니다.

## 2. 들어 있는 파일

| 파일 | 내용 |
|---|---|
| `swe_turns.jsonl` | 측정할 턴(프롬프트). 한 줄 = `{"id", "messages", "tools", "meta"}` |
| `swe_meta.json` | 데이터 통계: 턴 수, 입력 길이 분포, 원본 파일 sha256 |
| `spec_bench.py` | 측정 스크립트 (vLLM 오프라인 API) |
| `prep_swe.py` | 전처리 스크립트 (참고용, 실행 불필요) |
| `run.sh` | 실행 스크립트 |

## 3. 필요한 환경

- Linux, NVIDIA GPU, Python 3.10 이상
- `pip install vllm==0.31.0` (팀이 코랩에서 검증한 버전. 설치가 안 되면 최신 버전도 괜찮고, 버전은 자동 기록됩니다)
- 디스크: 모델 다운로드 약 75GB (Qwen3-32B 약 66GB + Qwen3-4B 약 8GB). 위치를 바꾸려면 `export HF_HOME=/큰/디스크/경로`
- 인터넷: Hugging Face 에서 공개 모델(Apache-2.0)을 내려받습니다. 로그인 불필요

### GPU 메모리 (fp16 도 bf16 과 같은 2바이트라 동일, 배치 1, 가중치 + KV 캐시 계산값. 실행 중 임시 메모리 몇 GB 별도)

| 최대 문맥 (`MAXLEN`) | 4B + 32B 필요량 | 가능한 구성 예 |
|---|---|---|
| 32,768 (기본) | 약 87GB | H200 1장, 또는 80GB 2장 (`TP=2`) |
| 131,072 (`YARN=1`) | 약 124GB | H200 1장(빠듯함), 80GB 2장 (`TP=2`) |

80GB GPU **1장**뿐이라면 `TARGET=Qwen/Qwen3-14B` 로 바꿔 주세요 (4B+14B, 32K 기준 약 48GB).

## 4. 실행 순서

```bash
pip install vllm==0.31.0
bash run.sh smoke        # ① 빠른 점검 (처음에는 모델 약 75GB 다운로드 시간이 대부분)
bash run.sh all          # ② 전체 측정 → 마지막에 results_날짜.tar.gz 생성
```

- ① 에서 오류 없이 `[saved]` 가 두 번 나오면 ② 를 실행해 주세요.
- 설정은 실행 앞에 붙여 바꿉니다. 예: `TP=2 bash run.sh all`, `LIMIT=40 bash run.sh all`
- 오래 걸리므로 `tmux` 나 `nohup bash run.sh all > all.log 2>&1 &` 로 실행해 주시면 좋습니다.
- 중간에 끊기면 단계별로 다시 실행할 수 있습니다: `bash run.sh spec` → `bash run.sh summary` → `bash run.sh pack`

### 주요 설정 (`run.sh` 맨 위)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `TARGET` / `DRAFT` | Qwen3-32B / Qwen3-4B | 모델 |
| `TP` | 2 | 타겟을 나눠 올릴 GPU 수 (bf16 실행과 같게) |
| `PREFIX` | 1 | 1 이면 캐싱 켬(C0b), 0 이면 끔(C0a) |
| `DTYPE` | float16 | 연산 정밀도. `DTYPE=bfloat16` 으로 바꾸면 이전 꾸러미와 같아짐 |
| `MAXLEN` | 32768 | 최대 문맥. 넘는 턴은 자동으로 건너뛰고 결과에 기록 |
| `YARN` | 0 | 1 이면 131K 문맥 (`MAXLEN=131072` 와 함께). 아래 주의 참고 |
| `REPS` / `WARMUP` | 3 / 1 | 턴마다 3회 실행, 첫 1회는 버림 |
| `GAMMAS` | "3 5" | 추측 디코딩 γ |
| `LIMIT` | (전체) | 앞에서 N턴만 측정 (시간이 부족할 때) |

## 5. 예상 시간 계산

빠른 점검 로그의 각 턴 출력(`입력 N토큰, 출력 M토큰, TPOT T ms`)과 소요 시간으로 계산해 주세요.

> 전체 시간 ≈ (턴 수) × (턴 1회 시간) × `REPS` × 2(첫 토큰 측정 포함) × (1 + 1 + γ 개수) + 모델 불러오기(실행마다 수 분)

시간이 부족하면 `LIMIT` 이나 `REPS=2 WARMUP=1` 로 줄여 주세요. 줄인 값은 `env.txt` 에 자동 기록되어 저희가 분석할 때 반영합니다.

## 6. 돌려받을 것

`bash run.sh all` 마지막에 생기는 **`results_날짜.tar.gz` 하나**만 보내 주시면 됩니다. 안에는 다음이 들어 있습니다.
- `env.txt` : GPU, 드라이버, vLLM·torch 버전, 사용한 설정, 프롬프트 파일 sha256
- `*.json` : 단계별 원시 결과 (턴별 시간, 채택 카운터, 출력 토큰)
- `summary.csv`, `summary.json` : 요약표
- `logs/` : 실행 로그

## 7. 주의와 문제 해결

| 증상 | 해결 |
|---|---|
| `CUDA out of memory` | `GPU_MEM=0.95` 또는 `MAXLEN=24576`, 그래도 안 되면 `TP=2` 나 `TARGET=Qwen/Qwen3-14B` |
| 많은 턴이 `[skip]` | 입력이 `MAXLEN` 보다 긴 턴입니다. 정상 동작이며 결과에 기록됩니다 (32K 기준 건너뛸 턴 수는 `swe_meta.json` 의 `over_32768_minus_256`) |
| `YARN=1` 에서 오류 | vLLM 버전에 따라 설정 키 이름이 다를 수 있습니다. 기본값(32K)으로 측정해 주시면 됩니다. 특히 추측 디코딩에서 드래프트(4B)의 긴 문맥 지원은 아직 저희가 검증하지 못했습니다 |
| `[경고] 추측 디코딩 카운터가 0` | vLLM 버전에 따라 지표 이름이 다른 경우입니다. 로그를 그대로 보내 주세요 |
| `[경고] ... 수치 오버플로 의심` | Qwen3 는 bf16 으로 학습된 모델이라 fp16 에서는 큰 활성값이 범위(±65504)를 넘어 inf/NaN 이 될 수 있습니다. 출력이 비거나 같은 토큰만 반복되면 이 경고가 뜹니다. **빠른 점검에서 이 경고가 나오면 전체 실행 전에 로그를 보내 주세요** |
| `reset_prefix_cache 실패` 경고 | 이전 요청이 남아 캐시를 못 비운 경우입니다. 드물게만 나오면 무시해도 되고, 자주 나오면 로그를 보내 주세요 |
| CUDA graph 관련 오류 | `spec_bench.py` 명령에 `--enforce-eager` 를 추가 (모든 단계에 똑같이) |

## 캐싱 끔 / 켬 속도 비교 (C0a 대 C0b)

이 꾸러미는 짝 꾸러미(`swe_package_fp16` = 캐싱 끔, `swe_package_fp16_cache` = 캐싱 켬)와 함께 씁니다.

```
작업폴더/
├─ swe_package_fp16/          ← 캐싱 끔: bash run.sh all
└─ swe_package_fp16_cache/    ← 캐싱 켬: bash run.sh all
```

- **두 압축 파일을 같은 상위 폴더에 풀어 주세요.** 나중에 끝나는 쪽의 `run.sh all` 이 다른 쪽 결과를 찾아 비교를 자동으로 계산합니다.
- 두 꾸러미를 동시에 돌리지 말고 **하나씩 차례로** 실행해 주세요 (같은 GPU를 쓰면 시간이 섞입니다).
- 자동 비교가 건너뛰어졌으면 둘 다 끝난 뒤 아무 쪽에서나 `bash run.sh cachecmp` 를 실행하면 비교 후 다시 압축합니다. 폴더가 다르면 `OFF_DIR=... ON_DIR=... bash run.sh cachecmp`.

비교 결과 (결과 폴더에 저장, 압축 파일에 포함):

| 파일 | 내용 |
|---|---|
| `summary_cache.csv/json` | 모드별(타겟 단독 = C0a 대 C0b, 드래프트 단독, 추측 디코딩 γ=3·5) TTFT·TPOT·E2E 의 끔/켬 중앙값과 **속도향상(끔 ÷ 켬, 턴별 비율의 중앙값)**, 길이 구간별 속도향상, 캐시된 입력 비율, 끔/켬 출력 동일 비율. 그리고 같은 캐싱 조건에서 추측 디코딩 ÷ 타겟 단독 속도향상 |
| `summary_cache_turns.csv` | 턴별 원자료 (캐시된 토큰 수, 새로 계산한 토큰 수, 끔/켬 시간) — 그래프용 |

## 8. 데이터 출처

- 기록: [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) (CC-BY-4.0), `resolved == 1` 기록만 사용
- 전처리: 기록을 seed 0 으로 추출, 기록마다 assistant 턴을 앞→뒤 구간으로 나눠 구간별 1턴 선택. 입력 = 해당 턴 직전까지의 대화(이후 정보 미포함) + 도구 정의(`tools.json`), 압축 없음
- 모델: [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B), [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) (Apache-2.0)
