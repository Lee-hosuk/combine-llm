# combine-llm

코딩 에이전트 프롬프트 압축과 추측 디코딩을 결합한 경량 LLM 추론 가속 연구의 실험 코드입니다.
현재는 압축이 없는 기준 조건(C0a, C0b, C2)의 측정 코드와 결과를 담고 있습니다.

## 실험 조건

| 조건 | 내용 | 실행 |
|---|---|---|
| C0a | Qwen3-32B 단독, 프리픽스 캐싱 끔 | `PREFIX=0 OUT=results_fp16 bash run.sh target` |
| C0b | Qwen3-32B 단독, 프리픽스 캐싱 켬 | `bash run.sh target` |
| C2 | Qwen3-4B → Qwen3-32B 추측 디코딩(γ=3, 5), 압축 없음 | `bash run.sh draft && bash run.sh spec` |

`bash run.sh all`은 타겟 단독, 드래프트 단독, 추측 디코딩, 요약, 결과 압축을 차례로 실행합니다.
캐싱을 끈 추측 디코딩까지 측정하려면 `PREFIX=0 OUT=results_fp16 bash run.sh all`을 실행하세요.

`run.sh`는 C0b 실행 꾸러미의 파일입니다. `PREFIX=0` 실행은 같은 설정을 재현하는 방법이며, C0a 결과를 만든 실행 스크립트 자체는 별도 꾸러미에 있어 이 저장소에 포함되어 있지 않습니다.

```bash
cd speculative_decoding
pip install vllm==0.31.0
bash run.sh smoke
bash run.sh all
```

필요 환경과 설정, 문제 해결 방법은 [speculative_decoding/README.md](speculative_decoding/README.md)를 참고하세요.

## 파일

| 경로 | 내용 |
|---|---|
| `speculative_decoding/spec_bench.py` | vLLM 측정 스크립트 |
| `speculative_decoding/run.sh` | 실행 스크립트 |
| `speculative_decoding/prep_swe.py` | 턴 추출 전처리 스크립트 |
| `speculative_decoding/swe_turns.jsonl` | 측정 입력 200턴 |
| `speculative_decoding/swe_meta.json` | 입력 통계와 원본 파일 sha256 |
| `results/` | 측정 결과. 조건별 설명은 [results/README.md](results/README.md) 참고 |

`swe_turns.jsonl`의 sha256은 `374f3986141a1314a602f63e4385c57df226a463d989329f513c9bd43b4d788c`입니다.

## 데이터와 모델 출처

- 입력 데이터는 [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)(CC-BY-4.0)에서 추출했습니다. `resolved == 1` 기록 50개에서 기록당 4턴을 골랐고, 프롬프트 형태로 잘라 낸 것 외에 압축은 하지 않았습니다.
- 모델은 [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)와 [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)(Apache-2.0)를 사용합니다.
