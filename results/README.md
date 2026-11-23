# 상세 실험 결과

- 실험 장비: Orange Pi 6 Plus
- 원본 출력(실행별 폴더, 로그): 용량 문제로 미포함
- 요약 CSV와 그림: `csv/`
- 날짜: 실제 날짜 기준 (보드 시계는 최대 11/10~11/11로 어긋나 있었음)

## 문서 (`docs/`)

| 문서 | 내용 |
|---|---|
| [results-20261121.md](docs/results-20261121.md) | **보고서용 보강 실험 총정리** (11/20~21, 모두 n=1)<br>- E5: KV swap × A720 decode 주파수<br>- E6: 사전 계산 테이블 검증<br>- E7: 동시 실행 + decode만 1.5 GHz<br>- NPU 단독, 교차, 세 자원 동시 실행<br>- E4: CPU×GPU 4×4<br>- E8: KV 오프로딩의 간섭 감소<br>- E9: 작은 모델 손익분기점<br>- 테이블 칸 재측정 |
| [governor-lite.md](docs/governor-lite.md) | 사전 계산 테이블을 실행 중에 적용하는 데몬<br>- 설계<br>- 보드 실행 결과 2회<br>- 고친 테이블로 재실행한 결과 |
| [precomputed-table.md](docs/precomputed-table.md) | 지배 자원 주파수 기준 사전 계산 테이블<br>- CPU 전용, GPU 전용, 동시 실행<br>- 조회표, 미측정 칸 |
| [cpu-gpu-memory-sweeps-20261113.md](docs/cpu-gpu-memory-sweeps-20261113.md) | CPU, GPU 주파수 스윕, 동시 실행, 메모리 대역폭 보정 |
| [thermal-ab-20261113.md](docs/thermal-ab-20261113.md) | 열 면적 A/B: naive 대 KV swap × 주파수 6개 설정 |
| [thermal-ab-phase-20261114.md](docs/thermal-ab-phase-20261114.md) | 단계별 주파수: prefill 기본, decode만 1.5 GHz |
| [thermal-ab-gpu-20261118.md](docs/thermal-ab-gpu-20261118.md) | GPU 전용 실행의 열 면적 (GPU 350~1000 MHz) |
| [thermal-ab-co-20261118.md](docs/thermal-ab-co-20261118.md) | CPU 작업과 GPU 작업 동시 실행의 열 면적 |
| [thermal-ab-ck-20261119.md](docs/thermal-ab-ck-20261119.md) | 긴 컨텍스트 CPU 작업의 KV 위치(RAM/swap)와 A720 주파수별 동시 실행 열 면적 |
| [corun-kvswap-20261119.md](docs/corun-kvswap-20261119.md) | CPU 쪽 KV swap의 GPU 슬로다운 감소 효과 (8개 조합) |
| [corun-kvswap-ctx-20261119.md](docs/corun-kvswap-ctx-20261119.md) | 컨텍스트 3K, 7K, 11K 반복, A720 읽기 대역폭(perf) 기록 |
| [kv-offload-plan.md](docs/kv-offload-plan.md) | KV 오프로딩 구현 계획과 현황 (패치 01~05 설계) |
| [kvswap-8k.md](docs/kvswap-8k.md), [kvswap-niah.md](docs/kvswap-niah.md), [kvswap-reuse.md](docs/kvswap-reuse.md), [kvswap-threads.md](docs/kvswap-threads.md), [kvswap-gpu-hybrid.md](docs/kvswap-gpu-hybrid.md) | KV 오프로딩 단독 측정<br>- 8K 컨텍스트<br>- 선택 방식별 NIAH 품질<br>- 재사용 버퍼<br>- 스레드 수<br>- GPU 혼합 시험 |
| [board-facts-20261113.md](docs/board-facts-20261113.md) | 보드 확인 사항: 메모리, PMU, 에너지 모델, 열, 주파수 제어 |

## 데이터 (`csv/`)

- 문서에서 링크한 요약 CSV와 그림
- 파일명의 날짜: 측정 묶음
- 주의: 묶음마다 열 면적 기준선이 달라, 열은 **한 묶음 안의 기준선 대비 비율로만** 사용

| 접두사 | 내용 |
|---|---|
| `e4_`, `e5_`, `e6_`, `e7_`, `e8_`, `e9_`, `npu_`, `kvphase_ab_`, `gov*` | `results-20261121.md`, `governor-lite.md`의 표 (11/20~21) |
| `thermal_ab_*` | 열 면적 A/B 설정별 요약, 실행별 값 |
| `corun_*`, `corun_kvswap_*` | 동시 실행 스윕, KV swap 간섭 |
| `cpufreq_sweep_*`, `gpufreq_sweep_*`, `cpu_sweep_*`, `gpu_sweep_*` | 주파수 스윕 |
| `membw_calib_*`, `freq_steps_*` | 메모리 대역폭 보정, 주파수 단계 확인 |
| `precomputed_table_*` | 사전 계산 테이블 |
| `ri_cpu_gpu_20261119.*` | CPU×GPU RI 행렬 그림 |
