# 스크립트 색인

영역별 폴더로 나뉜다. 별도 표시가 없으면 보드(Orange Pi 6 Plus)에서 실행한다. 원본 출력은 저장소 루트의 `runs/`(git 제외)에 쌓이고, 남길 요약만 `docs/*/results/`에 복사한다. 문서 색인은 ../docs/README.md에 있다.

## kvswap/: KV cache 디스크 오프로딩

| 파일 | 내용 |
|---|---|
| `run_stage1.sh` | 단일 실행. `base`, `file`, `select` 모드, cgroup 메모리 제한, `iostat`와 `mpstat` 기록 |
| `sweep_stage1.sh` | 측정 행렬 일괄 실행(B0, B1, B2, 프리페치, G, 선택량 스윕) |
| `summarize_stage1.py` | 결과 폴더를 `summary.csv`, `summary_mean.csv`로 요약 |
| `make_kvswap_adapter.py` | 저랭크 어댑터(SVD) 생성. torch, transformers 필요. **노트북에서 실행** |
| `niah_kvswap.py` | 선택 방식별 NIAH 축소판 품질 비교 |

llama.cpp 패치는 [../patches/](../patches/)에 있다(01~05).

## sweep/: 주파수 스윕과 자원 간 상호작용

| 파일 | 내용 |
|---|---|
| `sweep_cpufreq.py` | A720 주파수를 바꿔 가며 CPU 실행, GPU 실행, 메모리 대역폭 측정 |
| `sweep_gpufreq.py` | GPU 주파수를 바꿔 가며 같은 것을 측정(GPU 메모리 대역폭은 `clpeak`) |
| `sweep_corun.py` | CPU decode와 GPU decode를 동시에 실행해 서로의 슬로다운 측정 |
| `sweep_npu.py` | NPU 주파수 단독, CPU/GPU 주파수의 NPU 영향, NPU×CPU, NPU×GPU, 세 자원 동시 실행, 고정 작업량 열 면적(`--with-heat`). NPU 실행 명령은 `--npu-cmd`로 받는다(`NPU_RESULT` 계약, 래퍼는 `npu/llm/npu_llm_run.py`). 반복 추론은 `--npu-persistent`. `--analyze`로 `summary.csv`(슬로다운 포함) 재생성. 자원 사용률 컬럼 포함. 11/20 축소 격자로 보드 시험 완료. 계획은 ../docs/sweep/ri-corun-boost-plan.md |
| `util_sampler.py` | 시간 창의 자원 사용률: CPU busy/iowait(`/proc/stat`), 디스크 util·MB/s(`/proc/diskstats`), GPU busy(Mali runtime PM), CPU 쪽 버스 대역폭(perf `bus_access`, `perf_event_paranoid<=0` 필요). `sweep_npu.py`, `joint/thermal_ab.py`가 사용 |
| `ri_cpu_gpu_proxy.py` | 11/13 스윕에서 CRAVE Fig. 7 형식의 CPU×GPU RI 행렬을 그림. `--npu-csv-*`로 NPU 칸을 추가 |
| `membw_probe.c`, `membw_calib.sh` | 알려진 양을 스트리밍해서 PMU 카운터를 대역폭으로 보정 |
| `sweep_ri.sh` | 두 자원 주파수 격자 스윕(RI 행렬용, 보드 미실행). 기본 실행기는 `../kvswap/run_stage1.sh` |
| `ri.py`, `ri_post.py` | RI 분석과 실행 폴더 후처리(`sweep_ri.sh` 결과용) |
| `freq_check.sh`, `freq_steps.sh` | 주파수 고정 동작 확인(root)과 CPU 주파수 단계 탐색 |

`sweep_cpufreq.py`, `sweep_gpufreq.py`, `sweep_corun.py`는 서로 import하므로 같은 폴더에 있어야 한다. 자세한 사용법은 각 파일 머리말과 [결과 문서](../../results/docs/cpu-gpu-memory-sweeps-20261113.md)에 있다.

```
python3 scripts/sweep/sweep_cpufreq.py --dry-run     # 계획만 출력
python3 scripts/sweep/sweep_cpufreq.py               # 전체 스윕 (sudo 비밀번호는 필요할 때 한 번)
SUDO_PASS=... python3 scripts/sweep/sweep_gpufreq.py # 비대화형
```

## joint/: KV swap과 주파수를 함께

| 파일 | 내용 |
|---|---|
| `thermal_ab.py` | naive 대비 KV swap과 주파수 조절의 열, 속도, 품질 비교. `../sweep/sweep_cpufreq.py`를 import하고 `../kvswap/run_stage1.sh`를 호출한다. 설정: `naive`, `freq_<MHz>`, `kv_default`, `kv_<MHz>`, `phase_<MHz>`, `kvphase_<MHz>`(phase는 첫 생성 토큰부터 고정). 동시 실행 설정 `co_g<GPU|def>_c<A720|def>`(GPU 작업 1개 + CPU 작업 `--co-cpu-reqs`개). 부분 오프로드 설정 `p<N>_g<GPU|def>_c<A720|def>`(파일럿에서 속도 비정상으로 격자는 미실행). GPU 설정: `gnaive`, `g<GPU MHz>_c<A720 MHz|def>`(Vulkan 빌드, KV swap 없음). 옵션: `--resume <dir>`(중단된 실행 이어하기), `--only`, `--ttft-slack`. 계획과 판정 규칙은 ../docs/joint/kvswap-x-sweep-plan.md |
| `governor_lite.py` | 0.5초마다 상태(CPU prefill/decode, GPU 작업)를 감지해 사전 계산 테이블대로 A720 주파수를 바꾸는 데몬(GPU 작업이 돌 때만 A720 1.5 GHz). `thermal_ab.py`의 `ck_<ram\|swap>_gov` 설정이 쓰고, 단독 실행도 된다. 보드 미실행. 설명은 [../docs/joint/governor-lite.md](../../results/docs/governor-lite.md) |
| `gov_replay.py` | governor 실행과 기준선을 나란히 재생하는 MP4/GIF와 정지 그림(영상용). matplotlib 필요(보드: `/mnt/ssd/viz-venv`) |
| `intro_video.py` | 실험 소개 영상: 측정값으로 그린 장면들과 `gov_replay.py` 영상을 이어 붙인 MP4 (docs/joint/intro-video.md) |
| `corun_kvswap.py` | CPU decode(RAM KV 또는 KV swap, 7K 컨텍스트) 한 프로세스가 도는 동안 GPU decode(`llama-bench`, Vulkan)를 구간별로 끼워 넣어 서로의 슬로다운과 A720 대역폭(perf, `SUDO_PASS` 필요)을 잰다. CPU 속도는 `out.txt`를 FIFO로 읽어 토큰마다 시각을 기록한다. `../sweep/sweep_cpufreq.py`, `../sweep/sweep_corun.py`를 import하고 `../kvswap/run_stage1.sh`를 호출한다. 옵션: `--resume`, `--analyze`, `--let-finish`. 보드 미실행. 계획은 ../docs/joint/corun-kvswap-plan.md |

## npu/: NPU

| 파일 | 내용 |
|---|---|
| `npu_freq.sh`, `npu_smoke.py` | 컴파일된 `.cix` 모델의 지연을 NPU 주파수별로 측정(더미 입력) |
| `llm/check_board.sh` | NPU LLM 경로 A, B 준비 상태를 읽기 전용으로 점검 |
| `llm/export_qwen_onnx.py` | Qwen 계열을 prefill, decode 정적 ONNX로 내보냄. **x86 PC에서 실행** |
| `llm/make_cixbuild_cfg.py` | 위 결과로 `cixbuild` 설정 생성 |
| `llm/npu_llm_run.py` | Qwen2.5-0.5B prefill/decode `.cix`를 NOE로 실행하고 `NPU_RESULT`를 출력(`sweep_npu.py`용). decode는 argmax 토큰을 되먹이고 KV 창을 한 행씩 민다. `--persistent`는 죽을 때까지 반복 |
| `llm/npu_freq_run.sh` | 임의 명령을 NPU 주파수별로 실행하고 시간, 온도를 CSV로 기록 |

경로 A의 설명은 ../docs/npu/npu-llm-plan.md에 있다.
