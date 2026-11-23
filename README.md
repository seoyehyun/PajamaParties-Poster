# Optimizing Cross-Resource Coordination for Thermal-Aware On-device LLM Inference

**파자마파티즈** (김휘영, 김지호, 서예현, 임예랑) · 연세대학교 컴퓨터종합설계 2026-2 · 지도 정진규 교수님, 조교 하승원

포스터 지면에 담기 어려운 **코드, 상세 실험 결과**를 정리한 저장소입니다.

## 연구 한 줄 요약

온디바이스 LLM 추론에서는 CPU, GPU, NPU가 공유 메모리를 두고 경쟁합니다. 이 연구는 Orange Pi 6 Plus(CIX P1)에서 자원별 주파수와 **KV cache 오프로딩**(KV를 NVMe로 내림)이 속도와 발열에 주는 영향을 측정하고, **속도 제약을 지키면서 발열이 가장 낮은 운용 조건**을 찾았습니다.

## 핵심 결과

포스터의 Figure 번호와 같습니다. 자세한 수치와 조건은 오른쪽 문서에 있습니다.

| | 결과 | 상세 |
|---|---|---|
| **Fig. 1** 주파수 민감도 | prefill은 CPU, GPU, NPU 모두 주파수에 거의 비례합니다. decode는 CPU가 약 1.5 GHz 이후 포화하고(메모리 읽기 약 39~44 GB/s), GPU는 거의 비례하며, NPU는 둔감합니다(주파수 3배에 prefill 2.49배, decode 1.56배). | [cpu-gpu-memory-sweeps](results/docs/cpu-gpu-memory-sweeps-20261113.md), [results-20261121](results/docs/results-20261121.md) 4-1절 |
| **Fig. 2** 자원 간 상호작용 | 단독 실행에서는 자원이 서로 독립입니다. 동시 실행의 decode에서 CPU 주파수가 올라가면 GPU가 최대 **+74.6%** 느려집니다. prefill 동시 실행은 모든 조합에서 간섭이 0.5% 이내입니다. NPU는 CPU와 같이 돌 때 최대 +16.6% 느려지는데(원인은 NPU가 아니라 호스트 쪽 메모리 경합), GPU와는 거의 간섭하지 않습니다. | [results-20261121](results/docs/results-20261121.md) 4~5절 |
| **Fig. 3** KV 오프로딩 | CPU의 DRAM 읽기가 약 절반(39~44 → 20~26 GB/s)으로 줄고, decode는 같거나 최대 11% 빠릅니다(2.2 GHz에서 73.6 → 65.6 ms/token). 동시 실행 GPU 슬로다운은 컨텍스트가 길수록 크게 줄어 **11K에서 +87% → +17%**입니다(3K +43 → +24%, 7K +64 → +19%). | [results-20261121](results/docs/results-20261121.md) 1, 6절, [corun-kvswap-ctx](results/docs/corun-kvswap-ctx-20261119.md) |
| **Fig. 4** 발열 | CPU 전용에서 decode만 1.5 GHz로 낮추면 TTFT는 그대로이고 열 면적이 12.6% 줄어듭니다. CPU와 GPU 동시 실행에서 **KV 오프로딩 + decode 1.5 GHz**는 속도 제약을 지키며 열 면적을 **19%** 줄입니다(CPU 작업 TPOT 0.97배, GPU 작업 0.79배, TTFT 1.05배). | [thermal-ab-phase](results/docs/thermal-ab-phase-20261114.md), [results-20261121](results/docs/results-20261121.md) 3절 |
| **동적 관리** | 사전 계산 테이블을 실행 중에 적용하는 `governor-lite`를 구현했습니다. CPU 단계(prefill, decode)와 GPU 작업을 감지해 A720 주파수를 바꾸고, 같은 작업량의 고정 설정(decode 1.5 GHz)과 같은 결과를 냈습니다. KV 오프로딩만 쓸 때보다 첫 토큰 이후 열이 36% 낮습니다. | [governor-lite](results/docs/governor-lite.md), [precomputed-table](results/docs/precomputed-table.md) |

**부가 결과.** 작은 모델(Qwen2.5-0.5B)에서 KV 오프로딩은 컨텍스트 256토큰에서는 손해(TPOT 1.13배)이고 4K에서 0.88배, 16K에서 0.43배입니다. 손익분기점은 256과 4K 사이입니다([results-20261121](results/docs/results-20261121.md) 6b절).

## 저장소 구성

| 폴더 | 내용 |
|---|---|
| [results/](results/) | 상세 실험 결과. 문서는 `results/docs/`, 원본 요약 CSV와 그림은 `results/csv/`. 문서 색인은 [results/README.md](results/README.md) |
| [code/](code/) | 실험과 분석 스크립트, llama.cpp용 KV 오프로딩 패치 01~05. 사용법은 [code/README.md](code/README.md) |

## 실험 환경

| 항목 | 내용 |
|---|---|
| 보드 | Orange Pi 6 Plus (CIX P1: A720 ×8 + A520 ×4, Mali GPU, NPU), RAM 31.7 GB, 커널 6.6.89-cix |
| 추론 엔진 | CPU, GPU: llama.cpp (Vulkan GPU `-ngl 99`). NPU: CIX NOE(`.cix`) |
| 모델 | CPU, GPU: Qwen3-1.7B Q4_K_M. NPU: Qwen2.5-0.5B int8 (**모델이 달라서 NPU 결과는 자기 단독 실행 대비 비율로만 비교**) |
| KV 오프로딩 | KVSwap을 단순화해 llama.cpp에 구현(저랭크 키 예측기로 고른 KV 그룹만 NVMe에서 direct I/O로 읽음). 패치는 [code/patches/](code/patches/) |
| 측정 | TTFT(prefill 시간), TPOT, DRAM 읽기 대역폭(A720 PMU `l2d_cache_refill`), 온도, **열 면적**(온도 상승 × 시간) |

**"최적"의 기준.** 보드에 전력 센서가 없어서 에너지를 직접 재지 못합니다. 그래서 **열 면적(가장 뜨거운 열 존의 온도 상승을 식는 꼬리까지 시간으로 적분한 값)** 을 에너지 대리 지표로 쓰고, "TPOT과 TTFT가 기준선의 1.10배 이내, 품질 0.93 이상"을 지키는 설정 중 열 면적이 가장 낮은 것을 최적으로 봅니다.

## 재현

보드(Orange Pi 6 Plus)와 llama.cpp 빌드(`b11278`, 커밋 `931351ea5`, `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9.2-a+i8mm+bf16+dotprod+fp16`)가 필요합니다. 패치 적용 순서와 실행 명령은 [code/README.md](code/README.md)에 있습니다. 모델 파일과 보드 접속 정보는 이 저장소에 없습니다.

## 한계

- 대부분의 값은 **반복 1~2회**입니다(`results-20261121`의 값은 n=1). 열 면적 판정의 노이즈 폭은 최소 5%로 잡았고, 요청 전체 열 면적의 차이가 작은 칸은 노이즈 안에 듭니다.
- **열 면적은 에너지가 아니라 대리 지표**입니다. 열 존의 온도 해상도는 1 °C라서 NPU의 발열은 측정되지 않았습니다.
- 모델 하나(Qwen3-1.7B), 컨텍스트 몇 점, GPU와 A720 주파수 일부 점만 측정했습니다. NPU는 다른 모델(0.5B)입니다.
- KV 오프로딩은 CPU 프로세스에만 구현되어 있고, GPU와 한 프로세스에서 같이 쓰면 세그폴트가 납니다. 3K 이하의 짧은 컨텍스트에서는 이득이 없거나 손해입니다.
- KV 오프로딩의 품질(0.945)은 NIAH 점검 값을 판정에 입력한 것이고, 모든 설정에서 다시 재지는 않았습니다.
- 메모리 주파수는 이 보드에서 직접 바꿀 수 없어서 메모리 축은 PMU 대역폭으로만 간접 확인했습니다.

## 선행 연구

CRAVE (EuroSys '25), EnerInfer, KVSwap (MobiSys '26)
