# 선행 연구

본 연구는 **CRAVE의 자원 상호작용 분석, EnerInfer의 LLM 에너지·열 관리, KVSwap의 KV cache 오프로딩**을 기반으로 한다. 이를 자원 상호작용 분석, KV cache 오프로딩 분석 및 단계 인식 주파수 관리로 확장하였으며, 본 문서는 각 연구와의 연계 및 구현 범위를 설명한다.

프로젝트 전체의 배경과 대표 결과는 [README](README.md), 실험별 근거는 [결과 안내](results/README.md)에서 볼 수 있다.

## 1. 사전 지식

LLM 추론은 입력 문맥을 일괄 처리하는 **prefill**과 토큰을 순차 생성하는 **decode**로 구분한다. 두 단계의 연산 및 메모리 접근 특성이 다르므로, 단계별로 적합한 주파수 설정을 검토해야 한다.

| 용어 | 이 문서에서의 의미 |
|---|---|
| **SoC** | CPU·GPU·NPU 같은 연산 장치와 메모리 제어 기능을 함께 묶은 칩 |
| **DVFS** | 동작 전압과 주파수를 조절해 성능·소비전력의 균형을 바꾸는 방법. 본 코드가 직접 설정하는 값은 주파수이다. |
| **거버너(governor)** | 사용률 등 현재 상태를 보고 동작 주파수를 정하는 프로그램·정책 |
| **자원 간 상호작용(RI)** | 한 장치의 주파수를 바꾸었을 때 다른 장치의 성능도 달라지는 관계 |
| **사전 계산 테이블** | 미리 시험한 설정을 표로 저장해 두고 실행 중에 현재 상태에 맞는 값을 고르는 방식 |
| **KV cache** | 앞서 처리한 문맥의 중간 계산 결과. 문맥이 길어질수록 저장량과 읽을 양이 커진다. |
| **어텐션** | 다음 토큰을 만들 때 앞선 문맥의 각 부분을 얼마나 참고할지 계산하는 과정 |
| **저랭크 키 예측기** | KV의 키 정보를 작은 차원으로 압축해, 읽어야 할 KV 그룹을 추정하는 장치 |
| **열 면적** | 실행 전보다 높아진 온도를 시간에 따라 누적한 값. 본 연구의 발열 비교 지표이며 에너지 직접 측정값은 아니다. |

## 2. 선행 연구 비교

| 연구 | 주요 연구 대상 | 본 연구와의 연계 |
|---|---|---|
| **CRAVE** · EuroSys 2025 | 자원 간 상호작용을 고려한 CPU·GPU·메모리 통합 DVFS | 자원 간 상호작용 분석, 사전 설정표 |
| **EnerInfer** · arXiv, 2026년 6월 | 응답 속도 제약을 고려한 LLM 에너지·열 관리 | prefill·decode 구분, 속도 조건을 둔 주파수 선택 |
| **KVSwap** · MobiSys 2026 | 긴 문맥의 KV cache 저장장치 오프로딩 및 선택적 접근 | 키 예측기, 그룹 단위 선택, KV 오프로딩·재사용 |
| **HeteroInfer** · SOSP 2025 | GPU·NPU 이기종 병렬 추론 | 공유 메모리와 이기종 장치 특성을 함께 분석하는 관점 |
| **Neuralink** · ASPLOS 2025 | 모델 가중치의 저장 배치 및 I/O 효율 개선 | 읽는 양뿐 아니라 저장 배치·읽기 단위도 중요하다는 관점 |
| **I/O-Cool** · HotStorage 2026 | 학습 과정의 I/O 구간을 활용한 온도 제어 | 주파수 외에 I/O 패턴도 열 관리에 영향을 줄 수 있다는 관점 |

중간 자료에서 중심적으로 소개한 연구는 CRAVE·EnerInfer·KVSwap이다. HeteroInfer·Neuralink는 관련 연구와 참고문헌으로, I/O-Cool은 I/O와 온도의 관계를 해석하는 추가 참고로 구분한다. 논문별 개선 수치는 장비·작업·비교 기준이 달라 본 실험의 개선율과 직접 비교하지 않는다.

## 3. 주요 기반 연구

### CRAVE — 자원 상호작용 기반 통합 DVFS

**[CRAVE: Analyzing Cross-Resource Interaction to Improve Energy Efficiency in Systems-on-Chip](https://ghose.cs.illinois.edu/papers/25eurosys_crave.pdf)**

Dipayan Mukherjee 외 · EuroSys 2025 · [DOI](https://doi.org/10.1145/3689031.3717498)

**연구 배경:** CPU·GPU가 같은 메모리를 쓰면 각 장치의 속도와 소비전력이 다른 자원의 설정에도 영향을 받는다. 자원별 사용률만 보고 주파수를 독립적으로 정하면 이 관계를 놓칠 수 있다.

**주요 방법:** 중간발표에서 소개한 지표는 RI·PPR·Utility다. RI는 한 자원의 주파수와 다른 자원의 성능 사이의 **Spearman 순위 상관계수**, PPR은 효율적인 주파수 조합을 판별하는 전력·성능 지표, Utility는 자원 간 결합을 반영해 지배 자원을 판단하는 지표다. 지배 자원은 현재 부하에서 성능을 좌우하는 주요 자원을 의미한다.

사전 측정(Training)에서 RI와 PPR에 따른 설정표를 만들고, 실행 중(Runtime)에는 주기적으로 지배 자원을 판단해 그에 맞는 구성을 조회한다. 자세한 정의는 [원 논문 4절](https://ghose.cs.illinois.edu/papers/25eurosys_crave.pdf)에 있다.

**본 연구 적용:** 주파수 변화가 해당 자원과 타 자원의 성능에 미치는 영향을 측정하고, 측정 결과를 기반으로 설정표를 구성하였다. 입력 처리와 답변 생성, 단독 실행과 동시 실행을 나누어 측정하였다.

**구현 범위:** CRAVE의 전력 대비 성능 최적화 전체를 재현한 것은 아니다. 실험 보드에서는 메모리 주파수를 변경할 수 없어 CPU 메모리 읽기량을 간접 지표로 사용하였으며, 속도 제약을 만족하는 설정 중 열 면적 최소화를 선택 기준으로 설정하였다.

본 연구의 RI 분석은 **주파수 변화에 대한 성능 변화의 로그 기울기**를 사용한다. 측정점이 적고 성능이 한 방향으로만 변하면 상관계수가 ±1에 몰려 변화 크기를 구분하기 어렵기 때문이다. 음수는 해당 주파수를 높일수록 상대 성능이 떨어지는 관계를 의미한다. [분석 코드](code/scripts/sweep/ri_cpu_gpu_proxy.py)는 참고용 Spearman 상관계수도 계산하지만, 본 연구의 기울기와 원 논문의 상관계수를 같은 척도로 비교하지 않는다.

중간 자료에서는 **CPU·GPU·NPU·Memory 4×4 RI를 계획**하였으나, 실제 분석 범위는 **CPU·GPU·NPU 3×3 RI**이다. 메모리 주파수는 직접 변경할 수 없어 CPU 읽기량으로 메모리 병목을 분석하였다. `n/m`은 미측정 항목을 나타낸다.

연결되는 결과: [CPU 스윕](results/csv/cpufreq_sweep_20261113_summary.csv), [CPU·GPU 동시 실행](results/csv/corun_20261113_summary.csv), [사전 계산 테이블](results/csv/precomputed_table_20261118.csv).

### EnerInfer — 응답 속도 제약 기반 에너지·열 관리

**[EnerInfer: Energy-Aware On-Device LLM Inference](https://arxiv.org/abs/2606.23001)**

Bohua Zou 외 · arXiv:2606.23001, 2026년 6월

**연구 배경:** 최대 속도로 답변을 생성하면 전력과 발열 부담이 커진다. 사용자가 요구하는 생성 속도를 만족한다면 주파수를 조금 낮출 여지가 있지만, 어느 설정이 효율적인지는 모델·엔진·장비에 따라 달라진다.

**주요 방법:** 사전 단계에서 모델 구조로 NPU·메모리 주파수별 처리량과 전력을 예측해 표를 만든다. 실행 중에는 에너지 효율과 처리량의 **Pareto Frontier**를 바탕으로 응답 속도 조건을 만족하는 구성을 선택한다. Pareto Frontier는 처리량과 에너지 효율의 상충관계를 나타내는 후보 집합이다. 피드백으로 예측 오차를 보정하고, 미래 온도가 임계치를 넘을 것으로 예상되면 **MPC 기반 열 제어**, 즉 예측한 온도를 바탕으로 미리 설정을 조절하는 모드로 전환한다. Prefill은 빠르게 유지하고 최적화는 decode에 집중한다. 중간발표의 Offline·Runtime 설명과 대응하며 근거는 [논문 3~4절](https://arxiv.org/html/2606.23001v1)에 있다.

**본 연구 적용:** Prefill과 decode를 구분하고, **TTFT와 TPOT을 기준선 대비 허용 범위 내로 유지하는 조건**을 적용하였다. 요청 전체의 주파수 제한과 decode 구간만의 주파수 제한을 별도로 비교하였다.

**구현 범위:** 여러 모델에 일반화하는 처리량·전력 예측기나 미래 온도 예측 제어는 구현하지 않았다. 측정한 설정을 직접 표로 정리하고 `governor-lite`가 적용한다. 온도는 설정 비교 지표로 사용하며, 실행 중 미래 온도를 예측하여 주파수를 결정하는 제어는 포함하지 않는다. 전력 센서 부재라는 문제의식은 공유하되, 본 연구의 열 면적은 EnerInfer의 에너지 효율과 별개 지표다.

연결되는 결과: [단계별 CPU 주파수 실험](results/csv/thermal_ab_phase_20261114_summary.csv), [긴 문맥 동시 실행 E7](results/csv/e7_ck_summary.csv), [자동 적용 재실행](results/csv/gov2_summary.csv).

### KVSwap — 저장장치 기반 KV cache 오프로딩

**[KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference](https://arxiv.org/abs/2511.11907)**

Huawei Zhang, Chunwei Xia, Zheng Wang · MobiSys 2026 · [공식 학회 프로그램](https://www.sigmobile.org/mobisys/2026/program/) · [출판 기록](https://eprints.whiterose.ac.uk/id/eprint/240121/)

**연구 배경:** 문맥이 길어질수록 KV cache가 커져 메모리 용량의 한계에 부딪힌다. CPU와 GPU가 메모리를 공유하는 기기에서는 GPU의 KV를 CPU 메모리로 옮기는 것만으로 전체 RAM 사용량을 줄이기 어렵다. 저장장치로 옮길 수는 있지만 매번 전체 KV를 읽으면 느려진다.

**주요 방법:** 사전에 SVD, 즉 행렬의 주요 성분을 뽑는 분해로 저랭크 K 어댑터를 만들고 그룹 크기와 선택·재사용 예산을 정한다. 실행 중에는 전체 KV를 저장장치에 보관하고 압축된 키 정보로 중요한 그룹을 예측한다. 재사용 버퍼에 없는 그룹만 미리 읽어 연산과 I/O를 겹친다. 중간발표에서 설명한 Offline·Runtime 흐름이다. 방법의 개요는 [논문 초록](https://arxiv.org/abs/2511.11907)과 [공식 발표 소개](https://www.sigmobile.org/mobisys/2026/program/)에서 확인할 수 있다.

**본 연구 적용:** 저랭크 키 예측기, 그룹 단위 선택, direct I/O, 재사용 버퍼를 llama.cpp의 CPU 경로에 단순화해 구현하였다. 패치는 [01~05](code/patches/), 준비·실행 절차는 [재현 절차](code/README.md)에 있다.

**구현 범위:** 원 논문의 GPU 스테이징, 전체 비동기 파이프라인, 장치별 튜닝 및 대규모 품질 평가는 재현 범위에서 제외하였다. 본 연구는 **오프로딩에 따른 CPU 읽기량, 주파수 민감도, 동시 실행 GPU 지연 및 열 면적의 변화**를 분석하였다.

본 구현의 `KV swap`은 저장장치 이동과 선택적 어텐션을 묶은 설정이다. 실험 결과는 **CPU 읽기 부담 감소에 따른 연산 쪽으로의 병목 이동과 동시 실행 간섭 완화**를 지지한다. 저장장치가 RAM보다 빨라져서 얻은 효과라는 뜻은 아니다. 별도 0.5B 추가 실험에서는 디스크 읽기가 매우 적어 선택적 읽기와 재사용 효과가 주된 것으로 해석하였다.

연결되는 결과: [문맥 길이별 GPU 간섭](results/csv/corun_kvswap_ctx_20261119_summary.csv), [KV 선택의 NIAH 검사](results/csv/kvswap_niah_4k_20261107.csv), [주파수 민감도·작은 모델 결과 해설](results/README.md).

## 4. 관련 연구

### HeteroInfer — GPU·NPU 이기종 추론

**[Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference](https://arxiv.org/abs/2501.14794)**

Le Chen 외 · SOSP 2025 · 시스템 이름: HeteroInfer

이 연구는 모바일 GPU·NPU의 연산 특성과 메모리 대역폭을 분석하고, 이를 활용해 하나의 LLM 추론을 두 장치에 나누어 실행한다. 장치 사이의 동기화와 작업 분할도 설계한다. 방법과 평가 개요는 [논문](https://arxiv.org/abs/2501.14794)에 있다.

본 실험은 해당 분석 관점을 참고하여 다중 연산 장치의 동시 실행 간섭을 측정하였다. HeteroInfer는 단일 추론의 이기종 병렬화를 대상으로 하며, 본 연구는 별도 추론 요청 간 자원 경합을 대상으로 한다. 두 연구는 동시 실행의 구성과 분석 목적에서 차이가 있다.

### Neuralink — 저장 배치 및 I/O 효율

**[Neuralink: Fast on-Device LLM Inference with Neuron Co-Activation Linking](https://doi.org/10.1145/3676642.3736114)**

Tuowei Wang 외 · ASPLOS 2025 · [공개 원고](https://arxiv.org/abs/2410.19274)

이 연구는 모델 가중치를 플래시에 두고 필요한 뉴런만 읽는 추론에서, 작은 읽기 요청이 너무 많아지는 문제를 다룬다. 함께 활성화되는 뉴런을 가깝게 배치해 연속 읽기를 늘리고, 초당 I/O 요청 수(IOPS)의 제약을 줄인다. 방법의 개요는 [공개 원고](https://arxiv.org/abs/2410.19274)에 있다.

본 연구의 저장장치 접근 대상은 KV cache이다. Neuralink는 **총 읽기량과 함께 요청 크기·접근 패턴·저장장치 특성을 평가해야 한다**는 분석 관점을 제공한다. 본 실험에서도 디스크 사용률만으로 병목을 판단하지 않고 CPU의 I/O 대기와 읽기량을 함께 기록하였다.

### I/O-Cool — I/O 기반 온도 제어

**[I/O-Cool: Using I/Os for Temperature Regulation When Training on the Edge](https://doi.org/10.1145/3837053.3837345)**

Yanis Mohammedi 외 · HotStorage 2026 · [저자 발표 자료](https://www.hotstorage.org/2026/slides/IO-Cool.pdf)

이 연구는 메모리가 제한된 기기의 학습에서 연산 구간과 데이터 로딩 구간이 번갈아 나타나는 점을 이용한다. I/O 구간의 CPU 활동 감소를 냉각에 활용하며, 읽기 청크 크기와 CPU 주파수를 공동 선택하여 온도 제한을 만족시킨다. 설명은 [발표 자료 10~14쪽](https://www.hotstorage.org/2026/slides/IO-Cool.pdf)에 있다.

본 연구에서는 KV 오프로딩 후 열이 줄어든 이유를 해석할 때 참고하였다. 다만 본 측정에서 CPU I/O 대기는 최대 약 2%로 작았고, CPU 읽기량 감소와 decode 시간 변화가 함께 나타났다. 따라서 **I/O 대기 구간의 냉각 효과와 열 면적 감소 사이의 인과관계는 별도로 검증되지 않았다.** 대기 효과를 따로 분리한 실험도 수행하지 않았다. 관련 측정은 [KV·CPU 주파수 비교](results/csv/e5_summary.csv)에 있다.

## 5. 연구 결과 및 기여

본 연구는 **KV 접근 방식과 자원 간 간섭, 주파수 설정 및 발열의 연관성**을 분석하였다. CRAVE의 자원 상호작용 분석, EnerInfer의 단계별 운용 전략, KVSwap의 오프로딩 기법을 기반으로 다음 관계를 실험적으로 검증하였다.

| 확인한 관계 | 관찰한 결과 | 해석 범위 |
|---|---|---|
| KV 읽기 방식 ↔ CPU 주파수의 효과 | 7K CPU 실험에서 읽기량이 약 39~44 → 20~26 GB/s로 줄고, 높은 주파수에서 decode 이득이 커졌다. | 메모리 읽기가 줄어 병목 성격이 달라졌다는 해석을 지지한다. |
| CPU의 KV 읽기 ↔ 별도 GPU 작업의 간섭 | CPU 1.5 GHz·GPU 1 GHz, 약 11K 문맥에서 GPU 슬로다운이 +87% → +17%로 줄었다. | 문맥 길이와 주파수에 따라 변화 폭이 다르다. |
| KV 오프로딩 + 단계별 CPU 주파수 ↔ 발열 | E7에서 요청 전체 열 면적 19% 감소, CPU·GPU TPOT과 CPU TTFT 대리값이 모두 기준선 1.10배 이내였다. | 특정 7K 동시 실행 구성의 n=1 결과이다. |
| 측정한 설정표 ↔ 실행 중 자동 적용 | `governor-lite`가 decode 1.5 GHz 고정 설정과 유사한 결과를 자동 적용으로 재현하였다. | KV 오프로딩만 사용한 설정 대비 TTFT 유지, 첫 토큰 이후 열 약 36% 감소. |

근거: [KV·CPU 주파수 비교](results/csv/e5_summary.csv), [긴 문맥 동시 실행 E7](results/csv/e7_ck_summary.csv), [문맥 길이별 간섭](results/csv/corun_kvswap_ctx_20261119_summary.csv), [자동 적용 수정 후 결과](results/csv/gov2_summary.csv).

### NPU 분석 범위

NPU는 단독 주파수 스윕, CPU·GPU와의 동시 실행, 세 자원 동시 실행까지 **분석 대상**에 포함하였다. 그러나 자동 제어 프로그램이 CPU·GPU·NPU를 모두 함께 최적화하는 것은 아니다.

- NPU·CPU 동시 decode에서 NPU가 최대 +16.6% 느려졌으며, 호스트 CPU 처리 시간 증가가 주요 원인이라는 해석을 지지한다.
- NPU·GPU 조합은 측정한 조건에서 간섭이 작았다.
- NPU는 Qwen2.5-0.5B int8을 사용해 CPU·GPU의 Qwen3-1.7B와 모델이 다르다. 각자의 단독 실행 대비 변화로 해석한다.
- NPU 발열 차이는 센서 해상도 아래여서 열 면적에 근거한 주파수 선택 규칙을 만들지 못하였다.

근거: [NPU 단독·동시 실행](results/csv/npu_decode_summary.csv). 해당 모델·주파수 조합에서 확인한 자원 간 관계로 해석한다.

### 구현 및 검증의 한계

메모리 주파수 제어, 전력·에너지 직접 측정, 미래 온도 예측 제어, KVSwap 전체 파이프라인과 대규모 품질 평가, 다양한 모델·요청에서의 자동 제어 일반화는 포함하지 않았다. 품질 입력값 약 0.945도 이전에 측정한 어텐션 보존 비중이며, 전체 설정에서 답변 정답률을 재측정한 값이 아니다.

본 실험 결과는 **KV 접근 방식과 단계별 주파수 설정의 통합 설계 필요성**을 뒷받침한다.

## 6. 연구 계획 대비 수행 범위

| 중간 자료의 연구 방향 | 수행 내용 및 관련 분석 |
|---|---|
| 자원 상호작용을 NPU까지 확장 | [주파수 민감도](results/README.md#frequency-sensitivity)·[자원 간 상호작용](results/README.md#resource-interaction): 세 자원의 주파수 민감도와 단독·동시 실행 관계 |
| KV 오프로딩 후 병목 변화 분석 | [KV 오프로딩](results/README.md#kv-offloading): CPU 읽기량·주파수 민감도·별도 GPU 작업의 지연 |
| 지배 자원 기반 온라인 제어 | [단계 인식 주파수 관리](results/README.md#governor-lite): 속도 제약 안에서 고른 설정과 단계별 자동 적용 |
| 에너지 효율 평가 | 전력 직접 측정 대신 열 면적으로 발열 비교. 전력 측정 환경의 에너지 기준 재검증을 후속 과제로 제안 |

중간 자료의 Firefly·RK3588S와 후속 실험의 Orange Pi·CIX P1은 다른 실험 환경이다. 플랫폼과 모델이 다르므로 초기 주파수 결과와 후속 측정값은 구분하여 해석한다. [실험 환경 전환 및 수행 범위](README.md#research-progress)
