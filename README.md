# Optimizing Cross-Resource Coordination for Thermal-Aware On-device LLM Inference

**파자마파티즈** (김휘영, 김지호, 서예현, 임예랑)
- 연세대학교 컴퓨터종합설계 2026-2
- 지도: 정진규 교수님, 하승원 조교님

## 연구 요약

- 배경: 온디바이스 LLM 추론 시 CPU, GPU, NPU가 공유 메모리를 두고 경쟁
- 대상: Orange Pi 6 Plus (CIX P1)
- 측정: 자원별 주파수와 KV cache 오프로딩(KV를 NVMe로 내림)이 속도와 발열에 미치는 영향
- 결과: 속도 제약을 지키면서 발열이 가장 낮은 운용 조건 도출

## 핵심 결과

- 포스터 Figure 번호와 대응
- 수치와 조건은 오른쪽 상세 문서 참조

| | 결과 | 상세 |
|---|---|---|
| **Fig. 1** 주파수 민감도 | - prefill: CPU, GPU, NPU 모두 주파수에 거의 비례<br>- decode: CPU는 약 1.5 GHz 이후 포화(메모리 읽기 약 39~44 GB/s), GPU는 거의 비례, NPU는 둔감<br>- NPU: 주파수 3배에 prefill 2.49배, decode 1.56배 | [cpu-gpu-memory-sweeps](results/docs/cpu-gpu-memory-sweeps-20261113.md), [results-20261121](results/docs/results-20261121.md) 4-1절 |
| **Fig. 2** 자원 간 상호작용 | - 단독 실행: 자원 간 독립<br>- 동시 실행 decode: CPU 주파수 상승 시 GPU 최대 **+74.6%** 느려짐<br>- 동시 실행 prefill: 모든 조합에서 간섭 0.5% 이내<br>- NPU: CPU와 동시 실행 시 최대 +16.6% 느려짐(원인은 NPU가 아닌 호스트 쪽 메모리 경합), GPU와는 간섭 거의 없음 | [results-20261121](results/docs/results-20261121.md) 4~5절 |
| **Fig. 3** KV 오프로딩 | - CPU DRAM 읽기 약 절반으로 감소(39~44 → 20~26 GB/s)<br>- decode 속도 동일하거나 최대 11% 향상(2.2 GHz에서 73.6 → 65.6 ms/token)<br>- 동시 실행 GPU 슬로다운: 컨텍스트가 길수록 감소 폭 증가, 11K에서 **+87% → +17%**(3K +43 → +24%, 7K +64 → +19%) | [results-20261121](results/docs/results-20261121.md) 1, 6절, [corun-kvswap-ctx](results/docs/corun-kvswap-ctx-20261119.md) |
| **Fig. 4** 발열 | - CPU 전용: decode만 1.5 GHz로 낮추면 TTFT 유지, 열 면적 12.6% 감소<br>- CPU+GPU 동시 실행: **KV 오프로딩 + decode 1.5 GHz**로 속도 제약 유지, 열 면적 **19%** 감소(CPU 작업 TPOT 0.97배, GPU 작업 0.79배, TTFT 1.05배) | [thermal-ab-phase](results/docs/thermal-ab-phase-20261114.md), [results-20261121](results/docs/results-20261121.md) 3절 |
| **동적 관리** | - 사전 계산 테이블을 실행 중에 적용하는 `governor-lite` 구현<br>- CPU 단계(prefill, decode)와 GPU 작업을 감지해 A720 주파수 전환<br>- 같은 작업량의 고정 설정(decode 1.5 GHz)과 동일한 결과<br>- KV 오프로딩만 사용 시 대비 첫 토큰 이후 열 36% 감소 | [governor-lite](results/docs/governor-lite.md), [precomputed-table](results/docs/precomputed-table.md) |

**부가 결과**
- 대상: 작은 모델(Qwen2.5-0.5B)
- 컨텍스트 256토큰: KV 오프로딩 손해(TPOT 1.13배)
- 컨텍스트 4K: TPOT 0.88배, 16K: 0.43배
- 손익분기점: 256과 4K 사이
- 상세: [results-20261121](results/docs/results-20261121.md) 6b절

## 저장소 구성

| 폴더 | 내용 |
|---|---|
| [results/](results/) | - 상세 실험 결과<br>- 문서: `results/docs/`, 요약 CSV와 그림: `results/csv/`<br>- 색인: [results/README.md](results/README.md) |
| [code/](code/) | - 실험·분석 스크립트<br>- llama.cpp용 KV 오프로딩 패치 01~05<br>- 사용법: [code/README.md](code/README.md) |

## 실험 환경

| 항목 | 내용 |
|---|---|
| 보드 | Orange Pi 6 Plus (CIX P1: A720 ×8 + A520 ×4, Mali GPU, NPU), RAM 31.7 GB, 커널 6.6.89-cix |
| 추론 엔진 | - CPU, GPU: llama.cpp (GPU는 Vulkan, `-ngl 99`)<br>- NPU: CIX NOE (`.cix`) |
| 모델 | - CPU, GPU: Qwen3-1.7B Q4_K_M<br>- NPU: Qwen2.5-0.5B int8<br>- NPU는 모델이 달라 자기 단독 실행 대비 비율로만 비교 |
| KV 오프로딩 | - KVSwap을 단순화해 llama.cpp에 구현<br>- 저랭크 키 예측기로 고른 KV 그룹만 NVMe에서 direct I/O로 읽음<br>- 패치: [code/patches/](code/patches/) |
| 측정 | TTFT(prefill 시간), TPOT, DRAM 읽기 대역폭(A720 PMU `l2d_cache_refill`), 온도, **열 면적**(온도 상승 × 시간) |

**"최적"의 기준**
- 전력 센서 부재로 에너지 직접 측정 불가
- 대리 지표: 열 면적 (가장 뜨거운 열 존의 온도 상승을 식는 꼬리까지 시간 적분)
- 조건: TPOT, TTFT가 기준선의 1.10배 이내, 품질 0.93 이상
- 최적: 조건을 만족하는 설정 중 열 면적 최소

## 재현

- 필요: 보드(Orange Pi 6 Plus), llama.cpp 빌드
  - 커밋 `931351ea5` (빌드 b11278)
  - 옵션 `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9.2-a+i8mm+bf16+dotprod+fp16`
- 패치 적용 순서와 실행 명령: [code/README.md](code/README.md)
- 저장소 미포함: 모델 파일, 보드 접속 정보

## 한계

- 반복 횟수: 대부분 1~2회 (`results-20261121`의 값은 n=1)
- 노이즈 폭: 열 면적 판정 최소 5%, 요청 전체 열 면적 차이가 작은 칸은 노이즈 범위 내
- 열 면적은 에너지가 아닌 대리 지표
- 열 존 온도 해상도 1 °C로 NPU 발열 측정 불가
- 측정 범위 제한: 모델 1개(Qwen3-1.7B), 컨텍스트 일부, GPU·A720 주파수 일부 점
- NPU는 다른 모델(0.5B)
- KV 오프로딩 제약
  - CPU 프로세스에만 구현
  - GPU와 한 프로세스에서 사용 시 세그폴트
  - 3K 이하 짧은 컨텍스트에서는 이득 없음 또는 손해
- 품질(0.945): NIAH 점검 값을 판정에 입력, 전 설정 재측정 아님
- 메모리 주파수 직접 변경 불가: 메모리 축은 PMU 대역폭으로 간접 확인

## 선행 연구

- CRAVE (EuroSys '25)
- EnerInfer
- KVSwap (MobiSys '26)
- 정리 문서: [related-work.md](related-work.md) (참고 연구 3편, NPU 확장 현황, 본 연구와의 비교표 포함)
