# 코드

- 내용: 실험·분석 스크립트, llama.cpp용 KV 오프로딩 패치
- 실행 위치: **`code/`를 저장소 루트로 간주**, 스크립트 간 `scripts/...` 상대 경로 사용 → 아래 명령은 `code/`에서 실행
- 실행 환경: 대부분 보드(Orange Pi 6 Plus) 전용
- 보드 없이 가능: `--dry-run`, `--analyze`, `ri_cpu_gpu_proxy.py`, `gov_replay.py`

```
code/
  patches/   llama.cpp용 KV 오프로딩 패치 01~05
  scripts/
    kvswap/  KV 오프로딩 실행, 일괄 실행, 요약, 저랭크 어댑터 생성, NIAH 품질 비교
    sweep/   CPU, GPU, NPU 주파수 스윕, 동시 실행, 메모리 대역폭 보정, RI 분석, 사용률 샘플러
    joint/   KV 오프로딩 × 주파수 실험: 열 면적 A/B, KV swap과 GPU 동시 실행, governor-lite
    npu/     NPU 지연 측정, NPU LLM 준비(ONNX 내보내기, cixbuild 설정, 실행 래퍼)
    exp_queue.sh   보고서용 실험(E1~E9) 일괄 실행 큐
  data/      프롬프트
```

- 스크립트별 한 줄 설명: [scripts/README.md](scripts/README.md) (원 저장소 색인)

## 준비

| 항목 | 내용 |
|---|---|
| 보드 | - Orange Pi 6 Plus<br>- 비밀번호 없는 sudo 규칙 필요 (`cpufreq`, `devfreq`, `drop_caches`에 `tee`만 허용) |
| llama.cpp | - 커밋 `931351ea5` (빌드 b11278)<br>- 빌드 옵션 `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9.2-a+i8mm+bf16+dotprod+fp16`<br>- GPU 실험: Vulkan 빌드 별도 필요 |
| 모델 | - Qwen3-1.7B Q4_K_M (Hugging Face `unsloth/Qwen3-1.7B-GGUF`)<br>- 기본 경로 `$HOME/LLM/Qwen3-1.7B-Q4_K_M.gguf` |
| 접속 | - 보드 SSH 접속<br>- 접속 정보(IP, 키, 비밀번호) 저장소 미포함 |

## KV 오프로딩 패치 적용

- 적용 순서 준수 필요
- llama.cpp 소스 디렉터리에서 실행

```sh
git apply ../patches/01-kv-mmap-select-prefetch.patch ../patches/02-kv-direct-io.patch \
          ../patches/03-kv-flush-prefill.patch ../patches/04-kv-predictor.patch ../patches/05-kv-reuse-buffer.patch
```

| 패치 | 내용 |
|---|---|
| 01 | mmap KV 파일, 마스크 선택, 프리페치 |
| 02 | direct I/O로 메모리 제한 없이 항상 오프로딩 |
| 03 | prefill 쓰기를 decode 전에 파일로 내림 |
| 04 | 저랭크 키 예측기로 레이어별 top-M 그룹 선택, 선택 즉시 direct I/O 읽기 |
| 05 | direct I/O 재사용 버퍼 |

- 저랭크 어댑터 생성: `scripts/kvswap/make_kvswap_adapter.py` (`--model Qwen/Qwen3-1.7B --out adapters/`)

## 주요 실험 명령

- 전체 옵션: 각 스크립트 상단 설명 참조
- 모든 스크립트: `--dry-run`으로 계획과 예상 시간 사전 확인 가능

```sh
# 열 면적 A/B: naive 대 KV swap × 주파수 (6개 설정 × 1라운드 약 1.5시간)
python3 scripts/joint/thermal_ab.py --dry-run
python3 scripts/joint/thermal_ab.py --configs naive,kv_1500 --repeat 2
python3 scripts/joint/thermal_ab.py --analyze runs/thermal_<ts>        # 저장된 실행으로 리포트 재계산

# KV swap의 동시 실행 GPU 슬로다운 감소 (A720 읽기 대역폭 기록은 perf용 sudo 필요)
SUDO_PASS=... python3 scripts/joint/corun_kvswap.py

# 단계 인식 주파수 관리(governor-lite)와 고정 설정 비교
python3 scripts/joint/thermal_ab.py --prompt-chars 32000 --ntok 6500 --ctx 14336 \
    --ck-gpu-delay 30 --ck-gpu-bursts 3 --ck-gpu-ntok 700 --ck-gpu-off 40 \
    --configs ck_ram_gdef_cdef,ck_swap_gdef_cdef,ck_swap_gdef_d1500,ck_swap_gov --repeat 1
python3 scripts/joint/gov_replay.py --help                             # 실행 기록의 영상·그림 변환

# 주파수 스윕, 동시 실행, NPU
python3 scripts/sweep/sweep_cpufreq.py --help
python3 scripts/sweep/sweep_corun.py --freqs 1200,1500,1800,2200 --gpu-freqs 350,600,800,1000 --corun-phase decode
python3 scripts/sweep/sweep_npu.py --help

# 보고서용 실험 E1~E9 일괄 실행
bash scripts/exp_queue.sh
```

**설정 이름 규칙** (`thermal_ab.py`)

| 이름 | 뜻 |
|---|---|
| `naive`, `kv_default` | RAM KV, KV swap (주파수는 기본 거버너) |
| `phase_<MHz>`, `kvphase_<MHz>` | prefill 기본, 첫 토큰부터 A720 고정 (RAM KV, KV swap) |
| `co_g<GPU\|def>_c<A720\|def>` | CPU 작업과 GPU 작업 동시 실행 (짧은 컨텍스트) |
| `ck_<ram\|swap>_g<GPU\|def>_c<A720\|def>` | 긴 컨텍스트 CPU 작업(KV 위치 ram 또는 swap)과 GPU 작업 동시 실행, A720을 요청 전체에 고정 |
| `ck_<ram\|swap>_g<GPU\|def>_d<A720>` | 위와 동일, A720을 첫 토큰부터 고정 (prefill 기본) |
| `ck_<ram\|swap>_gov` | governor-lite가 주파수 결정 |

## 사용 전 유의 사항

- 스크립트는 보드의 `thermal_zone`, `cpufreq`, `devfreq`, `perf`를 직접 읽고 씀 → 다른 보드는 경로와 주파수 표 수정 필요
- 실험 중 보드에서 다른 작업 금지 (온도, 대역폭 측정 오염)
- NPU 스크립트: CIX NOE SDK와 NPU용 `.cix` 모델 필요, 모델 파일 저장소 미포함
- 스크립트 주석과 일부 도움말: 영어와 한국어 혼용
