# 코드

실험과 분석에 쓴 스크립트와 llama.cpp용 KV 오프로딩 패치입니다. **스크립트는 이 폴더(`code/`)를 저장소 루트로 보고** `scripts/...` 상대 경로로 서로를 찾으므로, 아래 명령은 `code/`에서 실행합니다. 대부분 보드(Orange Pi 6 Plus)에서 도는 스크립트이고, 보드가 없어도 되는 것은 `--dry-run`, `--analyze`, `ri_cpu_gpu_proxy.py`, `gov_replay.py`입니다.

```
code/
  patches/   llama.cpp용 KV 오프로딩 패치 01~05
  scripts/
    kvswap/  KV 오프로딩 실행, 일괄 실행, 요약, 저랭크 어댑터 생성, NIAH 품질 비교
    sweep/   CPU, GPU, NPU 주파수 스윕, 동시 실행, 메모리 대역폭 보정, RI 분석, 사용률 샘플러
    joint/   KV 오프로딩 × 주파수 실험: 열 면적 A/B, KV swap과 GPU 동시 실행, governor-lite
    npu/     NPU 지연 측정, NPU LLM 준비(ONNX 내보내기, cixbuild 설정, 실행 래퍼)
    exp_queue.sh   보고서용 실험(E1~E9)을 한 번에 돌리는 큐
  data/      프롬프트
```

스크립트별 한 줄 설명은 [scripts/README.md](scripts/README.md)(원 저장소의 색인)에 있습니다.

## 준비

| 항목 | 내용 |
|---|---|
| 보드 | Orange Pi 6 Plus. 비밀번호 없이 쓰는 sudo 규칙이 필요합니다(`cpufreq`, `devfreq`, `drop_caches`에 `tee`만 허용) |
| llama.cpp | 커밋 `931351ea5`(빌드 b11278)를 `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9.2-a+i8mm+bf16+dotprod+fp16`로 빌드. GPU 실험은 Vulkan 빌드가 별도로 필요 |
| 모델 | Qwen3-1.7B Q4_K_M (Hugging Face `unsloth/Qwen3-1.7B-GGUF`). 기본 경로 `$HOME/LLM/Qwen3-1.7B-Q4_K_M.gguf` |
| 접속 | 보드는 SSH로 접속합니다. 접속 정보(IP, 키, 비밀번호)는 이 저장소에 없습니다 |

## KV 오프로딩 패치 적용

순서가 중요합니다. llama.cpp 소스 디렉터리에서:

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

저랭크 어댑터는 `scripts/kvswap/make_kvswap_adapter.py`로 만듭니다(`--model Qwen/Qwen3-1.7B --out adapters/`).

## 주요 실험 명령

각 스크립트의 맨 위 설명에 전체 옵션이 있습니다. 모두 `--dry-run`으로 계획과 예상 시간을 먼저 볼 수 있습니다.

```sh
# 열 면적 A/B: naive 대 KV swap × 주파수 (6개 설정 × 1라운드 약 1.5시간)
python3 scripts/joint/thermal_ab.py --dry-run
python3 scripts/joint/thermal_ab.py --configs naive,kv_1500 --repeat 2
python3 scripts/joint/thermal_ab.py --analyze runs/thermal_<ts>        # 저장된 실행으로 리포트 다시 계산

# KV swap이 동시에 도는 GPU의 슬로다운을 줄이는가 (A720 읽기 대역폭 기록은 perf용 sudo 필요)
SUDO_PASS=... python3 scripts/joint/corun_kvswap.py

# 단계 인식 주파수 관리 (governor-lite)와 고정 설정 비교
python3 scripts/joint/thermal_ab.py --prompt-chars 32000 --ntok 6500 --ctx 14336 \
    --ck-gpu-delay 30 --ck-gpu-bursts 3 --ck-gpu-ntok 700 --ck-gpu-off 40 \
    --configs ck_ram_gdef_cdef,ck_swap_gdef_cdef,ck_swap_gdef_d1500,ck_swap_gov --repeat 1
python3 scripts/joint/gov_replay.py --help                             # 실행 기록을 영상과 그림으로

# 주파수 스윕, 동시 실행, NPU
python3 scripts/sweep/sweep_cpufreq.py --help
python3 scripts/sweep/sweep_corun.py --freqs 1200,1500,1800,2200 --gpu-freqs 350,600,800,1000 --corun-phase decode
python3 scripts/sweep/sweep_npu.py --help

# 보고서용 실험 E1~E9 한 번에
bash scripts/exp_queue.sh
```

설정 이름은 `thermal_ab.py`의 규칙을 따릅니다.

| 이름 | 뜻 |
|---|---|
| `naive`, `kv_default` | RAM KV, KV swap(주파수는 기본 거버너) |
| `phase_<MHz>`, `kvphase_<MHz>` | prefill은 기본, 첫 토큰부터 A720을 고정(RAM KV, KV swap) |
| `co_g<GPU|def>_c<A720|def>` | CPU 작업과 GPU 작업 동시 실행(짧은 컨텍스트) |
| `ck_<ram|swap>_g<GPU|def>_c<A720|def>` | 긴 컨텍스트 CPU 작업(KV 위치 ram 또는 swap)과 GPU 작업 동시 실행, A720을 요청 전체에 고정 |
| `ck_<ram|swap>_g<GPU|def>_d<A720>` | 위와 같고 A720을 첫 토큰부터 고정(prefill은 기본) |
| `ck_<ram|swap>_gov` | governor-lite가 주파수를 정함 |

## 사용 전에 알아둘 점

- 스크립트는 보드의 `thermal_zone`, `cpufreq`, `devfreq`, `perf`를 직접 읽고 씁니다. 다른 보드에서는 경로와 주파수 표를 고쳐야 합니다.
- 실험 중에는 보드에서 다른 작업을 하지 마세요. 온도와 대역폭 측정이 오염됩니다.
- NPU 스크립트는 CIX NOE SDK와 NPU용 `.cix` 모델이 있어야 합니다. 모델 파일은 이 저장소에 없습니다.
- 이 폴더의 스크립트 주석과 일부 도움말은 영어와 한국어가 섞여 있습니다.
