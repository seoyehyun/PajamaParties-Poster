# 실험 코드 및 재현 절차

본 폴더는 **자원 상호작용 분석, KV cache 오프로딩 및 단계 인식 주파수 관리**에 사용한 스크립트와 llama.cpp 패치를 제공한다. 재현 대상은 Orange Pi 6 Plus 실험이며, 중간 자료의 Firefly·RK3588S/RKLLM 환경과 구분한다. 연구 개요와 측정 근거는 [프로젝트 소개](../README.md) 및 [실험 결과](../results/README.md)에 기술하였다.

실제 측정은 Orange Pi 6 Plus의 Linux에서 실행한다. 보드가 없어도 일부 스크립트의 실행 계획을 출력하거나, 필요한 원본 로그를 확보한 경우 분석·재생을 할 수 있다. 이 문서의 셸 명령은 Linux용이다.

## 1. 실험 구성 및 스크립트

| 실험 목적 | 관련 코드 | 제어 변수 및 측정 항목 |
|---|---|---|
| 실행 계획 확인 | `thermal_ab.py --dry-run` 등 | 설정 목록과 예상 시간 |
| [주파수 민감도](../results/README.md#frequency-sensitivity) | `scripts/sweep/sweep_cpufreq.py`, `sweep_gpufreq.py`, `sweep_npu.py` | 장치별 주파수와 prefill·decode 속도 |
| [자원 간 상호작용](../results/README.md#resource-interaction) | `sweep_corun.py`, `sweep_npu.py`, `ri_cpu_gpu_proxy.py` | 단독·동시 실행 간섭과 RI 분석 |
| [KV 오프로딩의 주파수 민감도](../results/README.md#kv-offloading) | `scripts/joint/thermal_ab.py` | CPU decode 주파수별 RAM KV·오프로딩 비교. E5 옵션은 `exp_queue.sh`에 있다. |
| [KV 오프로딩의 동시 실행 간섭](../results/README.md#kv-offloading) | `scripts/joint/corun_kvswap.py` | KV 위치·문맥 길이에 따른 GPU 슬로다운 |
| [성능·발열 비교](../results/README.md#thermal-latency) | `scripts/joint/thermal_ab.py` | 전체 고정·decode만 조절·오프로딩의 열과 응답 시간 |
| [Governor-lite](../results/README.md#governor-lite) | `thermal_ab.py` + `governor_lite.py` | 상태 감지 → 테이블 조회 → A720 주파수 적용 → 고정 설정과 비교 |

저장한 실행을 영상·그림으로 재생하는 코드는 `scripts/joint/gov_replay.py`다. 실행별 원본 로그와 NumPy·matplotlib, 영상 생성에는 ffmpeg가 필요하다.

`--dry-run`은 계획을 출력하고 종료하므로 모델 실행이나 주파수 변경을 하지 않는다. 이 옵션은 아래에서 예시로 든 측정용 Python 스크립트가 지원한다. 셸 스크립트나 어댑터 생성기까지 모두 지원하는 옵션은 아니다.

## 2. 실행 환경 구성

### 작업 디렉터리

보드에 저장소를 배치하고 `code/`를 작업 디렉터리로 설정한다.

~~~sh
cd /path/to/PajamaParties-Poster/code
POSTER_CODE="$PWD"
~~~

이하의 `scripts/...`는 이 폴더 기준 상대 경로다. 새 측정의 기본 출력 위치도 `code/runs/`다.

~~~text
code/
├── patches/                 llama.cpp용 패치 01~05
└── scripts/
    ├── kvswap/              KV 실행, 선택용 어댑터 생성, NIAH 답변 검사
    ├── sweep/               주파수 스윕, 동시 실행, 메모리·사용률 측정
    ├── joint/               발열 비교, KV와 GPU 동시 실행, 자동 주파수 관리
    ├── npu/llm/             NPU 모델 내보내기·설정·실행
    └── exp_queue.sh         보고서용 실험을 순서대로 실행하는 큐
~~~

### 실험 환경 및 의존성

| 항목 | 요구 조건 |
|---|---|
| 보드 | Orange Pi 6 Plus, CIX P1. CPU 코어 번호와 센서·주파수 경로는 해당 보드를 기준으로 설정되어 있다. |
| CPU 엔진 | llama.cpp 커밋 `931351ea5`(기록상 빌드 b11278)에 패치 01~05 적용 |
| GPU 엔진 | Vulkan을 활성화한 llama.cpp 빌드. CPU·GPU 동시 실행에 필요하다. |
| 주 모델 | Qwen3-1.7B Q4_K_M GGUF. 기록에서 사용한 배포본은 `unsloth/Qwen3-1.7B-GGUF`다. |
| 입력 문서 | C4 텍스트 기반 `haystack_c4.txt`. 입력 길이와 내용은 측정 결과에 영향을 미친다. |
| KV 선택용 어댑터 | 모델별 저랭크 키 예측기 파일. 아래 생성 절차를 따른다. |
| 저장장치 | KV 파일과 로그를 저장할 NVMe SSD. 예시는 `/mnt/ssd/`를 사용한다. |
| 권한·도구 | Python 3, Bash, `taskset`, 비대화형 sudo 설정. PMU 측정에는 사용 가능한 `perf`와 카운터 접근 권한도 필요하다. |

측정 스크립트는 `sudo -n true`와 `sudo -n tee`를 사용한다. 실제 실행 전에 CPU 주파수(`cpufreq`), GPU·NPU 주파수(`devfreq`), 페이지 캐시 비우기(`drop_caches`)에 필요한 규칙이 설정되어 있어야 한다. PMU 권한이 없으면 일부 대역폭 열이 비어 있을 수 있다.

모델·입력 문서·어댑터·보드 접속 정보는 저장소에 포함되지 않는다. 실제 측정은 보드에 접속한 환경에서 수행한다.

### 파일 경로 및 실행 옵션

다음은 `thermal_ab.py`와 `corun_kvswap.py`의 주요 기본값이다.

| 대상 | 기본 경로 | 변경 옵션 |
|---|---|---|
| 모델 | `$HOME/PajamaParties/models/Qwen3-1.7B-Q4_K_M.gguf` | `--model` |
| CPU 실행 파일 | `$HOME/llama-base/build/bin/llama-completion` | `--llama-bin` |
| 입력 문서 | `/mnt/ssd/prompts/haystack_c4.txt` | `--prompt-src` |
| 예측기 어댑터 | `/mnt/ssd/adapters/kvswap-adapter-r128.gguf` | `--adapter` |
| KV 저장 파일 | `/mnt/ssd/kv.bin` | `--kv-file` |
| 측정할 디스크 이름 | `nvme0n1` | `--dev` |
| GPU 실행 파일 | `thermal_ab.py`: `$HOME/llama-base/build-vulkan/bin/llama-completion` | `--gpu-bin` |
| GPU 벤치마크 | `corun_kvswap.py`: `$HOME/llama-base/build-vulkan/bin/llama-bench` | `--gpu-bench` |

모델 경로 변경 시 `--model /실제/경로/model.gguf`를 지정한다. 기타 스크립트의 기본값과 지원 옵션은 각 파일의 `--help`에 제시되어 있다.

## 3. KV 오프로딩 엔진 구성

### 패치 적용 및 빌드

`POSTER_CODE`는 앞에서 설정한 이 저장소의 `code/` 절대 경로다. 다음 명령은 별도로 준비한 llama.cpp 소스 폴더에서 실행한다. 소스는 위의 기준 커밋이어야 하며, 이미 적용한 패치를 다시 적용하지 않는다.

~~~sh
cd "$HOME/llama-base"
git rev-parse --short HEAD

git apply --check "$POSTER_CODE"/patches/0[1-5]-*.patch
git apply "$POSTER_CODE"/patches/0[1-5]-*.patch

cmake -S . -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv9.2-a+i8mm+bf16+dotprod+fp16
cmake --build build --parallel
~~~

파일명의 번호에 따라 01 → 05 순서로 적용된다. 적용 전 검사(`--check`)가 실패하면 기준 커밋과 기존 패치 적용 여부를 확인한다.

| 패치 | 추가하는 기능 |
|---|---|
| 01 | KV를 파일로 저장하고 일부 그룹을 선택해 읽는 기본 경로 |
| 02 | 운영체제 페이지 캐시를 우회하는 direct I/O 경로. 명시적 RAM 제한 없이도 저장장치 접근을 측정한다. |
| 03 | 입력 처리에서 만든 KV를 답변 생성 전에 파일로 내보내는 처리 |
| 04 | 압축된 키 정보로 중요한 KV 그룹을 예측하고, 선택한 그룹을 읽는 처리 |
| 05 | 최근 읽은 KV를 재사용하는 버퍼 |

GPU 실험까지 수행하려면 같은 소스에서 Vulkan 빌드도 준비한다. Vulkan 개발 라이브러리가 설치되어 있어야 한다.

~~~sh
cmake -S . -B build-vulkan \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv9.2-a+i8mm+bf16+dotprod+fp16
cmake --build build-vulkan --parallel
cd "$POSTER_CODE"
~~~

KV 오프로딩은 CPU 실행에만 적용한다. 같은 프로세스에서 GPU KV 경로와 결합하면 충돌하는 제약이 있다. 아래의 동시 실행 실험은 CPU와 GPU를 **별도 프로세스**로 실행한다.

### KV 선택용 어댑터 생성

어댑터는 원본 모델의 키 정보를 압축하여 생성한 모델별 예측기 파일이며, 읽을 KV 그룹의 중요도 추정에 사용한다. 생성에는 GGUF 추론 파일과 별개로 Hugging Face 형식의 모델 가중치가 필요하다.

PyTorch, Transformers, GGUF, Datasets, NumPy를 설치한 Python 환경에서 실행한다. 노트북 등 별도 머신에서 생성한 뒤 보드로 복사할 수 있다.

~~~sh
python3 scripts/kvswap/make_kvswap_adapter.py \
    --model Qwen/Qwen3-1.7B \
    --samples 16 --seqlen 2048 --ranks 128 \
    --out adapters/
~~~

결과는 `adapters/kvswap-adapter-r128.gguf`다. 보드의 기본 어댑터 경로에 배치하거나, 실험 명령에 `--adapter`로 위치를 지정한다. 위 명령은 C4 validation 텍스트를 사용하므로 모델과 데이터에 접근할 수 있어야 한다.

## 4. 목적별 실행 예제

다음 명령은 기본 파일 경로의 구성이 완료된 환경을 가정한다. `--dry-run`으로 실행 계획을 검토한 후 보드에서 실제 측정을 수행한다. 실행 계획 출력은 파일 존재 여부와 하드웨어 구성의 검증을 포함하지 않는다.

### A. CPU 주파수 조절과 KV 오프로딩의 발열 비교

입력 약 7K토큰, 출력 2,000토큰에서 다음 다섯 설정을 비교한다. 아래 명령은 과거 측정값의 재사용 없이 모든 설정을 새로 측정한다. `--prompt-chars`는 문자 수이므로 실제 토큰 수는 입력 내용에 따라 달라진다.

| 설정 | KV 읽기 방식 | CPU 주파수 |
|---|---|---|
| `naive` | RAM에서 전체 KV 읽기 | 기본 정책 |
| `freq_1500` | RAM에서 전체 KV 읽기 | 입력 처리부터 답변 생성까지 1.5 GHz |
| `phase_1500` | RAM에서 전체 KV 읽기 | 입력 처리는 기본, 답변 생성만 1.5 GHz |
| `kv_default` | KV 오프로딩·선택적 읽기 | 기본 정책 |
| `kvphase_1500` | KV 오프로딩·선택적 읽기 | 입력 처리는 기본, 답변 생성만 1.5 GHz |

~~~sh
python3 scripts/joint/thermal_ab.py \
    --prompt-chars 32000 --ntok 2000 --ctx 10240 \
    --configs naive,freq_1500,phase_1500,kv_default,kvphase_1500 \
    --repeat 2 --dry-run
~~~

실제 측정 시 동일 명령에서 `--dry-run`을 제거한다. `--repeat 2`는 다섯 설정을 두 라운드 수행하도록 지정한다. 설정당 입력 처리·답변 생성·냉각까지 포함하므로 수십 분에서 수 시간이 걸릴 수 있으며, 출력된 시간은 추정치다.

근거 결과: [단계별 주파수 실험](../results/csv/thermal_ab_phase_20261114_summary.csv), [KV·단계별 주파수 재측정](../results/csv/kvphase_ab_summary.csv).

### B. KV 오프로딩에 따른 GPU 동시 실행 간섭 측정

CPU가 긴 문맥의 답변을 생성하는 동안 GPU 작업을 실행한다. CPU의 KV를 RAM에 둘 때와 오프로딩할 때를 비교한다.

~~~sh
python3 scripts/joint/corun_kvswap.py \
    --a720 1500 --gpu-freqs 1000 --trials 2 \
    --prompt-chars 32000 --ctx 12288 --dry-run
~~~

실제 측정 시 `--dry-run`을 제거한다. A720 메모리 읽기량 기록에는 `perf` 접근 권한이 필요하다. 대역폭 기록을 제외한 속도 측정은 `--no-pmu`로 지정한다.

근거 결과: [문맥 길이별 동시 실행](../results/csv/corun_kvswap_ctx_20261119_summary.csv), [다른 주파수의 GPU 간섭](../results/csv/e8_7k_summary.csv).

### C. 자동 주파수 적용과 고정 설정 비교

긴 CPU 답변 생성 중 짧은 GPU 작업을 세 번 실행하고, RAM 기준선·KV 오프로딩·decode 1.5 GHz·자동 적용을 비교한다.

~~~sh
python3 scripts/joint/thermal_ab.py \
    --prompt-chars 32000 --ntok 6500 --ctx 14336 \
    --ck-gpu-delay 30 --ck-gpu-bursts 3 \
    --ck-gpu-ntok 700 --ck-gpu-off 40 \
    --configs ck_ram_gdef_cdef,ck_swap_gdef_cdef,ck_swap_gdef_d1500,ck_swap_gov \
    --repeat 1 --dry-run
~~~

`--dry-run` 제거 시 해당 설정에서 `governor_lite.py`가 함께 실행되므로 데몬의 별도 실행은 필요하지 않다.

본 예제는 governor-lite와 단계별 고정 설정의 비교 실험이다. 현재 설정표에서는 KV 오프로딩의 decode 구간이 GPU 유무와 관계없이 1.5 GHz이다. 자동 적용은 고정 decode 1.5 GHz와 유사한 결과를 보였다. 열 면적 36% 감소는 **KV 오프로딩 단독 적용 설정 대비 첫 토큰 이후**의 값이며, 별도 동시 실행 실험의 RAM 기준선 대비 요청 전체 열 면적 19% 감소와 구분한다.

근거 결과: [자동 적용의 수정 후 재실행](../results/csv/gov2_summary.csv)과 [결과 해설](../results/README.md).

### D. CPU·GPU·NPU 주파수 스윕

전체 옵션과 실행 계획은 다음 명령으로 확인한다.

~~~sh
python3 scripts/sweep/sweep_cpufreq.py --help
python3 scripts/sweep/sweep_gpufreq.py --help
python3 scripts/sweep/sweep_npu.py --help

python3 scripts/sweep/sweep_corun.py \
    --freqs 1200,1500,1800,2200 \
    --gpu-freqs 350,600,800,1000 \
    --corun-phase decode --dry-run
~~~

NPU 실측은 추가로 CIX NOE SDK와 NPU용 `.cix` 모델, [NPU 실행 래퍼](scripts/npu/llm/npu_llm_run.py)가 필요하다. CPU·GPU용 GGUF 파일을 그대로 사용할 수는 없다.

### E. 보고서용 실험 일괄 실행

[exp_queue.sh](scripts/exp_queue.sh)는 E1~E9 실험을 순서대로 실행한다. 앞의 개별 실험과 NPU 환경까지 준비한 뒤 사용한다.

~~~sh
OUT=/mnt/ssd/runs/reproduction bash scripts/exp_queue.sh
~~~

완료 단계는 `.done` 파일로 기록되며 재실행 시 제외된다. 일부 단계만 실행하는 경우 `ONLY="e5 e6_cpu e7_ck"`와 같이 대상 실험을 지정한다. 큐 스크립트는 `--dry-run`을 지원하지 않으므로 실행 전 파일 경로와 설정을 검토한다. 실험 번호별 정의는 [실험 결과](../results/README.md)에 제시되어 있다.

## 5. 출력 데이터 및 재분석

| 출력 | 의미 |
|---|---|
| `code/runs/thermal_<시각>/` | 발열 비교 원본. `meta.json`, `runs.jsonl`, `summary.csv` 및 설정별 디렉터리가 생성된다. |
| `code/runs/corunkv_<시각>/` | KV·GPU 간섭 실험 원본. `windows.csv`와 `summary.csv`로 구간·설정별 결과를 확인한다. |
| 설정별 `trace.csv` | 온도·주파수 시계열 |
| `gov.csv`, `gov_events.jsonl` | 자동 주파수 관리의 상태·결정 기록 |
| `cpu_progress.csv` | CPU 출력 진행 기록. 바이트/초는 토큰/초의 대리값이다. |

다음 명령의 예시 경로를 실제 실행 디렉터리로 대체한다.

~~~sh
python3 scripts/joint/thermal_ab.py --analyze runs/thermal_YYYYMMDD_HHMMSS
python3 scripts/joint/corun_kvswap.py --analyze runs/corunkv_YYYYMMDD_HHMMSS
python3 scripts/joint/gov_replay.py --help
~~~

재분석에는 해당 실행의 원본 파일이 필요하며 `results/csv/`의 요약 자료만으로는 대체할 수 없다. 품질 입력값과 판정 제약은 원본 실행과 동일하게 지정한다.

`ri_cpu_gpu_proxy.py`의 입력 경로는 이전 저장소 구조인 `code/docs/sweep/results/`로 고정되어 있다. 현재 폴더에서 재생성하려면 코드의 `R` 경로를 저장소의 `results/csv/`에 맞춰야 한다. 기존 분석 결과는 [CPU·GPU RI 그림](../results/csv/ri_cpu_gpu_20261119.png)으로 제공한다. 보관된 그림은 CPU·GPU 기준의 분석 결과이며, NPU를 포함한 분석에는 해당 NPU 측정 데이터가 추가로 필요하다.

## 6. 실험 설정 명명 규칙

`c`는 CPU의 A720 주파수, `g`는 GPU 주파수, `d`는 decode 구간에만 적용하는 A720 주파수를 나타낸다. 숫자의 단위는 MHz이며 `def`는 기본 정책이다.

| 예시 | 의미 |
|---|---|
| `freq_1500` | RAM KV, 요청 전체에서 CPU 1.5 GHz |
| `kv_1500` | KV 오프로딩, 요청 전체에서 CPU 1.5 GHz |
| `phase_1500` / `kvphase_1500` | RAM KV / 오프로딩, decode만 CPU 1.5 GHz |
| `gnaive` / `g1000_cdef` | GPU 단독 실행의 기본 설정 / GPU 1 GHz·CPU 기본 |
| `co_g1000_c1500` | 짧은 CPU 요청 여러 개와 GPU 요청을 동시 실행, GPU 1 GHz·CPU 전체 1.5 GHz |
| `ck_ram_gdef_cdef` | 긴 CPU 요청과 GPU 요청 동시 실행, RAM KV·기본 주파수 |
| `ck_swap_g1000_c1500` | CPU KV 오프로딩, GPU 1 GHz·CPU 요청 전체 1.5 GHz |
| `ck_swap_g1000_d1500` | 위와 같은 동시 실행에서 CPU는 decode만 1.5 GHz |
| `ck_swap_gov` | CPU KV 오프로딩, A720 주파수는 자동 관리 |

외부 부하는 온도와 메모리 경합 조건에 영향을 미치므로, 측정 중 다른 고부하 작업의 실행을 제한한다. 타 보드 적용 시 코어 번호, 주파수 정책, 센서·`devfreq` 경로 및 `perf` 실행 파일의 검증이 필요하다. 스크립트 주석에 남은 `docs/...` 경로는 이전 작업 저장소의 구조를 가리킬 수 있다. 이 배포본의 결과 해설은 [results/README.md](../results/README.md)에 있다.
