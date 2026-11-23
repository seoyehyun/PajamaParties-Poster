# 보드에서 확인한 사실 (Orange Pi 6 Plus, 2026-11-13)

2026-11-13에 보드를 직접 확인하면서 알게 된 것만 모았다. 이전 확인은 board-results-2026-11-03.md와 board-freq-config.md에 있고, 여기서는 그와 다르거나 새로 확인한 것을 적는다. 읽기 전용으로 확인한 것이 대부분이고, 확인하지 못한 것은 그렇게 적었다. 접속 정보는 저장소에 올리지 않는다.

## 메모리

| 항목 | 확인 내용 |
|---|---|
| 종류와 속도 | LPDDR5, Configured Memory Speed 6400 MT/s (`dmidecode`). RAM 32 GB |
| 컨트롤러 | DDR 컨트롤러 4개(`ddrc0~3`). `/sys/kernel/debug/clk/clk_summary`에 `ddrc*_aclk` 1066.7 MHz, `ddr_phy*_sync`와 `ddrc*_dficlk` 약 800 MHz, `npu_memclk` 750 MHz가 보인다 |
| **주파수 조절** | **불가능.** 위 클럭은 읽기 전용(debugfs `clk_*` 파일이 `r--r--r--`)이고 활성화 카운트가 0인 `deviceless` 항목이다. `devfreq`에는 비디오 코덱(`CIXH3010:00`), NPU(`CIXH4000:00`), GPU(`CIXH5000:00`)만 있고 메모리 노드가 없다. `cix-ddr-lp` 드라이버는 조작용 sysfs를 내지 않는다. 바꿀 수 있다면 펌웨어(UEFI) 설정뿐일 텐데 확인하지 못했다 |
| 대역폭 | CPU 8코어(A720) 읽기 천장 약 39~40 GB/s, 쓰기 약 56 GB/s(`membw_probe`). GPU OpenCL 전역 메모리 대역폭 최대 48.8 GB/s(`clpeak`, GPU 1000 MHz). DRAM 이론 한계는 확인하지 못했다. 자세한 값은 [cpu-gpu-memory-sweeps-20261113.md](cpu-gpu-memory-sweeps-20261113.md) |

## 성능 카운터(PMU)

- `perf_event_paranoid=2`라서 일반 사용자는 못 읽는다. 커널 6.6.89용 `perf` 래퍼가 없고 `linux-tools-6.8.0-147`만 설치돼 있다. `/usr/lib/linux-tools/6.8.0-147-generic/perf`를 sudo로 직접 부르면 동작한다.
- PMU는 두 개다. `armv8_pmuv3_0` = cpu0-1,6-11(A720), `armv8_pmuv3_1` = cpu2-5(A520). 읽기 대역폭에는 `l2d_cache_refill × 64 B`가 맞았고 `LLC-load-misses`(`ll_cache_miss_rd`)는 대역폭이 아니다(보정 표는 위 결과 문서 4절).
- **uncore PMU는 없다.** `/sys/kernel/debug/arm-cmn`은 비어 있고 `perf list`에 CMN, DDRC 이벤트가 없다. GPU의 메모리 트래픽은 CPU PMU로 볼 수 없다.

## 전력과 에너지

- 전력 센서가 없다. `power_supply`에는 AC만 있고 `powercap`은 없다. OPP(`/sys/kernel/debug/opp`)의 `supply-0`이 비어 있어 전압도 읽을 수 없다.
- **커널 Energy Model은 있다**(`/sys/kernel/debug/energy_model/`, root 필요). 정책 대표 CPU(cpu0, 2, 6, 8, 10)와 GPU(`CIXH5000:00`)에 항목이 있다. cpu6(A720)의 표는 다음과 같다(단위와 절대값은 신뢰할 근거가 없는 모델 값이다. 정적 전력, 메모리, 스토리지는 포함돼 있지 않다).

| MHz | 800 | 1200 | 1500 | 1800 | 1900 | 2000 | 2100 |
|---|---|---|---|---|---|---|---|
| power | 500 | 1500 | 1900 | 2450 | 2633 | 2816 | 3000 |

  표는 2.1 GHz까지이고 cpu6 정책 상한(2.3 GHz)과 우리가 쓴 2.2 GHz는 없다.
- 이 표만으로 에너지를 계산하면 800 MHz가 가장 효율적으로 나오는데(2.1 GHz 대비 decode 토큰당 코어 에너지 약 0.43배), 정적 전력과 메모리 전력이 빠져서 실제 최적점은 더 높을 것이다. 이 점이 전력계 없이는 "에너지 최적" 주장이 약한 이유다.

## 열

- 열 존 13개가 모두 `acpitz`라서 이름으로 CPU, GPU를 구분할 수 없다. 거버너는 `power_allocator`.
- 트립: 모든 존 critical 98 °C. 존 7, 8, 9, 10은 passive 85 °C와 60 °C, 존 12는 passive 85 °C와 70 °C가 더 있다. 쿨링 장치는 `Processor` 12개(최대 상태 10)와 GPU `devfreq-CIXH5000:00` 1개(최대 상태 5)다. 팬 노드는 보이지 않는다.
- 유휴 34~38 °C. 오늘 스윕에서 점 종료 시 최고 64 °C(CPU 스윕), 65 °C(GPU 스윕), 68 °C(동시 실행)였고 스로틀 흔적은 없었다. 어느 존이 어느 트립에 해당하는지는 모른다.
- NVMe 온도 센서는 `hwmon`에 노출되지 않는다(`nvme` CLI도 없음). KV swap의 NVMe 발열은 읽을 수 없다.

## 주파수 제어

- A720은 정책 4개(`policy0` cpu0-1 최대 2.6 GHz, `policy6` cpu6-7 2.3, `policy8` cpu8-9 2.2, `policy10` cpu10-11 2.5), A520은 `policy2`(cpu2-5, 최대 1.8 GHz). 드라이버는 `cppc_cpufreq`이고 `scaling_available_frequencies`는 없다. 4개 정책을 같은 값으로 맞추려면 공통 상한 2.2 GHz를 넘길 수 없다.
- **`scaling_cur_freq`와 `cpuinfo_cur_freq`는 요청값을 되풀이한다.** 실제 주파수는 `acpi_cppc/feedback_ctrs`로 봐야 하고, 요청은 다음 단계로 올림된다(11/10 단계 탐색, [freq_steps_20261110.csv](../csv/freq_steps_20261110.csv)). 오늘 쓴 값의 실제 주파수(MHz, 정책 0 / 6 / 8 / 10): 800 → 800/800/800/800, 1000 → 1200/1200/1200/1200, 1200 → 1500/1200/1200/1200, 1500 → 1500/1500/1500/1500, 1800 → 1900/1800/1800/1900, 2200 → 2500/2200/2200/2200.
- GPU(`CIXH5000:00`)는 72, 216, 350, 600, 800, 1000 MHz(기본 `simple_ondemand`, min=max로 고정). NPU는 400, 600, 800, 1200 MHz(`userspace`).
- sudoers는 `tee`(cpufreq, devfreq, `drop_caches`)와 `true`만 비밀번호 없이 허용한다. 그래서 주파수 고정과 KV swap 실행(`run_stage1.sh`)은 비밀번호 없이 되지만 `perf`는 비밀번호가 필요하다.

## GPU와 실행 파일

- Mali-G720-Immortalis(clpeak 기준 10 컴퓨트 유닛), Vulkan 1.3.296, **OpenCL도 동작한다**(`clpeak -p 0`이 ARM Platform의 Mali를 인식). 드라이버는 `cix-gpu-umd` 2.0.0.
- llama.cpp 빌드: `~/llama-base/build`(CPU, KV swap 패치 포함), `build-vulkan`(Vulkan), `build-p2`(CPU). 세 빌드 모두 `libllama.so`에 `LLAMA_KV_*` 환경변수 문자열이 있어 패치가 들어 있다. `~/llama.cpp/llama-cpu`, `llama-gpu`는 오늘 확인 시점에 라이브러리 경로(`~/llama.cpp/build/bin`)가 비어 있어 실행되지 않았다. 오늘 스윕은 위 세 빌드의 `llama-bench`를 썼다.
- KV swap 실행에 필요한 파일은 보드에 있다: 프롬프트 `/mnt/ssd/prompts/`(`haystack_c4.txt`, `p_8k.txt`), 예측기 어댑터 `/mnt/ssd/adapters/kvswap-adapter-r{32,64,128}.gguf`, NVMe(`/mnt/ssd`, 458 GB).

## 기타

- 보드는 인터넷이 없다(DNS 해석 실패). 모델과 파일은 PC에서 받아 SSH로 흘려보냈다.
- **보드 시계를 믿을 수 없다.** 11/13 기준 2026-11-10으로 표시됐고, 재부팅할 때마다 마지막으로 저장된 시각(`fake-hwclock`, 시간당 저장)으로 되돌아간다. 부팅 기록(`last reboot`)이 모두 정각(14:00, 06:00, 15:00 등)으로 찍히는 것은 이 때문이다(실제 재부팅 시각이 아니다). 같은 재부팅 직전에 끝난 실행이 15:00:28, 그 뒤 실행 시작 줄이 15:03:41로 찍혔는데 부팅 기록은 15:00:03이었다. 보드에서 만든 폴더 이름의 날짜는 이 값이라서 사건 시각은 이 PC 시계로 맞춘다.
- NPU 드라이버(`aipu` KMD 5.11.0)에는 WBUF/SLOT ioctl이 없고 벤더 런타임은 정상이다. 자세한 내용은 npu-llm-plan.md 7절.

## 비정상 재부팅

- **정상 종료 기록 없이 재부팅한 흔적이 11/3부터 있다**(`last`에 `crash` 표시, `shutdown` 기록 없음). 11/14 실험 중에는 3번 일어났다. 이 세션의 시도 횟수로 보면 RAM KV 실행 12회에 0번, KV swap(NVMe 직접 I/O) 실행 16회에 3번이다. 표본이 작아 통계적으로 유의하지 않다(p ≈ 0.17).
- 재부팅 직전 2초 간격으로 저장한 커널 로그에 **오류가 없다**(마지막 줄이 KV swap 실행 시작 때의 `drop_caches`). `pstore`는 비어 있고 영구 저널은 없다(로그는 램에 있다가 주기적으로 디스크로 옮겨져서, 갑자기 끊기면 사라진다). 커널이 기록할 새 없는 리셋이라 전원 또는 하드웨어 쪽 문제로 추정한다(미확인). 전원 어댑터를 점검해야 한다.
- cron, systemd 타이머, `at`에 재부팅 작업은 없고 워치독은 꺼져 있다. 앞서 "정각 재부팅이라 외부 전원 제어 같다"고 본 추정은 위 시계 문제 때문에 근거가 없다.
- **이 PC의 절전(최신 대기 모드)이 ssh 세션을 끊는다.** 보드는 절전하지 않는다(자동 절전 없음, `suspend` 기록 없음). 절전과 겹친 시각에 띄운 `nohup` 실행 하나가 약 10분 뒤 죽었고, `setsid`로 분리해서 띄운 실행은 PC가 자는 동안에도 살아남았다. 보드에서 긴 작업을 띄울 때는 `setsid`로 분리하고 ssh 호출을 짧게 끝낸다.
