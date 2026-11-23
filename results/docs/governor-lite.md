# governor-lite: 지배 자원 기반 주파수 관리의 실행 중 적용 (2026-11-21, 보드 실행 2회, 각 n=1)

연구 방법 ③(지배 자원 기반 주파수 관리)은 지금까지 사전 계산 테이블([precomputed-table.md](precomputed-table.md))을 만들고 새 측정으로 검증한 데까지였다(E6, E7). 이 문서는 그 테이블을 **실행 중에 자동으로 조회해 적용하는** 최소 구현과, 그것이 **고정 설정보다 나은지**를 시험하는 방법이다. 결과는 실험 소개 영상에 넣을 수 있게 애니메이션으로 만든다.

| 파일 | 내용 |
|---|---|
| [scripts/joint/governor_lite.py](../../code/scripts/joint/governor_lite.py) | 0.5초마다 상태를 감지하고 테이블대로 A720 주파수를 바꾸는 데몬. 단독 실행도 가능(터미널 시연용) |
| [scripts/joint/thermal_ab.py](../../code/scripts/joint/thermal_ab.py) | 설정 `ck_<ram\|swap>_gov`(데몬이 주파수를 정함), 옵션 `--ck-gpu-delay`, `--ck-gpu-bursts`, `--ck-gpu-off`, `--gov-period`, `--gov-stable`. GPU 구간과 그 외 구간의 CPU 작업 속도(`cpu_Bps_on`, `cpu_Bps_off`)를 따로 낸다 |
| [scripts/joint/gov_replay.py](../../code/scripts/joint/gov_replay.py) | governor 실행을 기준선, 고정 설정들과 나란히 놓고 MP4(또는 GIF)와 정지 그림으로 재생. **노트북에서 실행**(matplotlib, ffmpeg) |

## 왜 동적이어야 하나

테이블에서 실행 중에 바뀌는 레버는 사실상 하나다: **GPU 작업이 돌면 A720을 1.5 GHz로 내리고, 끝나면 되돌린다.** 근거는 RI 행렬의 CPU 주파수 → GPU 성능 −0.57(동시 실행)이다. 나머지 칸은 기본 거버너와 같거나, 고정 설정("decode만 1.5 GHz")으로 충분하다. GPU를 1000 MHz로 고정하는 것은 동시 실행에서 효과가 없어서(열 1.01) 뺐다.

KV swap일 때는 상태마다 최적이 다르다. 그래서 고정 설정은 하나를 포기해야 한다.

| 구간 | A720 1.5 GHz | 기본 |
|---|---|---|
| CPU decode만 | 열 이득 없음(0.90 대 0.90), TPOT 약 14% 느림(76.8 대 67.1 ms) | **최적** |
| CPU decode + GPU | **최적**(E7: 열 0.81) | 열 0.90 |

governor는 두 구간에서 각각 최적을 고른다. 이 실험의 질문은 **"governor가 두 고정 설정의 장점을 함께 얻는가"**다.

**(11/21 재측정 후)** 위 표의 "CPU decode만" 칸은 11/14의 1회 측정(0.904 대 0.903)이었다. 같은 조건으로 다시 잰 결과(n=2, [results-20261121.md](results-20261121.md) 6c) KV swap에서도 decode 1.5 GHz가 첫 토큰 이후 열 0.78~0.84(decode 구간 0.75), TPOT naive 대비 1.02로 "최적"이었다. 그러니 KV swap에서도 decode는 내내 1.5 GHz가 최적이고, 이 시나리오에서 실행 중에 바뀌는 레버는 prefill → decode 전환뿐이다(고정 설정 "decode만 1.5 GHz"와 같은 결정). `governor_lite.py`의 이 칸은 11/21에 1500으로 고쳤다. 아래 결과는 고치기 전 규칙으로 돌린 것이다.

## 동작

| 감지 | 방법 |
|---|---|
| CPU 작업 | A720 사용률(`/proc/stat`) > 0.3이면 실행 중. 출력 파일(`out.txt`)이 비어 있으면 prefill, 첫 토큰이 나오면 decode(`phase_` 설정과 같은 신호) |
| GPU 작업 | Mali runtime-PM active 비율 > 0.3 |
| NPU | devfreq runtime-PM active 비율(기록만 함. NPU 발열은 센서 해상도 아래라 규칙 없음) |
| KV 위치 | 입력값. KV swap은 실행 시작 옵션이라 실행 중에 바꿀 수 없다 |

같은 상태가 2번 연속(1초) 보이면 적용한다. GPU는 항상 자기 거버너에 둔다.

| 상태 | A720 | 근거 |
|---|---|---|
| CPU prefill | 기본 | prefill은 연산 병목, 1.5 GHz면 TTFT +50.6% |
| CPU decode, GPU 없음, RAM KV | 1500 | decode만 1.5 GHz: 열 0.87, TTFT 1.00 |
| CPU decode, GPU 없음, KV swap | 1500 (11/21 수정, 이전 기본) | 재측정 n=2: 첫 토큰 이후 열 0.78~0.84, TPOT naive 대비 1.02. 이전 근거 "KV swap만 0.90, 1.5 GHz를 더해도 0.90"은 1회 측정 |
| CPU decode + GPU, KV swap | 1500 | E7: 열 0.81, CPU 작업 0.97, GPU 작업 0.79 |
| CPU decode + GPU, RAM KV | 1500 | 가장 가까운 측정 칸(요청 전체 1.5 GHz: 열 0.71, CPU 작업 1.07). decode만 1500은 미측정 |
| GPU만 / 대기 | 기본 | GPU만일 때 주파수를 낮추면 열이 늘었다(1.09~1.44) |

## 실험

긴 CPU decode(7K 프롬프트 + 6,500토큰, KV swap. 첫 실행은 4,500토큰이라 GPU 작업 수가 달랐다) 위에 GPU 작업(700토큰)을 세 번 넣는다. 첫 토큰 30초 뒤에 시작하고, 끝난 뒤 40초씩 쉰다. 작업량은 모든 설정에서 같다.

```
python3 scripts/joint/thermal_ab.py --prompt-chars 32000 --ntok 6500 --ctx 14336 \
    --ck-gpu-delay 30 --ck-gpu-bursts 3 --ck-gpu-ntok 700 --ck-gpu-off 40 \
    --configs ck_ram_gdef_cdef,ck_swap_gdef_cdef,ck_swap_gdef_d1500,ck_swap_gov --repeat 1
```

| 설정 | 역할 |
|---|---|
| `ck_ram_gdef_cdef` | 기준선(RAM KV, 기본 거버너) |
| `ck_swap_gdef_cdef` | 고정 1: KV swap + 기본(GPU 구간의 열을 놓침) |
| `ck_swap_gdef_d1500` | 고정 2: KV swap + decode 내내 1.5 GHz(GPU 없는 구간에서 느림) |
| `ck_swap_gov` | governor |

- 시간: 설정당 약 15~18분(prefill 약 340초 + decode 약 300~400초 + 식히는 꼬리), 4개 약 1~1.2시간. `--repeat 2`면 약 2~2.5시간. n=1이면 열 판정 노이즈 폭이 5%라 차이가 그 안에 들 수 있다.
- 결과: `summary.csv`의 `heat_post_per_tok`(첫 토큰 이후 열), `cpu_Bps_on`, `cpu_Bps_off`(GPU 구간, 그 외 구간의 CPU 작업 출력 속도, 바이트/초로 토큰 속도의 대리값), `gpu_tpot_ms`. 실행 폴더마다 `gov.csv`, `gov_events.jsonl`(governor), `cpu_progress.csv`, `trace.csv`.

### 판정

| 결과 | 해석 |
|---|---|
| governor의 첫 토큰 이후 열 ≈ 고정 2(차이 5% 이내) **그리고** GPU 없는 구간 CPU 속도 ≈ 고정 1(차이 5% 이내) | **두 고정 설정의 장점을 함께 얻음.** 동적 관리의 근거 |
| 열은 고정 2와 같지만 CPU 속도가 고정 2와 같음 | 전환이 늦거나 잘못됨. `gov.csv`의 상태 판정 확인 |
| 세 KV swap 설정의 열 차이가 노이즈(5%) 안 | GPU 구간이 짧아 열로는 구분 안 됨. 속도 차이만 결과로 쓰고, 반복을 늘리거나 GPU 구간을 늘림 |

## 결과 (2026-11-21, n=1)

위 실험 명령 그대로 한 번 실행했다(perf 켬). 원자료는 보드의 `/mnt/ssd/runs/gov_20261121`, 표는 [gov_runs.csv](../csv/gov_runs.csv), [gov_summary.csv](../csv/gov_summary.csv), 마지막 프레임은 [gov.png](../csv/gov.png)(영상 `gov.mp4`는 보드 실행 폴더에만 있음).

| 설정 | GPU 작업 | 첫 토큰 이후 열(zone8, 합계 °C·s) | CPU 속도 GPU 중 / 그 외 (B/s) | GPU TPOT | CPU TPOT | TTFT |
|---|---|---|---|---|---|---|
| `ck_ram_gdef_cdef` 기준선 | 2번 | 12,760 | 41.7 / 45.3 | 88.1 ms | 86.9 ms | 333 s |
| `ck_swap_gdef_cdef` 고정 1 | 2번 | 7,986 | 59.4 / **63.8** | 58.4 | **66.7** | 347 |
| `ck_swap_gdef_d1500` 고정 2 | 3번 | **5,493** | 51.4 / 55.5 | 51.5 | 77.1 | 347 |
| `ck_swap_gov` governor | 3번 | 6,615 | 53.5 / **63.3** | **49.4** | 71.4 | 347 |

**작업량이 같지 않았다.** GPU 작업 한 번이 90~145초로 예상보다 길어서, CPU decode가 일찍 끝난 기준선과 고정 1에는 세 번째 GPU 작업이 들어가지 못했다(GPU 1,398 대 2,097토큰). 같은 작업량끼리의 비교는 **governor 대 고정 2**뿐이다. 고정 1의 열은 GPU 작업 한 번이 빠진 값이라 governor에 불리한 쪽으로 치우친다. 다음 실행은 `--ntok 5500`(또는 `--ck-gpu-bursts 2`)로 모든 설정에 같은 수가 들어가게 한다.

**판정: 두 고정 설정의 장점을 함께 얻지는 못했다(중간 지점).**

- GPU 없는 구간 CPU 속도: governor 63.3 대 고정 1 63.8 B/s(−1%). 조건 충족.
- 첫 토큰 이후 열: governor 6,615 대 고정 2 5,493(+20%, 같은 작업량). 조건 미충족. 고정 1에서 고정 2까지 줄어드는 폭의 약 2/3만 얻었다(작업량 차이를 무시한 per-token 값으로 1.35 → 1.00 → 0.83).
- 고정 2 대비: GPU 없는 구간이 14% 빠르고(63.3 대 55.5) CPU TPOT 7% 빠르며(71.4 대 77.1) 열은 20% 많다. 열과 속도의 교환이지 우위가 아니다.
- 원인은 표의 "CPU decode, GPU 없음, KV swap → 기본" 칸이다. GPU 작업 사이 구간에서 governor는 2.3 GHz로 돌아가 zone8이 약 66 °C까지 오르고, 같은 구간의 고정 2는 약 56 °C다(gov.png 온도 그래프). 그 칸의 근거("KV swap만 0.90, 1.5 GHz를 더해도 0.90")는 11/14의 1회 측정이었다. 같은 조건으로 다시 재니(n=2) KV swap에서도 decode 1.5 GHz가 첫 토큰 이후 열 0.78~0.84, TPOT naive 대비 1.02로 "최적" 정의상 최적이었다(결과 6c). 즉 중간 지점은 열과 속도 사이의 정책 선택이 아니라 **1회 측정에 기댄 칸** 때문에 GPU 없는 구간의 열 이득을 놓친 결과다. 칸은 1500으로 고쳤고, 고친 governor로 다시 돌린 결과는 아래 "재실행"이다.

**보고서와 포스터에 쓰는 방법.** 보고서에는 한 문단으로 넣는다: "테이블을 실행 중에 적용하는 governor를 구현했고, 고정 설정 대비 속도 +14%·열 +20%의 교환 관계를 확인했다. 동적 적용이 주는 것은 단일 최적점이 아니라 정책 선택지다." 구현했다는 사실과 한계를 함께 보여 "관리는 안 했나"라는 질문에 답한다. 포스터에서는 빼거나 한 줄만 넣는다(공간 대비 메시지가 약함). 영상에는 "실행 중 적용 시연"으로 재생 영상을 쓴다(intro-video.md).

**(11/21 재측정 후, 위 문단 수정 필요)** "교환 관계, 정책 선택지"라는 해석은 위 원인 분석으로 근거가 약해졌다. 수정안(재실행 반영): "테이블을 실행 중에 적용하는 governor를 구현해 고정 설정과 비교했다. 첫 실행에서는 governor의 열이 가장 낮은 고정 설정보다 20% 많았는데, 원인은 1회 측정에 기댄 테이블 칸('KV swap이면 decode 주파수를 낮춰도 이득 없음')이었다. 그 칸을 다시 재서(n=2) 고친 뒤 재실행하니, governor는 단계 신호 없이 사용률과 출력만으로 그 고정 설정과 같은 결과(첫 토큰 이후 열 기준선 대비 0.32, KV offload만 쓸 때보다 36% 낮음)를 냈다."

보드에서 처음 확인할 것(위 목록)의 결과:

1. prefill 감지: 정상. 0~40초는 모델 로드라 대기, 40초부터 prefill, 387초부터 decode(실제 prefill 347초와 일치).
2. 흔들림: GPU 작업마다 시작 약 5초 뒤 1~2초짜리 잘못된 전환이 한 번 있었다. GPU 작업은 시작 후 약 20초가 모델 로드라 Mali가 실제로 쉰다. 실제 연산이 시작되면 1초 안에 1500으로 바뀌었다. 영향은 작아 `--gov-stable`은 그대로 둔다. decode 641표본 중 310(48%)이 1500.
3. 복원: 정상. 끝난 뒤 `schedutil`, 800~2300 MHz.
4. GPU 작업 세 번이 decode 안에 들어갔는가: 아니다(위 "작업량이 같지 않았다"). 고정 2와 governor의 세 번째 작업도 decode와 각각 61초, 30초만 겹쳤다.

## 재실행 결과: 고친 테이블 (2026-11-21, n=1)

"KV swap, GPU 없음" 칸을 1500으로 고친 `governor_lite.py`로 네 설정을 다시 돌렸다. 생성 길이를 6,500토큰(ctx 14336)으로 늘려 **모든 설정에 GPU 작업이 세 번** 들어갔다(GPU 2,097토큰, CPU 6,499토큰으로 같음). perf는 꺼져 있었다(이 실험의 지표와 무관). 원자료는 보드의 `/mnt/ssd/runs/gov2_20261121`, 표는 [gov2_runs.csv](../csv/gov2_runs.csv), [gov2_summary.csv](../csv/gov2_summary.csv), 마지막 프레임은 [gov2.png](../csv/gov2.png)다.

| 설정 | 첫 토큰 이후 열(zone8, 토큰당) | CPU 속도 GPU 중 / 그 외 (B/s) | GPU TPOT | CPU TPOT | TTFT |
|---|---|---|---|---|---|
| `ck_ram_gdef_cdef` 기준선 | 2.83 (1.00) | 38.2 / 39.8 | 88.4 ms | 92.9 ms | 333 s |
| `ck_swap_gdef_cdef` 고정 1 | 1.43 (0.50) | **60.1 / 63.7** | 58.5 | **66.0** | 346 |
| `ck_swap_gdef_d1500` 고정 2 | 0.92 (0.33) | 51.5 / 54.0 | **52.6** | 76.6 | 346 |
| `ck_swap_gov` governor | **0.92 (0.32)** | 51.5 / 54.0 | 52.7 | 76.7 | 346 |

**판정: governor는 고정 2와 같은 결과를 자동으로 냈다(열 차이 1%, 속도 차이 0.2% 안).** 고친 테이블에서는 decode 내내 1.5 GHz가 답이라 실행 중 전환은 prefill → decode 한 번이고(386초, 실제 prefill 346초 + 모델 로드 약 40초와 맞음), governor는 단계 신호를 받지 않고 A720 사용률과 출력 파일만으로 이 전환을 잡았다. GPU 작업 시작마다 1~1.5초짜리 상태 흔들림(decode + GPU → decode → decode + GPU)이 있었지만 두 규칙 모두 1500이라 주파수는 바뀌지 않았다.

- 고정 1 대비: 첫 토큰 이후 열 −36%, GPU TPOT −10%, GPU 없는 구간 CPU 속도 −15%. 열 최저(최적 정의) 쪽을 고른 결과다.
- 첫 실행의 "중간 지점"(고정 2 대비 열 +20%)은 사라졌다. 원인이 1회 측정에 기댄 칸이었다는 위 분석과 맞다.
- 기준선은 식는 구간이 `--tail-max` 안에 끝나지 않아(`tail_complete=0`) 열이 과소 측정됐다. 기준선 대비 비율은 실제보다 크게(보수적으로) 나온 값이다. 세 KV swap 설정끼리의 비교에는 영향이 없다.
- 한계: n=1. 이 시나리오에서는 "동적"으로 하는 일이 단계 전환 하나라, governor의 이점은 고정 설정을 사람이 단계별로 걸어 주지 않아도 된다는 데 있다. GPU 유무에 따라 값이 갈리는 칸(RAM KV 등)에서의 실행 중 전환은 아직 측정하지 않았다.

## 영상 만들기

보드에서는 matplotlib이 시스템 Python에 없어서 `/mnt/ssd/viz-venv/bin/python -I scripts/joint/gov_replay.py ...`로 만든다(약 4분). 노트북에서는 보드의 실행 폴더(`runs/thermal_<ts>/`)를 노트북으로 복사한 뒤:

```
python3 scripts/joint/gov_replay.py --gov runs/thermal_<ts>/ck_swap_gov_r1 --base runs/thermal_<ts>/ck_ram_gdef_cdef_r1 \
    --ref "고정: KV swap만=runs/thermal_<ts>/ck_swap_gdef_cdef_r1" --ref "고정: decode 1.5 GHz=runs/thermal_<ts>/ck_swap_gdef_d1500_r1" \
    --out gov.mp4 --still gov.png --speed 25
```

1920×1080 화면에 작업 타임라인(CPU prefill, decode, GPU 작업 세 번), A720 주파수, GPU 주파수, 온도가 시간에 따라 그려진다. 고정 설정 두 개는 점선으로 겹친다. 오른쪽에는 그 순간 governor가 감지한 상태, 테이블 규칙, 적용값, 근거(측정값), 전환 기록, 누적 열 면적이 나온다. 마지막 화면에서 각 설정의 열, TTFT, GPU 구간과 그 외 구간의 CPU 시간 배율, GPU TPOT 배율(기준선 대비)이 표로 나온다. `--speed 25`면 약 12분 실행이 30초 영상이 된다.

## 보드 없이 확인한 것

- 세 파일 `py_compile`, `thermal_ab.py --dry-run`(새 설정과 옵션).
- **버스트 실행 경로**: 가짜 CPU 작업(출력 파일에 쓰고, GPU 작업 중에는 2배 느리게 씀), 가짜 GPU 작업, 가짜 `taskset`으로 `CoKV`를 그대로 돌렸다. GPU 작업 세 번이 지정한 간격으로 시작되고 종료 코드가 모였고, CPU 출력 속도가 GPU 구간 52 B/s, 그 외 87 B/s로 나뉘었다(가짜 작업의 2배 감속과 맞음).
- **governor 판단**: 가짜 센서로 버스트 시나리오를 돌렸다. 전환 8번이 상태 변화 후 1초 안에 일어났다(decode 347.5 s, GPU 합류 377.5/467.5/557.5 s, GPU 종료 427.5/517.5/607.5 s, 종료 635.5 s).
- **재생**: 합성 온도 곡선과 위 로그로 MP4와 정지 그림을 만들어 화면 구성을 확인했다. **합성 데이터라 숫자는 의미가 없다.**

## 보드에서 처음 확인할 것

1. A720 사용률 기준(0.3)이 prefill 시작(모델 로드, 캐시 비우기) 동안 대기로 잘못 판정되지 않는지. `gov.csv`의 `cpu_phase` 열로 본다.
2. GPU 작업의 짧은 prefill과 종료 직후에 `gpu_busy`가 0.3 근처에서 흔들려 전환이 반복되지 않는지. 반복되면 `--gov-stable 4`.
3. 데몬의 주파수 쓰기와 `thermal_ab.py`의 복원이 충돌하지 않는지. 실행이 끝나면 `scaling_governor`가 원래대로 돌아와야 한다.
4. 모든 설정에서 GPU 작업 세 번이 CPU decode 안에 들어갔는지(`runs.jsonl`의 `gpu_bursts`와 `t1`). 고정 2는 decode가 가장 길고, 기준선(RAM KV)은 컨텍스트가 길어질수록 느려진다.

## 한계

- 상태 감지는 휴리스틱이다(사용률 기준, 출력 파일). 일반적인 앱에서 prefill과 decode를 구분하려면 추론 엔진이 알려 주는 신호가 필요하다.
- CPU 속도는 출력 바이트로 잰 대리값이다. 같은 프롬프트와 시드라 설정 간 비교에는 쓸 수 있지만 토큰/초는 아니다.
- 테이블은 측정한 칸만 근거가 있다. RAM KV 동시 실행의 decode만 1500 칸은 미측정이라 가장 가까운 칸을 쓴다.
- KV 위치, GPU, NPU는 조절하지 않는다. 메모리 주파수는 이 보드에서 바꿀 수 없다.
