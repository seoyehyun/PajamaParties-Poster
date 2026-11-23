#!/usr/bin/env bash
# exp_queue.sh -- the board experiments of docs/report-experiment-plan.md, one after another, resumable after a reboot.
#
#   setsid nohup bash scripts/exp_queue.sh >> /mnt/ssd/runs/q20261120/queue.log 2>&1 < /dev/null &
#   tail -f /mnt/ssd/runs/q20261120/queue.log
#   ONLY="e5 e6_cpu" bash scripts/exp_queue.sh        # run only these steps (still skips finished ones)
#
# A finished step leaves $OUT/<step>.done and is skipped next time, so after a reboot the same command continues.
#   thermal_ab / corun_kvswap steps resume their own run directory (finished runs are kept);
#   sweep_npu / sweep_corun steps restart (the partial directory is renamed to <dir>.partial<N>).
# A failed step is logged and the queue moves on; the next start retries it.
# Before (re)starting: `sudo sysctl kernel.perf_event_paranoid=-1` (lost on reboot), else the bus_GBps columns stay empty.
# Do not run other heavy work on the board meanwhile: every step measures speed and temperature.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-/mnt/ssd/runs/q20261120}
mkdir -p "$OUT"
cd "$REPO"
LOCK="$OUT/queue.lock"
exec 9> "$LOCK"
flock -n 9 || { echo "another exp_queue.sh is running ($LOCK)"; exit 1; }

STOP=0
trap 'STOP=1; echo "[queue] stop requested"; kill -TERM "$CHILD" 2>/dev/null' TERM INT
CHILD=0

T7="--prompt-chars 32000 --ntok 2000 --ctx 10240"                    # CPU-only 7K request (thermal-ab-20261113/-phase-20261114)
TG="--prompt-chars 6200 --ntok 1000 --ctx 4096"                      # GPU-only and co-run request (thermal-ab-gpu/-co-20261118)
TCK="--prompt-chars 32000 --ntok 3000 --ctx 12288"                   # ck: 7K CPU job + GPU job from its first token (thermal-ab-ck-20261119)
NPU_CMD="python3 $REPO/scripts/npu/llm/npu_llm_run.py --phase {phase} --persistent"

log() { echo "[queue $(date '+%m-%d %H:%M:%S')] $*"; }

# run_step <name> <kind: thermal|corunkv|fresh> <python script> <args...>
run_step() {
  local name=$1 kind=$2; shift 2
  local dir="$OUT/$name"
  if [[ -n "${ONLY:-}" && " $ONLY " != *" $name "* ]]; then return; fi
  if [[ -f "$OUT/$name.done" ]]; then log "skip $name (done)"; return; fi
  [[ $STOP == 1 ]] && return
  local where=()
  case $kind in
    thermal|corunkv) if [[ -f "$dir/meta.json" ]]; then where=(--resume "$dir"); else where=(--out-dir "$dir"); fi ;;
    fresh) if [[ -e "$dir" ]]; then local n=1; while [[ -e "$dir.partial$n" ]]; do n=$((n + 1)); done; mv "$dir" "$dir.partial$n"; fi
           where=(--out-dir "$dir") ;;
  esac
  log "start $name: paranoid=$(cat /proc/sys/kernel/perf_event_paranoid) temp=$(( $(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1) / 1000 ))C"
  log "  python3 $* ${where[*]}"
  python3 "$@" "${where[@]}" > "$OUT/$name.log" 2>&1 &
  CHILD=$!
  wait "$CHILD"; local rc=$?
  if [[ $STOP == 1 ]]; then
    pgrep -g $$ | grep -vx $$ | xargs -r kill -TERM 2>/dev/null   # children the step left behind (run_stage1.sh's llama-completion)
    log "stopped during $name (rc=$rc); run the queue again to continue"; return
  fi
  if [[ $rc == 0 ]]; then touch "$OUT/$name.done"; log "done $name"; else log "FAILED $name (rc=$rc), see $OUT/$name.log"; fi
}

log "queue start (out $OUT, git $(git rev-parse --short HEAD))"
# Reduced plan (2026-11-20): points that earlier runs already measured are reused, not re-run, and every step runs ONCE
# (one round / one trial, user decision: no repeats; the noise margin falls back to --min-gain and n=1 is marked in the report).
#   reused: 11/14 naive, kv_default, phase_1500 (n=2), kvphase_1500 (n=1); 11/18 GPU-only table rows (no gain to validate);
#           11/13 CPU x GPU decode co-run A720 {1200,2200} x GPU {350,600,1000}; 11/19 ck baseline and 3K/7K/11K context runs.

# P0  E5: does KV swap change the A720 frequency curve of decode? (C3)  TPOT + utilisation only: the heat-tail is cut to a
#     3-minute cool-down (heat is not used from this step). 1500 MHz comes from 11/14 (phase_1500, kvphase_1500).
run_step e5 thermal scripts/joint/thermal_ab.py $T7 --repeat 1 --tail-max 180 \
  --configs phase_1200,phase_1800,phase_2200,kvphase_1200,kvphase_1800,kvphase_2200

# P0  E6: independent check of the precomputed table (C6), new rounds: CPU-only and short-context co-run rows.
#     (GPU-only row = "leave it", gain 0.99: not re-run. Long-context co-run row: e7_ck below.)
run_step e6_cpu thermal scripts/joint/thermal_ab.py $T7 --repeat 1 --configs naive,phase_1500,kv_default
run_step e6_co  thermal scripts/joint/thermal_ab.py $TG --repeat 1 --configs co_gdef_cdef,co_g1000_c1500

# P0 (NPU)  E1 + E2 + E3 + E1b: decode with the 3-way grid and the fixed-work heat jobs; prefill without cross-solo and 3-way jobs.
#     Co-run grids NPU {400,1200} x A720 {1200,2200} / GPU {350,1000}, 1 trial.
NPU_GRID="--npu-coru-freqs 400,1200 --gpu-coru-freqs 350,1000 --trials 1"
run_step npu_decode fresh scripts/sweep/sweep_npu.py --phase decode --npu-persistent --npu-cmd "$NPU_CMD" --npu-model-bytes 4.8e8 \
  $NPU_GRID --with-heat --heat-iters 120
run_step npu_prefill fresh scripts/sweep/sweep_npu.py --phase prefill --npu-persistent --npu-cmd "$NPU_CMD" $NPU_GRID \
  --jobs npu_solo,cpu_solo,gpu_solo,corun_npu_cpu,corun_npu_gpu

# P1  E7 (+ E6 long-context row): co-run with the decode-only A720 1500, baseline and swap/cdef in the same batch
run_step e7_ck thermal scripts/joint/thermal_ab.py $TCK --repeat 1 --configs ck_ram_gdef_cdef,ck_swap_g1000_cdef,ck_swap_g1000_d1500

# P1  E4: CPU x GPU co-run. decode: only the cells 11/13 did not measure (with their own solo denominators); prefill: 4 x 4.
run_step e4_decode_mid fresh scripts/sweep/sweep_corun.py --freqs 1500,1800 --gpu-freqs 350,600,800,1000 --trials 1 --corun-phase decode
run_step e4_decode_800 fresh scripts/sweep/sweep_corun.py --freqs 1200,2200 --gpu-freqs 800 --trials 1 --corun-phase decode
# prefill: the full 4 x 4 grid, so the co-run prefill RI is not a 2-point Spearman (named _4x4: the first queue had a 2 x 2 step)
run_step e4_prefill_4x4 fresh scripts/sweep/sweep_corun.py --freqs 1200,1500,1800,2200 --gpu-freqs 350,600,800,1000 --trials 1 \
  --corun-phase prefill --gpu-reps 4

# P2  E8: KV swap x GPU interference at the frequencies 11/19 did not cover (C4). The 3K loss stays as measured on 11/19 (n=2).
run_step e8_7k corunkv scripts/joint/corun_kvswap.py --a720 1200,1800 --gpu-freqs 350,800 --trials 1

# P2  E9: KV swap break-even on a small model (Qwen2.5-0.5B Q4_K_M, KV row 128 values): contexts 256 / 4K / 16K, RAM vs swap.
#     Adapter r=32 (sigma 4; r=16 keeps only 0.81 of the key energy). G=16: a group of 8 cells is 2 KiB, below the 4 KiB O_DIRECT
#     alignment; M=25 keeps the same 400 selected tokens as M=50 x G=8. Built on the board (scripts/kvswap/make_kvswap_adapter.py).
T05="--model /mnt/ssd/models/q05/qwen2.5-0.5b-instruct-q4_k_m.gguf --adapter /mnt/ssd/adapters/q05/kvswap-adapter-r32.gguf --g 16 --m 25 \
  --ntok 1000 --pre 20 --tail-max 120 --repeat 1 --configs naive,kv_default"
run_step e9_256 thermal scripts/joint/thermal_ab.py $T05 --prompt-chars 1100 --ctx 2048
run_step e9_4k  thermal scripts/joint/thermal_ab.py $T05 --prompt-chars 18500 --ctx 6144
run_step e9_16k thermal scripts/joint/thermal_ab.py $T05 --prompt-chars 73000 --ctx 20480

# E6 short-context co-run again with the fixed utilisation window (a2241a8): e6_co's _d columns covered the time after the GPU
# job had ended. Also gives this borderline row (CPU-job TPOT 1.10x) a second round.
run_step e6_co2 thermal scripts/joint/thermal_ab.py $TG --repeat 1 --configs co_gdef_cdef,co_g1000_c1500

log "queue end"
