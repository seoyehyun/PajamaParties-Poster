#!/usr/bin/env bash
# KVSwap stage 1 measurement matrix. Run on the board from the llama.cpp directory.
# usage: scripts/kvswap/sweep_stage1.sh            (DRY=1 prints the plan only)
# env:   DEV=sda  CPUS/THREADS are passed through to run_stage1.sh
#        REPS=3  MEM_LIST="2G 1500M"  CELLS=400  NTOK=1000  RESULTS=results/<timestamp>
# order: repetitions are the outer loop so slow drift (temperature) spreads over all configs.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
REPS=${REPS:-3}; MEM_LIST=${MEM_LIST:-"2G 1500M"}; CELLS=${CELLS:-400}
RESULTS=${RESULTS:-runs/$(date +%Y%m%d_%H%M%S)}

# tag|mode|mem|G|M|prefetch
configs=("B0_base|base||||")
for mem in $MEM_LIST; do
  configs+=("B1_file_$mem|file|$mem|||")
  configs+=("B2_sel_g8_pf1_$mem|select|$mem|8|$((CELLS/8))|1")
  configs+=("B2_sel_g8_pf0_$mem|select|$mem|8|$((CELLS/8))|0")
  for g in 2 4 16; do
    configs+=("B2_sel_g${g}_pf1_$mem|select|$mem|$g|$((CELLS/g))|1")
  done
  for cells in 200 800; do
    configs+=("B2_sel_cells${cells}_pf1_$mem|select|$mem|8|$((cells/8))|1")
  done
done

echo "results -> $RESULTS ; ${#configs[@]} configs x $REPS reps"
for rep in $(seq 1 "$REPS"); do
  for c in "${configs[@]}"; do
    IFS='|' read -r tag mode mem g m pf <<<"$c"
    dir="$RESULTS/${tag}_r$rep"
    echo "== $tag rep $rep"
    [[ -n "${DRY:-}" ]] && { echo "   MEM_MAX=$mem G=$g M=$m PREFETCH=$pf run_stage1.sh $mode -> $dir"; continue; }
    # B1 reads the whole cache every step: cap its length so it finishes
    ntok=${NTOK:-1000}; [[ "$mode" == "file" ]] && ntok=${NTOK_FILE:-100}
    env OUTDIR="$dir" NTOK="$ntok" MEM_MAX="$mem" ${g:+G=$g} ${m:+M=$m} ${pf:+PREFETCH=$pf} \
      "$here/run_stage1.sh" "$mode" || echo "   (failed, continuing)"
    sleep "${COOLDOWN:-30}"                                     # let the board cool between runs
  done
done
[[ -z "${DRY:-}" ]] && python3 -I "$here/summarize_stage1.py" "$RESULTS" --dev "${DEV:-sda}"
