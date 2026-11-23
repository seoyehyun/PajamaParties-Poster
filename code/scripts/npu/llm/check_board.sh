#!/usr/bin/env bash
# Read-only readiness check for running an LLM on the CIX P1 NPU (docs/npu/npu-llm-plan.md). Run on the board. NOT yet run.
# Changes nothing. Prints one line per item: OK / MISSING / CHECK, so the output can be pasted into the plan.
#   route A (cixbuild .cix + NOE_Engine): needs /dev/aipu, libnoe, NOE_Engine in python, an x86 PC for cixbuild
#   route B (tinygrad ZHOUYI backend, github.com/Reonarudo/tinygrad-npu-examples): needs /dev/aipu, the UAPI header,
#           the vendor toolchain libs, python >= 3.12, numpy, tokenizers, libclang, and ideally the WBUF/SLOT ioctls
set -u
ok()   { printf 'OK      %-28s %s\n' "$1" "$2"; }
miss() { printf 'MISSING %-28s %s\n' "$1" "$2"; }
chk()  { printf 'CHECK   %-28s %s\n' "$1" "$2"; }

echo "== board"; uname -r; grep -m1 MemTotal /proc/meminfo; df -h / /mnt/ssd 2>/dev/null | tail -n +2

echo "== NPU device and driver"
[[ -c /dev/aipu ]] && ok /dev/aipu "$(ls -l /dev/aipu | awk '{print $1,$3,$4}')" || miss /dev/aipu "no character device"
lsmod | grep -qiE "aipu|armchina" && ok kernel-module "$(lsmod | grep -iE 'aipu|armchina' | head -2 | tr '\n' ';')" || chk kernel-module "not in lsmod (built in?)"
modinfo aipu 2>/dev/null | grep -E "^(filename|version|vermagic)" | head -3
dkms status 2>/dev/null | grep -iE "aipu|zhouyi" || chk dkms "no aipu entry in dkms status"
ls -d /usr/src/aipu* /usr/src/zhouyi* 2>/dev/null

echo "== UAPI header (route B looks here, in this order)"
cands=(${ZHOUYI_AIPU_HEADER:-} /usr/src/zhouyi-aipu-*/include/armchina_aipu.h /usr/include/misc/armchina_aipu.h
       /usr/include/linux/armchina_aipu.h /usr/share/cix/include/npu/kmd/armchina_aipu.h)
hdr=""; for h in "${cands[@]}"; do [[ -f $h ]] && { hdr=$h; break; }; done
[[ -z $hdr ]] && hdr=$(find /usr/src /usr/include /usr/share/cix -name 'armchina_aipu.h' 2>/dev/null | head -1)
if [[ -n $hdr ]]; then
  ok armchina_aipu.h "$hdr"
  for k in WBUF SLOT BUF_CACHE_INVALID; do
    n=$(grep -c "$k" "$hdr"); [[ $n -gt 0 ]] && ok "ioctl $k" "$n lines in header" || chk "ioctl $k" "not in header (route B: weights copied every token, or reads stale)"
  done
else
  miss armchina_aipu.h "not found: route B cannot generate its driver bindings"
fi

echo "== vendor toolchain (route B loads it in-process)"
T=${ZHOUYI_TOOLCHAIN_DIR:-/usr/share/cix/lib/onnxruntime}
ls "$T" 2>/dev/null | grep -E "aipu|toolchain" | sed 's/^/        /'
for l in libaiputoolchain.so libaiputoolchain_core.so libaipu_buildingtool.so libaipu_toolchain_core.so libaipu_buildtool.so; do
  [[ -f $T/$l ]] && ok "$l" present || chk "$l" "absent (the tinygrad README names libaipu_toolchain_core.so / libaipu_buildtool.so; this package may name them differently)"
done
dpkg -l cix-npu-onnxruntime cix-noe-umd cix-npu-driver 2>/dev/null | awk '/^ii/{print "        "$2,$3}'

echo "== python"
python3 --version
for m in numpy tokenizers libnoe NOE_Engine onnxruntime; do
  python3 -c "import $m" 2>/dev/null && ok "py:$m" "$(python3 -c "import $m;print(getattr($m,'__version__',''))" 2>/dev/null)" || miss "py:$m" "not importable by python3 (may live in ~/npu-models/.pydeps)"
done
PYTHONPATH=$HOME/npu-models/.pydeps python3 -c "import numpy,sys;print('        with .pydeps: numpy',numpy.__version__)" 2>/dev/null
ls /usr/lib/llvm-*/lib/libclang*.so* 2>/dev/null | head -2 | grep -q . && ok libclang "$(ls /usr/lib/llvm-*/lib/libclang*.so* | head -1)" || miss libclang "needed once by route B to generate bindings (apt install libclang-dev)"
command -v git >/dev/null && ok git "$(git --version)" || miss git ""

echo "== NPU clocks (devfreq, what npu_freq_run.sh sets)"
D=/sys/class/devfreq/CIXH4000:00
if [[ -d $D ]]; then
  echo "        governor=$(cat $D/governor) cur=$(cat $D/cur_freq) avail=$(cat $D/available_frequencies)"
  chk npu_memclk "the tinygrad README quotes npu_memclk 750 MHz; no sysfs node for it is known on this board"
else miss "$D" "no NPU devfreq node"; fi

echo "== network (needed to clone and to download weights)"
ping -c1 -W2 8.8.8.8 >/dev/null 2>&1 && ok internet reachable || chk internet "not reachable (docs/npu/npu-llm-plan.md 0.)"
