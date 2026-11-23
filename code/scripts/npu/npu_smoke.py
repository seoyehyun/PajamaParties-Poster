#!/usr/bin/env python3
"""Time a compiled .cix model on the NPU with dummy inputs. Measures speed only, not output quality."""
import json
import os
import resource
import statistics as st
import sys
import time

import numpy as np
from NOE_Engine import EngineInfer

model, reps = sys.argv[1], int(sys.argv[2])
freq_dir = "/sys/class/devfreq/CIXH4000:00"


def read(name):
    with open(f"{freq_dir}/{name}") as f:
        return f.read().strip()


t0 = time.perf_counter()
eng = EngineInfer(model)
load_s = time.perf_counter() - t0

inputs = []
for i, d in enumerate(eng.in_tensor_desc):
    # Mid-scale zeros after the engine's quantization step; sized from the tensor description.
    n = d.size // np.dtype(eng.input_type[i]).itemsize
    inputs.append(np.zeros(n, dtype=np.float32))

# Optional third arg: directory of input<i>.bin files, already quantized (byte-exact tensor sizes).
raw = None
if len(sys.argv) > 3:
    raw = [open(f"{sys.argv[3]}/input{i}.bin", "rb").read() for i in range(len(inputs))]
    assert [len(b) for b in raw] == [int(d.size) for d in eng.in_tensor_desc], "input size mismatch"


def run_raw():
    eng._create_job()
    for i, b in enumerate(raw):
        eng.npu.noe_load_tensor(eng.job_id, i, b)
    t = time.perf_counter()
    eng.npu.noe_job_infer_sync(eng.job_id, -1)
    d = (time.perf_counter() - t) * 1000
    eng.npu.noe_clean_job(eng.job_id)
    return d


cpu0 = time.process_time()
wall0 = time.perf_counter()
times = []
during = None
for r in range(reps):
    if raw:
        times.append(run_raw())
    else:
        eng.forward(inputs)
        times.append(eng._dur_time * 1000)
    if r == reps // 2:
        during = (read("cur_freq"), read("target_freq"))
wall = time.perf_counter() - wall0
cpu = time.process_time() - cpu0

steady = times[1:] if len(times) > 1 else times
print(json.dumps({
    "model": os.path.basename(model),
    "load_s": round(load_s, 2),
    "inputs": [int(d.size) for d in eng.in_tensor_desc],
    "input_types": [np.dtype(t).name for t in eng.input_type],
    "reps": reps,
    "input_source": "bin" if raw else "dummy",
    "first_ms": round(times[0], 2),
    "mean_ms": round(st.mean(steady), 2),
    "median_ms": round(st.median(steady), 2),
    "stdev_ms": round(st.pstdev(steady), 2),
    "cur_target_during": during,
    "cpu_per_wall": round(cpu / wall, 3),
    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
}))
sys.stdout.flush()
os._exit(0)  # libnoe aborts in interpreter teardown ("pure virtual method called"); results are already out
