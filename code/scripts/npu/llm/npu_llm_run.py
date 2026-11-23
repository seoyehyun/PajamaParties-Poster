#!/usr/bin/env python3
"""npu_llm_run.py -- run the Qwen2.5-0.5B prefill or decode .cix on the NPU and print NPU_RESULT lines for scripts/sweep/sweep_npu.py.

  NPU_RESULT {"tps": <tokens/s>, "tokens": <n>, "ms_per_tok": ..., "npu_ms": ..., "host_ms": ..., "npu_mhz": ...}

One result = one unit of work:
  prefill  one inference of the S=256 token prompt (tokens = 256)
  decode   --decode-tokens consecutive decode steps (default 32). Every step feeds the argmax token back, and the KV window slides by one row
           (the 255 past rows drop the oldest and append the new present row), so the host work of a real decode loop is in the timing.
tps = tokens / wall time of the unit, host work included (load inputs, infer, read outputs, argmax, KV shift). npu_ms is the infer_sync
time alone, host_ms the rest.

  --persistent   loop until killed (SIGTERM/SIGINT), one line per unit: the sweep's --npu-persistent mode
  --iters N      print N results and exit (default 1): the sweep's one-process-per-iteration mode, and for tests

The graphs (scripts/export_qwen_onnx.py, int8, S=256, P=255) include the embedding and the head:
  prefill  in: input_ids [1,256] int32                         out: logits [1,151936], present_{key,value}_i [1,2,256,64] x 24 layers
  decode   in: input_ids, position_ids [1,1] int32, past_* [1,2,255,64] x 48     out: logits, present_* [1,2,1,64] x 48
Decode starts from an all-zero past window (no prefill graph is loaded: two 480 MB graphs in one process risk the ~1 GiB NPU address window).
After 255 steps the window holds only generated rows. The latency does not depend on the values; the text is not meaningful (int8, see
docs/npu/npu-llm-plan.md), and we measure speed only.

  python3 scripts/npu/llm/npu_llm_run.py --phase decode --iters 5
  python3 scripts/sweep/sweep_npu.py --npu-persistent --phase decode \
      --npu-cmd 'python3 $HOME/PajamaParties/scripts/npu/llm/npu_llm_run.py --phase {phase} --persistent'
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:                 # the board has no system numpy; NOE_Engine needs it too (sweep_npu.py runs `taskset ... <cmd>`, no env prefix)
    sys.path.insert(0, os.path.expanduser("~/npu-models/.pydeps"))
    import numpy as np
from libnoe import NOE_TENSOR_TYPE_INPUT, NOE_TENSOR_TYPE_OUTPUT  # noqa: F401  (NOE_Engine re-exports the same module)
from NOE_Engine import EngineInfer

FREQ = "/sys/class/devfreq/CIXH4000:00/cur_freq"
STOP_SIGNALS = {signal.SIGTERM, signal.SIGINT}
HERE = Path(__file__).resolve().parent


def npu_mhz():
    try:
        return int(open(FREQ).read()) // 1_000_000
    except OSError:
        return None


class Graph:
    """EngineInfer only for loading and tensor descriptors; inference through the raw libnoe calls (no float conversion, no prints)."""

    def __init__(self, path, reuse_job):
        self.e = EngineInfer(str(path))
        self.npu = self.e.npu
        self.ins, self.outs = self.e.in_tensor_desc, self.e.out_tensor_desc
        self.reuse = reuse_job
        self.job = None

    def _job(self):
        if self.job is None:
            ret, self.job = self.npu.noe_create_job(self.e.graph_id, self.e.job_cfg)
            if ret != 0:
                raise RuntimeError("noe_create_job failed")
        return self.job

    def run(self, inputs, want):
        """inputs: list of bytes; want: output indices to read. Returns (outputs dict idx->bytes, infer seconds)."""
        job = self._job()
        for i, b in enumerate(inputs):
            if self.npu.noe_load_tensor(job, i, b) != 0:
                raise RuntimeError(f"noe_load_tensor {i} failed")
        # a signal arriving inside infer_sync interrupts the driver wait ("AIPU device state is abnormal"); hold it until the job is done
        signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
        t = time.perf_counter()
        ret = self.npu.noe_job_infer_sync(job, -1)
        dt = time.perf_counter() - t
        signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
        if ret != 0:
            raise RuntimeError(f"noe_job_infer_sync returned {ret}")
        out = {}
        for j in want:
            r, data = self.npu.noe_get_tensor(job, NOE_TENSOR_TYPE_OUTPUT, j)
            if r != 0:
                raise RuntimeError(f"noe_get_tensor {j} failed")
            out[j] = data
        if not self.reuse:
            self.npu.noe_clean_job(job)
            self.job = None
        return out, dt


def emit(tokens, wall, npu_s):
    print("NPU_RESULT " + json.dumps({"tps": round(tokens / wall, 4), "tokens": tokens, "ms_per_tok": round(wall * 1000 / tokens, 3),
                                      "npu_ms": round(npu_s * 1000, 3), "host_ms": round((wall - npu_s) * 1000, 3), "npu_mhz": npu_mhz()}),
          flush=True)


def prefill_loop(g, a, n_units):
    S = g.ins[0].size // 4
    rng = np.random.default_rng(a.seed)
    want = [0, 1]                     # logits and one KV output: reading all 48 KV outputs is host work a pipeline would do, see --read-all-kv
    if a.read_all_kv:
        want = list(range(len(g.outs)))
    k = 0
    while n_units is None or k < n_units:
        ids = rng.integers(0, a.vocab, size=S, dtype=np.int32)
        t0 = time.perf_counter()
        out, dt = g.run([ids.tobytes()], want)
        int(np.argmax(np.frombuffer(out[0], dtype=np.int8)))
        emit(S, time.perf_counter() - t0, dt)
        k += 1


def decode_loop(g, a, n_units):
    n_in, n_out = len(g.ins), len(g.outs)
    n_kv = n_in - 2
    assert n_out == n_kv + 1, f"decode graph: {n_in} inputs, {n_out} outputs; expected past_* = present_*"
    row = g.outs[1].size                                  # bytes of one new row of one tensor: KVH * HD (int8)
    P = g.ins[2].size // row
    # past_i and present_i share one quantization (true for the int8 build); otherwise requantize the new row
    same = all(abs(g.ins[2 + i].scale - g.outs[1 + i].scale) < 1e-6 * max(1.0, abs(g.ins[2 + i].scale))
               and g.ins[2 + i].zero_point == g.outs[1 + i].zero_point for i in range(n_kv))
    kvh = 2 if row % 2 == 0 else 1
    hd = row // kvh
    past = np.empty((n_kv, kvh, P, hd), dtype=np.int8)
    for i in range(n_kv):                                  # real zero = -zero_point in this quantization (q = x*scale - zp)
        past[i] = np.int8(np.clip(-g.ins[2 + i].zero_point, -128, 127))
    tok = np.zeros(1, dtype=np.int32)
    pos = np.full(1, P, dtype=np.int32)
    want = list(range(n_out))
    k = 0
    while n_units is None or k < n_units:
        t0 = time.perf_counter()
        npu_s = 0.0
        for _ in range(a.decode_tokens):
            out, dt = g.run([tok.tobytes(), pos.tobytes()] + [past[i].tobytes() for i in range(n_kv)], want)
            npu_s += dt
            tok[0] = int(np.argmax(np.frombuffer(out[0], dtype=np.int8)))
            pos[0] += 1
            past[:, :, :-1, :] = past[:, :, 1:, :]
            for i in range(n_kv):
                new = np.frombuffer(out[1 + i], dtype=np.int8).reshape(kvh, hd)
                if not same:
                    so, zo, si, zi = g.outs[1 + i].scale, g.outs[1 + i].zero_point, g.ins[2 + i].scale, g.ins[2 + i].zero_point
                    new = np.clip(np.round((new.astype(np.float32) + zo) / so * si - zi), -128, 127).astype(np.int8)
                past[i, :, -1, :] = new
        emit(a.decode_tokens, time.perf_counter() - t0, npu_s)
        k += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["prefill", "decode"])
    ap.add_argument("--model", default="qwen2.5-0.5b", help="file stem: <cix-dir>/<model>_<phase>.cix")
    ap.add_argument("--cix-dir", default=str(HERE))
    ap.add_argument("--cix", help="explicit .cix path (overrides --model/--cix-dir)")
    ap.add_argument("--persistent", action="store_true", help="loop until killed")
    ap.add_argument("--iters", type=int, default=1, help="results to print when not --persistent")
    ap.add_argument("--decode-tokens", type=int, default=32, help="decode steps per NPU_RESULT")
    ap.add_argument("--no-reuse-job", action="store_true", help="create and clean an NPU job per inference (like NOE_Engine.forward)")
    ap.add_argument("--read-all-kv", action="store_true", help="prefill: read all KV outputs back to the host (default: logits + one)")
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    for s in (signal.SIGTERM, signal.SIGINT):           # libnoe aborts in interpreter teardown; leave without it
        signal.signal(s, lambda *_: (sys.stdout.flush(), os._exit(0)))
    path = a.cix or os.path.join(a.cix_dir, f"{a.model}_{a.phase}.cix")
    t = time.perf_counter()
    g = Graph(path, reuse_job=not a.no_reuse_job)
    print(f"# loaded {os.path.basename(path)} in {time.perf_counter() - t:.1f} s: {len(g.ins)} inputs, {len(g.outs)} outputs",
          file=sys.stderr, flush=True)
    try:
        (prefill_loop if a.phase == "prefill" else decode_loop)(g, a, None if a.persistent else a.iters)
    except Exception as e:     # noqa: BLE001
        print(f"# error: {e}", file=sys.stderr, flush=True)
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
