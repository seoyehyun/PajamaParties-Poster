#!/usr/bin/env python3
"""Reduced needle-in-a-haystack (NIAH) check for the KVSwap selection modes (docs/kvswap/kv-offload-plan.md 5, 7-3).

A passcode sentence (the needle) is inserted at several depths of a real-text haystack, the prompt ends with a question,
and every config generates the answer greedily. A run counts as correct when the generated text contains the full
passcode. Only the decode steps use the KV selection; the first answer token comes from the prefill (full attention)
but with Qwen3 it is the space before the number, and digits are tokenized one by one, so all 7 digits are decoded
under selection (the uniform pick got even the first digit right in only 6 of 15 cases).

Configs are "name:ENV=VAL,ENV=VAL" (empty env = full attention), e.g.
  full:
  unif_m16:LLAMA_KV_SELECT=16
  pred_r128_m16:LLAMA_KV_SELECT=16,LLAMA_KV_ADAPTER=/mnt/ssd/adapters/kvswap-adapter-r128.gguf

Runs the KV cache in RAM (no offload file): selection quality does not depend on where the KV lives, and without the
offload path flash attention is optional (-fa off prefills faster on this board). Standard library only.

  python3 -I scripts/kvswap/niah_kvswap.py --haystack /mnt/ssd/prompts/haystack_c4.txt --out runs/niah_<date> \
      --config full: --config unif_m16:LLAMA_KV_SELECT=16 --config pred_r128_m16:LLAMA_KV_SELECT=16,LLAMA_KV_ADAPTER=...
"""
import argparse
import csv
import os
import random
import re
import subprocess
import sys
import time

NAMES = ["Mira", "Tobias", "Keiko", "Anselm", "Priya", "Dorian", "Lucia", "Henrik", "Amara", "Felix"]


def build_prompt(hay, n_chars, depth, name, code):
    """haystack cut to ~n_chars on sentence boundaries, needle at fraction `depth` of it"""
    text = hay[:n_chars]
    text = text[:text.rfind(". ") + 1]
    cut = int(len(text)*depth)
    dot = text.rfind(". ", 0, cut)
    cut = dot + 2 if dot > 0 else 0
    needle = f"The secret passcode for {name} is {code}. "
    question = (f"\n\nQuestion: According to the text above, what is the secret passcode for {name}?\n"
                f"Answer: The secret passcode for {name} is")
    return text[:cut] + needle + text[cut:] + question


def parse_cfg(s):
    name, _, envs = s.partition(":")
    env = {}
    for kv in filter(None, envs.split(",")):
        k, _, v = kv.partition("=")
        env[k] = v
    return name, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.expanduser("~/llama-base/build/bin/llama-completion"))
    ap.add_argument("--model", default=os.path.expanduser("~/PajamaParties/models/Qwen3-1.7B-Q4_K_M.gguf"))
    ap.add_argument("--haystack", required=True)
    ap.add_argument("--chars", type=int, default=16000, help="haystack length in characters (~4 chars/token)")
    ap.add_argument("--ctx", type=int, default=6144)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--needles", type=int, default=3, help="different name/passcode pairs per depth")
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--cpus", default="6-9")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fa", default="off")
    ap.add_argument("--window", default="128", help="LLAMA_KV_WINDOW for every config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="keep the rows already in <out>/niah.csv and skip them")
    args = ap.parse_args()

    hay = open(args.haystack, encoding="utf-8").read()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cfgs = [parse_cfg(c) for c in args.config]

    cases = []
    for depth in args.depths:
        for k in range(args.needles):
            name = NAMES[(len(cases)) % len(NAMES)]
            code = str(rng.randint(1000000, 9999999))
            off = rng.randint(0, max(0, len(hay) - args.chars - 1))
            cases.append((depth, name, code, off))

    rows = []
    csv_path = os.path.join(args.out, "niah.csv")
    done = set()
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f)]
        for r in rows:
            r["correct"] = int(r["correct"])
            r["depth"] = float(r["depth"])
        done = {(int(r["case"]), r["config"]) for r in rows}
        print(f"resume: {len(done)} runs already in {csv_path}", flush=True)
    for ci, (depth, name, code, off) in enumerate(cases):
        prompt = build_prompt(hay[off:], args.chars, depth, name, code)
        ppath = os.path.join(args.out, f"prompt_{ci:02d}.txt")
        with open(ppath, "w", encoding="utf-8") as f:
            f.write(prompt)
        for cname, cenv in cfgs:
            if (ci, cname) in done:
                continue
            env = dict(os.environ)
            env.update({"LLAMA_KV_WINDOW": args.window})
            env.update(cenv)
            cmd = ["taskset", "-c", args.cpus, args.bin, "-m", args.model, "-f", ppath, "-c", str(args.ctx),
                   "-n", "12", "--temp", "0", "--seed", "1", "-fa", args.fa, "-t", str(args.threads),
                   "--no-warmup", "--no-display-prompt", "-no-cnv"]   # plain completion, no chat template / thinking
            t0 = time.time()
            p = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace")
            dt = time.time() - t0
            out = p.stdout.strip()
            m_tok = re.search(r"prompt eval time =.*?/\s*(\d+) tokens", p.stderr)
            m_ev = re.search(r"kvswap_pred_eval: samples=\d+ mass_base=([\d.]+) mass_pred=([\d.]+) mass_oracle=([\d.]+) "
                             r"mass_uniform=([\d.]+) group_recall=([\d.]+)", p.stderr)
            digits = re.sub(r"\D", "", out)
            row = {
                "case": ci, "depth": depth, "name": name, "code": code, "config": cname,
                "correct": int(code in digits), "first_digit_ok": int(digits[:1] == code[:1]),
                "n_prompt": m_tok.group(1) if m_tok else "", "exit": p.returncode, "wall_s": f"{dt:.1f}",
                "output": out.replace("\n", " ")[:80],
            }
            if m_ev:
                row.update(dict(zip(["mass_base", "mass_pred", "mass_oracle", "mass_uniform", "recall"], m_ev.groups())))
            rows.append(row)
            print(f"case {ci:2d} depth {depth:.1f} {cname:16s} {'OK ' if row['correct'] else 'BAD'} "
                  f"n={row['n_prompt']} {dt:5.1f}s  {row['output'][:40]!r}", flush=True)
            keys = ["case", "depth", "name", "code", "config", "correct", "first_digit_ok", "n_prompt", "exit", "wall_s",
                    "mass_base", "mass_pred", "mass_oracle", "mass_uniform", "recall", "output"]
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

    print("\naccuracy by config:")
    for cname, _ in cfgs:
        rs = [r for r in rows if r["config"] == cname]
        print(f"  {cname:16s} {sum(r['correct'] for r in rs)}/{len(rs)}")
        for depth in args.depths:
            d = [r for r in rs if r["depth"] == depth]
            print(f"      depth {depth:.1f}: {sum(r['correct'] for r in d)}/{len(d)}")


if __name__ == "__main__":
    sys.exit(main())
