#!/usr/bin/env python3
"""Build the KVSwap low-rank key adapters A (docs/kvswap/kv-offload-plan.md 5-1). Run on a laptop, NOT on the board.

For every layer it collects the post-RoPE K cache of a few C4 samples (what llama.cpp stores in its K cache), builds the
covariance K^T K and keeps the top-r eigenvectors, which are the right singular vectors of K. The adapter maps a key row
k (1024 values, 8 KV heads x head_dim 128) to r values: k_lr = A^T k. One GGUF file per compression ratio sigma = 1024/r.

  pip install torch transformers gguf datasets numpy
  python3 scripts/kvswap/make_kvswap_adapter.py --model Qwen/Qwen3-1.7B --out adapters/

GGUF layout: tensor "kvswap.A.<il>", ggml ne = [n_embd_k_gqa, r] (numpy shape (r, n_embd_k_gqa), row j = j-th principal
direction), so ggml_mul_mat(A, k) with k ne [n_embd_k_gqa, n_tokens] gives [r, n_tokens].
"""
import argparse
import os
import sys

import numpy as np


def get_keys(past, il):
    """[1, n_kv_heads, T, head_dim] tensor of layer il, for the different transformers cache APIs"""
    if hasattr(past, "layers"):
        return past.layers[il].keys
    if hasattr(past, "key_cache"):
        return past.key_cache[il]
    return past[il][0]


def text_samples(args, tok):
    """yield token id tensors of exactly seqlen tokens"""
    import torch
    if args.text_files:
        stream = (open(p, encoding="utf-8", errors="ignore").read() for p in args.text_files)
    else:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        stream = (ex["text"] for ex in ds)
    buf = []
    for text in stream:
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        while len(buf) >= args.seqlen:
            yield torch.tensor([buf[:args.seqlen]])
            buf = buf[args.seqlen:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--ranks", type=int, nargs="+", default=[128, 64, 32], help="r values; sigma = n_embd_k_gqa / r")
    ap.add_argument("--text-files", nargs="*", help="use these files instead of streaming C4")
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import gguf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    n_layer = model.config.num_hidden_layers
    hd = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads   # Qwen2 configs have no head_dim
    d = model.config.num_key_value_heads * hd
    print(f"{args.model}: {n_layer} layers, n_embd_k_gqa = {d}", file=sys.stderr)

    cov = [np.zeros((d, d), dtype=np.float64) for _ in range(n_layer)]
    n_rows = 0
    with torch.no_grad():
        for n, ids in enumerate(text_samples(args, tok)):
            if n >= args.samples:
                break
            out = model(ids.to(args.device), use_cache=True)
            for il in range(n_layer):
                k = get_keys(out.past_key_values, il)               # [1, H, T, hd]
                k = k[0].permute(1, 0, 2).reshape(k.shape[2], d)    # cell row = head-major, like the llama.cpp K cache
                k = k.double().cpu().numpy()
                cov[il] += k.T @ k
            n_rows += ids.shape[1]
            print(f"sample {n + 1}/{args.samples}", file=sys.stderr)
    if n_rows == 0:
        sys.exit("no samples were read")

    os.makedirs(args.out, exist_ok=True)
    # eigh returns ascending eigenvalues; flip to descending
    eig = []
    for il in range(n_layer):
        w, v = np.linalg.eigh(cov[il])
        eig.append((w[::-1].clip(min=0), v[:, ::-1]))

    for r in args.ranks:
        path = os.path.join(args.out, f"kvswap-adapter-r{r}.gguf")
        wr = gguf.GGUFWriter(path, "kvswap-adapter")
        wr.add_uint32("kvswap.rank", r)
        wr.add_uint32("kvswap.n_layer", n_layer)
        wr.add_uint32("kvswap.n_embd_k_gqa", d)
        wr.add_string("kvswap.source_model", args.model)
        energy = []
        for il, (w, v) in enumerate(eig):
            A = np.ascontiguousarray(v[:, :r].T.astype(np.float32))   # (r, d)
            wr.add_tensor(f"kvswap.A.{il}", A)
            energy.append(w[:r].sum() / max(w.sum(), 1e-30))
        wr.write_header_to_file()
        wr.write_kv_data_to_file()
        wr.write_tensors_to_file()
        wr.close()
        print(f"{path}: sigma={d // r}, energy kept per layer min/mean = {min(energy):.3f}/{np.mean(energy):.3f}")


if __name__ == "__main__":
    main()
