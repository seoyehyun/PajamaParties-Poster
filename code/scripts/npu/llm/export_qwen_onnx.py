#!/usr/bin/env python3
"""Export fixed-shape prefill and decode ONNX graphs of a Qwen-style causal LM for cixbuild (CIX P1, Zhouyi X2 NPU).

Route A of docs/npu/npu-llm-plan.md. Run on an x86 PC with torch, transformers and onnx. NOT yet run against cixbuild.

The NPU compiler takes static shapes only, so the model is split like the community report that got a tiny Qwen
running on this NPU (see the plan): a prefill graph with a fixed prompt length S and a decode graph with ONE new
token and a fixed past length P. The KV cache is not hidden state, it is an input and an output of the decode graph.

  prefill.onnx   in : input_ids [1,S]
                 out: logits [1,V] (last position only), present_key_i / present_value_i [1,KVH,S,HD] per layer
  decode.onnx    in : input_ids [1,1], position_ids [1,1], (attention_mask [1,P+1] with --mask-input),
                      past_key_i / past_value_i [1,KVH,P,HD] per layer
                 out: logits [1,V], present_key_i / present_value_i [1,KVH,1,HD] (the new row only; --full-present: P+1 rows)

--host-embed keeps the embedding table and the output head OUT of the NPU graphs (the NPU address window is about 1 GiB
per graph and the tied 151,936 x 2,048 matrix alone is 0.3 GB at int8). Then:
  prefill.onnx   in : inputs_embeds [1,S,H]            out: hidden [1,H] (after the final norm), present_*
  decode.onnx    in : inputs_embeds [1,1,H], position_ids, past_*   out: hidden [1,H], present_*
and the board script does   inputs_embeds = embed_tokens[token]   and   logits = hidden @ lm_head.T   on the CPU
(host/embed_tokens.npy, host/lm_head.npy only if the head is not tied to the embedding, both float16).

Also writes meta.json (shapes and names for the cfg and the board scripts) and calibration arrays for cixbuild.
For latency and frequency measurements only the shapes matter, not the weights: --tiny builds a random 2-layer
model of the same family so the whole pipeline can be tried without a download.

  python export_qwen_onnx.py --tiny --seq-len 32 --out out_tiny --verify
  python export_qwen_onnx.py --model Qwen/Qwen2.5-0.5B-Instruct --seq-len 256 --out out_q05
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- KV cache helpers (transformers API moves a lot)
def make_cache(pairs):
    from transformers import DynamicCache
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(pairs))
    cache = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        cache.update(k, v, i)
    return cache


def cache_pairs(cache):
    if hasattr(cache, "to_legacy_cache"):
        return [tuple(x) for x in cache.to_legacy_cache()]
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


# ---------------------------------------------------------------- model
def load_model(args):
    if args.tiny:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=1024, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                          tie_word_embeddings=True)
        cfg._attn_implementation = "eager"
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(cfg)
        tok = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="eager")
        try:
            tok = AutoTokenizer.from_pretrained(args.model)
        except Exception:
            tok = None
    if args.fp16:
        model = model.half()
    return model.eval(), tok


def geometry(model):
    c = model.config
    return dict(
        layers=c.num_hidden_layers,
        kv_heads=c.num_key_value_heads,
        head_dim=getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads,
        vocab=c.vocab_size,
    )


class Prefill(torch.nn.Module):
    def __init__(self, model, seq_len, host_embed=False):
        super().__init__()
        self.m, self.s, self.he = model, seq_len, host_embed

    def forward(self, x):
        pos = torch.arange(self.s, device=x.device).unsqueeze(0)
        if self.he:      # x = inputs_embeds, the head runs on the host
            out = self.m.model(inputs_embeds=x, position_ids=pos, use_cache=True, return_dict=True)
            logits = out.last_hidden_state[:, -1, :]
        else:
            out = self.m.model(input_ids=x, position_ids=pos, use_cache=True, return_dict=True)
            logits = self.m.lm_head(out.last_hidden_state[:, -1, :])
        flat = []
        for k, v in cache_pairs(out.past_key_values):
            flat += [k.contiguous(), v.contiguous()]
        return (logits, *flat)


class Decode(torch.nn.Module):
    def __init__(self, model, past_len, mask_input, full_present, host_embed=False):
        super().__init__()
        self.m, self.p, self.mask_input, self.full, self.he = model, past_len, mask_input, full_present, host_embed

    def forward(self, x, position_ids, *rest):
        if self.mask_input:
            mask, past = rest[0].to(torch.bool), rest[1:]
        else:
            mask, past = torch.ones((1, self.p + 1), dtype=torch.bool), rest
        pairs = [(past[i], past[i + 1]) for i in range(0, len(past), 2)]
        kw = dict(inputs_embeds=x) if self.he else dict(input_ids=x)
        out = self.m.model(position_ids=position_ids, attention_mask=mask,
                           past_key_values=make_cache(pairs), use_cache=True, return_dict=True, **kw)
        logits = out.last_hidden_state[:, -1, :] if self.he else self.m.lm_head(out.last_hidden_state[:, -1, :])
        flat = []
        for k, v in cache_pairs(out.past_key_values):
            if not self.full:
                k, v = k[:, :, -1:, :], v[:, :, -1:, :]
            flat += [k.contiguous(), v.contiguous()]
        return (logits, *flat)


def export(wrapper, args_t, path, in_names, out_names):
    kw = dict(input_names=in_names, output_names=out_names, opset_version=17, do_constant_folding=True)
    try:
        torch.onnx.export(wrapper, args_t, str(path), dynamo=False, **kw)
    except TypeError:                       # older torch has no dynamo switch
        torch.onnx.export(wrapper, args_t, str(path), **kw)


# ---------------------------------------------------------------- host-side embedding / head, graph size
def embed_table(model):
    return model.model.embed_tokens.weight.detach().float().numpy()


def head_matrix(model):
    return model.lm_head.weight.detach().float().numpy()


def head_is_tied(model):
    return model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def graph_weight_report(path):
    """parameters held by the graph (initializers), as a rough int8 size: this is what must fit the ~1 GiB NPU window"""
    try:
        import onnx
    except ImportError:
        print(f"  {path.name}: pip install onnx to see the weight size")
        return None
    m = onnx.load(str(path), load_external_data=False)
    n = sum(int(np.prod(i.dims)) for i in m.graph.initializer)
    big = sorted(((int(np.prod(i.dims)), i.name, list(i.dims)) for i in m.graph.initializer), reverse=True)[:3]
    print(f"  {path.name}: {n/1e6:.0f}M parameters = {n/2**30:.2f} GiB at int8 (largest: "
          + ", ".join(f"{nm.split('/')[-1] or nm} {d}" for _, nm, d in big) + ")")
    return n


# ---------------------------------------------------------------- calibration data
def token_batches(args, tok, vocab, n, length):
    rng = np.random.default_rng(0)
    if args.calib_text and tok is not None:
        ids = tok(Path(args.calib_text).read_text(encoding="utf-8", errors="ignore"), return_tensors="np")["input_ids"][0]
        if len(ids) >= length:
            starts = rng.integers(0, len(ids) - length + 1, size=n)
            return np.stack([ids[s:s + length] for s in starts]).astype(np.int64)
        print("calibration text shorter than the sequence, using random tokens", file=sys.stderr)
    return rng.integers(0, vocab, size=(n, length)).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tiny", action="store_true", help="random 2-layer Qwen2 model, no download (pipeline test)")
    ap.add_argument("--seq-len", type=int, default=256, help="prefill prompt length S (fixed)")
    ap.add_argument("--past-len", type=int, default=None, help="decode past length P (default S-1)")
    ap.add_argument("--out", default="out_npu_llm")
    ap.add_argument("--fp16", action="store_true", help="export in half precision (default float32; cixbuild quantizes anyway)")
    ap.add_argument("--mask-input", action="store_true", help="decode graph takes attention_mask [1,P+1] as an input")
    ap.add_argument("--full-present", action="store_true", help="decode returns the whole KV (P+1 rows) instead of the new row")
    ap.add_argument("--calib-samples", type=int, default=8)
    ap.add_argument("--calib-text", default=None, help="plain text file for realistic calibration tokens")
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--host-embed", action="store_true",
                    help="embedding lookup and output head run on the host: graphs take inputs_embeds and return hidden")
    ap.add_argument("--verify", action="store_true", help="check the ONNX graphs with onnxruntime against PyTorch")
    args = ap.parse_args()

    S = args.seq_len
    P = args.past_len if args.past_len is not None else S - 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(args)
    g = geometry(model)
    L, KVH, HD, V = g["layers"], g["kv_heads"], g["head_dim"], g["vocab"]
    dt = torch.float16 if args.fp16 else torch.float32
    print(f"layers={L} kv_heads={KVH} head_dim={HD} vocab={V}  S={S} P={P}", flush=True)

    kv_names = lambda prefix: [f"{prefix}_{kind}_{i}" for i in range(L) for kind in ("key", "value")]
    HE = args.host_embed
    H = model.config.hidden_size
    tok_in, head_out = ("inputs_embeds", "hidden") if HE else ("input_ids", "logits")
    pre_in, pre_out = [tok_in], [head_out] + kv_names("present")
    dec_in = [tok_in, "position_ids"] + (["attention_mask"] if args.mask_input else []) + kv_names("past")
    dec_out = [head_out] + kv_names("present")

    with torch.no_grad():
        ids = torch.zeros((1, S, H), dtype=dt) if HE else torch.zeros((1, S), dtype=torch.long)
        export(Prefill(model, S, HE), (ids,), out / "prefill.onnx", pre_in, pre_out)
        print("wrote prefill.onnx", flush=True)

        d_ids = torch.zeros((1, 1, H), dtype=dt) if HE else torch.zeros((1, 1), dtype=torch.long)
        d_pos = torch.full((1, 1), P, dtype=torch.long)
        d_mask = [torch.ones((1, P + 1), dtype=torch.long)] if args.mask_input else []
        d_past = [torch.zeros((1, KVH, P, HD), dtype=dt) for _ in range(2 * L)]
        export(Decode(model, P, args.mask_input, args.full_present, HE), (d_ids, d_pos, *d_mask, *d_past),
               out / "decode.onnx", dec_in, dec_out)
        print("wrote decode.onnx", flush=True)
    print("weights inside the NPU graphs:")
    graph_weight_report(out / "prefill.onnx")
    graph_weight_report(out / "decode.onnx")
    if HE:
        (out / "host").mkdir(exist_ok=True)
        np.save(out / "host" / "embed_tokens.npy", embed_table(model).astype(np.float16))
        if not head_is_tied(model):
            np.save(out / "host" / "lm_head.npy", head_matrix(model).astype(np.float16))
        print(f"wrote host/ (embedding {V}x{H} float16; head {'tied to the embedding' if head_is_tied(model) else 'separate lm_head.npy'})")

    rows = P + 1 if args.full_present else 1
    meta = dict(
        model="tiny-random-qwen2" if args.tiny else args.model, seq_len=S, past_len=P, dtype=str(dt).split(".")[-1],
        layers=L, kv_heads=KVH, head_dim=HD, vocab=V, hidden=H, mask_input=args.mask_input, full_present=args.full_present,
        host_embed=HE, head_tied=bool(head_is_tied(model)),
        prefill=dict(file="prefill.onnx", inputs={tok_in: [1, S, H] if HE else [1, S]},
                     outputs={head_out: [1, H] if HE else [1, V], **{n: [1, KVH, S, HD] for n in pre_out[1:]}}),
        decode=dict(file="decode.onnx",
                    inputs={tok_in: [1, 1, H] if HE else [1, 1], "position_ids": [1, 1],
                            **({"attention_mask": [1, P + 1]} if args.mask_input else {}),
                            **{n: [1, KVH, P, HD] for n in dec_in if n.startswith("past_")}},
                    outputs={head_out: [1, H] if HE else [1, V], **{n: [1, KVH, rows, HD] for n in dec_out[1:]}}),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    if not args.no_calib:
        n = args.calib_samples
        toks = token_batches(args, tok, V, n, S)
        cdir_p, cdir_d = out / "calib_prefill", out / "calib_decode"
        cdir_p.mkdir(exist_ok=True)
        cdir_d.mkdir(exist_ok=True)
        emb = embed_table(model) if HE else None
        np.save(cdir_p / f"{tok_in}.npy", emb[toks] if HE else toks)   # (N, S[, H])
        past_k = [[] for _ in range(2 * L)]
        with torch.no_grad():
            for t in toks:
                o = model.model(input_ids=torch.from_numpy(t)[None], position_ids=torch.arange(S)[None], use_cache=True)
                for j, (k, v) in enumerate(cache_pairs(o.past_key_values)):
                    past_k[2 * j].append(k[0, :, :P, :].float().numpy())
                    past_k[2 * j + 1].append(v[0, :, :P, :].float().numpy())
        nxt = toks[:, P:P + 1]                                      # (N, 1): the token after the P cached ones
        np.save(cdir_d / f"{tok_in}.npy", emb[nxt] if HE else nxt)
        np.save(cdir_d / "position_ids.npy", np.full((n, 1), P, dtype=np.int64))
        if args.mask_input:
            np.save(cdir_d / "attention_mask.npy", np.ones((n, P + 1), dtype=np.int64))
        for name, arrs in zip([x for x in dec_in if x.startswith("past_")], past_k):
            np.save(cdir_d / f"{name}.npy", np.stack(arrs))        # (N, KVH, P, HD)
        print(f"wrote calibration arrays for {n} samples (array layout still to be checked against the SDK manual)")

    if args.verify:
        verify(model, out, S, P, g, args)


def verify(model, out, S, P, g, args):
    """prefill logits == full forward; decode logits == prefill logits when decode gets the first S-1 cached rows"""
    import onnxruntime as ort
    if P != S - 1:
        print("verify needs --past-len = seq-len - 1, skipped")
        return
    rng = np.random.default_rng(1)
    toks = rng.integers(0, g["vocab"], size=(1, S)).astype(np.int64)
    with torch.no_grad():
        ref = model(input_ids=torch.from_numpy(toks)).logits[:, -1, :].float().numpy()
    he = args.host_embed
    E, Wh = (embed_table(model), head_matrix(model)) if he else (None, None)
    tok_in = "inputs_embeds" if he else "input_ids"
    dtype = np.float16 if args.fp16 else np.float32
    x_of = (lambda t: E[t].astype(dtype)) if he else (lambda t: t)           # host lookup
    logits_of = (lambda h: h @ Wh.T) if he else (lambda l: l)                # host head
    sp = ort.InferenceSession(str(out / "prefill.onnx"), providers=["CPUExecutionProvider"])
    po = sp.run(None, {tok_in: x_of(toks)})
    pl = logits_of(po[0].astype(np.float32))
    e1 = float(np.abs(pl - ref).max())
    sd = ort.InferenceSession(str(out / "decode.onnx"), providers=["CPUExecutionProvider"])
    feed = {tok_in: x_of(toks[:, S - 1:S]), "position_ids": np.array([[S - 1]], dtype=np.int64)}
    if args.mask_input:
        feed["attention_mask"] = np.ones((1, S), dtype=np.int64)
    names = [i.name for i in sd.get_inputs() if i.name.startswith("past_")]
    for name, arr in zip(names, po[1:]):
        feed[name] = arr[:, :, :P, :].astype(np.float32 if not args.fp16 else np.float16)
    do = sd.run(None, feed)
    e2 = float(np.abs(logits_of(do[0].astype(np.float32)) - pl).max())
    new_k = do[1] if not args.full_present else do[1][:, :, -1:, :]
    e3 = float(np.abs(new_k - po[1][:, :, S - 1:S, :]).max())
    print(f"verify: prefill-vs-torch {e1:.2e}   decode-vs-prefill logits {e2:.2e}   decode new K row vs prefill {e3:.2e}")
    ok = max(e1, e2, e3) < (5e-2 if args.fp16 else 1e-3)
    print("verify:", "OK" if ok else "MISMATCH")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
