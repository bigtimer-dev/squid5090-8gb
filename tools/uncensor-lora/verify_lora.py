# Check that the AtomicChat rank-1 LoRA equals exact projection removal on the official weights:
#   attn:    A_attn  == -W_o^T b_l           experts: A_e == -W2_e^T b_l        embed: a[t] == -b_e . E[t]
# usage: uv run --no-project --with numpy --with gguf python -I verify_lora.py SRC/model-00001-of-00001.safetensors LORA.gguf
#   (GGUF_PY=/path/to/llama.cpp/gguf-py uses llama.cpp's copy of gguf; see make_uncensored.py)
import os, sys
if os.environ.get("GGUF_PY"): sys.path.insert(0, os.environ["GGUF_PY"])  # e.g. llama.cpp/gguf-py
import numpy as np
from gguf import GGUFReader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_header import header

st_path, lora_path = sys.argv[1], sys.argv[2]
H, base = header(st_path)
mm = np.memmap(st_path, dtype=np.uint8, mode='r')
def raw(name):
    a, b = H[name]['data_offsets']
    return mm[base + a: base + b]
def bf16(name):
    u = raw(name).view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(H[name]['shape'])

def e4m3_lut():
    lut = np.zeros(256, np.float32)
    for v in range(256):
        s = -1.0 if v & 0x80 else 1.0
        e = (v >> 3) & 0xF; m = v & 7
        if e == 15 and m == 7: lut[v] = np.nan
        elif e == 0: lut[v] = s * (m / 8.0) * 2.0 ** -6
        else: lut[v] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return lut
E4M3 = e4m3_lut()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)

def w2_expert(l, e):
    pre = f"model.llm.layers.{l}.mlp.experts.w2_weight"
    if H[pre]['dtype'] == 'U8':  # NVFP4
        q = raw(pre).reshape(128, 2304, 576)[e]
        sc = raw(pre + "_scale").reshape(128, 2304, 72)[e]
        amax = raw(pre + "_amax").view(np.float32)[e]
        den = raw(pre + "_outer_scale_denominator").view(np.float32)[0]
        lo = E2M1[q & 0xF]; hi = E2M1[q >> 4]
        w = np.stack([lo, hi], -1).reshape(2304, 1152)
        return w * np.repeat(E4M3[sc], 16, axis=1) * (amax / den)
    q = raw(pre).reshape(128, 2304, 1152)[e]                   # MXFP8
    sc = raw(pre + "_scale").reshape(128, 2304, 36)[e].astype(np.int32)
    return E4M3[q] * np.repeat(np.exp2(sc - 127).astype(np.float32), 32, axis=1)

r = GGUFReader(lora_path)
T = {t.name: np.asarray(t.data) for t in r.tensors}
def stats(pred, got):
    c = float(pred @ got / (np.linalg.norm(pred) * np.linalg.norm(got)))
    k = float(pred @ got / (pred @ pred))
    return f"cos {c:.6f} scale {k:.4f}"

for l in (0, 3, 6, 11, 17, 19, 20, 23):
    b = T[f"blk.{l}.attn_output.weight.lora_b"].reshape(2304).astype(np.float64)
    Ao = T[f"blk.{l}.attn_output.weight.lora_a"].reshape(3072).astype(np.float64)
    Wo = bf16(f"model.llm.layers.{l}.attn.wo_ud.weight").astype(np.float64)
    line = f"L{l:2d} attn: {stats(-(b @ Wo), Ao)} |"
    A = T[f"blk.{l}.ffn_down_exps.weight.lora_a"].reshape(128, 1152).astype(np.float64)
    worst = 1.0; ks = []
    for e in (0, 37, 77, 127):
        pred = -(b @ w2_expert(l, e).astype(np.float64))
        c = float(pred @ A[e] / (np.linalg.norm(pred) * np.linalg.norm(A[e]))); worst = min(worst, c)
        ks.append(float(pred @ A[e] / (pred @ pred)))
    print(line, f"experts(4): min cos {worst:.6f} scale {min(ks):.4f}..{max(ks):.4f}", flush=True)

a = T["token_embd.weight.lora_a"].reshape(-1).astype(np.float64)
be = T["token_embd.weight.lora_b"].reshape(-1).astype(np.float64)
E = bf16("model.llm.embed.weight")
idx = np.random.default_rng(0).choice(201024, 4096, replace=False); idx.sort()
pred = -(E[idx].astype(np.float64) @ be)
print("embed (4096 rows):", stats(pred, a[idx]), " max|a-pred|", float(np.abs(a[idx] - pred).max()), " max|a|", float(np.abs(a).max()))
