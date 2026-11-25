#!/usr/bin/env python3
"""Float64 numpy reference of the Inkling 2 vision encoder, following llama.cpp's mtmd inkling2v graph.

usage: vit_ref.py MODEL.safetensors IMAGE OUT_PREFIX

The image is used at its own size (padded bottom-right to a multiple of 40), as the reference processor does, so
test images must fit squid's token cap. Writes:
  OUT_PREFIX.json   dims {w, h, gw, gh, n, tokens}
  OUT_PREFIX.lnpre  float32 [n, 1024]   embeddings after ln_pre (row 0 = CLS)
  OUT_PREFIX.l0     float32 [n, 1024]   residual stream after layer 0
  OUT_PREFIX.enc    float32 [n, 1024]   encoder output (after layer 22)
  OUT_PREFIX.emb    float32 [T, 2304]   projector output = the text-side image embeddings
"""
import json
import math
import struct
import sys

import numpy as np
from PIL import Image

D, HEADS, HD, LAYERS, PATCH, GRID = 1024, 16, 64, 23, 20, 32


def open_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")

    def get(name):
        info = hdr[name]
        a, b = info["data_offsets"]
        assert info["dtype"] == "BF16", (name, info["dtype"])
        raw = np.frombuffer(mm[base + a : base + b], dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"]).astype(np.float64)

    return get


def layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


_ERF = np.frompyfunc(math.erf, 1, 1)


def gelu_erf(x):
    return 0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0)).astype(np.float64))


def interp_axis(n_in, n_out):
    # ggml bilinear, no align-corners: x = (i + 0.5) / sf - 0.5, clamped corners and weight
    sf = n_out / n_in
    out = []
    for i in range(n_out):
        x = (i + 0.5) / sf - 0.5
        x0 = math.floor(x)
        d = min(max(x - x0, 0.0), 1.0)
        out.append((min(max(x0, 0), n_in - 1), min(max(x0 + 1, 0), n_in - 1), d))
    return out


def resize_pos(pos, gw, gh):
    grid = pos.reshape(GRID, GRID, D)  # [h][w]
    if gw == GRID and gh == GRID:
        return grid.reshape(-1, D)
    ys, xs = interp_axis(GRID, gh), interp_axis(GRID, gw)
    out = np.empty((gh, gw, D))
    for oy, (y0, y1, dy) in enumerate(ys):
        for ox, (x0, x1, dx) in enumerate(xs):
            out[oy, ox] = (
                grid[y0, x0] * (1 - dx) * (1 - dy)
                + grid[y0, x1] * dx * (1 - dy)
                + grid[y1, x0] * (1 - dx) * dy
                + grid[y1, x1] * dx * dy
            )
    return out.reshape(-1, D)


def rope2d(t, pw, ph):
    # t [n, heads, 64]: dims [0,32) rotate with pos_w, [32,64) with pos_h; adjacent pairs (ggml mode 0), 16 freqs
    inv = 10000.0 ** (-np.arange(16) * 2.0 / 32.0)
    out = t.copy()
    for half, pos in ((0, pw), (32, ph)):
        ang = pos[:, None] * inv[None, :]
        c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        x0, x1 = t[:, :, half : half + 32 : 2], t[:, :, half + 1 : half + 32 : 2]
        out[:, :, half : half + 32 : 2] = x0 * c - x1 * s
        out[:, :, half + 1 : half + 32 : 2] = x0 * s + x1 * c
    return out


def main():
    st_path, img_path, prefix = sys.argv[1:4]
    get = open_st(st_path)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    W, H = -(-w // 40) * 40, -(-h // 40) * 40
    gw, gh = W // PATCH, H // PATCH
    full = np.full((H, W, 3), -1.0)
    full[:h, :w] = (np.asarray(img, dtype=np.float64) - 127.5) / 127.5
    patches = full.reshape(gh, PATCH, gw, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, PATCH * PATCH * 3)

    e = "model.visual.backbone.embeddings."
    x = patches @ get(e + "patch_embedding.weight").T + resize_pos(get(e + "position_embedding"), gw, gh)
    cls = get(e + "class_embedding") + get(e + "cls_position_embedding")
    x = np.concatenate([cls[None, :], x], 0)
    n = x.shape[0]
    x = layernorm(x, get("model.visual.backbone.ln_pre.weight"), get("model.visual.backbone.ln_pre.bias"))
    x.astype(np.float32).tofile(prefix + ".lnpre")

    pw = np.array([0] + [xx + 1 for yy in range(gh) for xx in range(gw)], dtype=np.float64)
    ph = np.array([0] + [yy + 1 for yy in range(gh) for xx in range(gw)], dtype=np.float64)
    for l in range(LAYERS):
        p = f"model.visual.backbone.layers.{l}."
        a = layernorm(x, get(p + "ln_1.weight"), get(p + "ln_1.bias"))
        q = a @ get(p + "self_attn.q_proj.weight").T + get(p + "self_attn.q_proj.bias")
        k = a @ get(p + "self_attn.k_proj.weight").T + get(p + "self_attn.k_proj.bias")
        v = a @ get(p + "self_attn.v_proj.weight").T + get(p + "self_attn.v_proj.bias")
        q = rope2d(q.reshape(n, HEADS, HD), pw, ph).transpose(1, 0, 2)
        k = rope2d(k.reshape(n, HEADS, HD), pw, ph).transpose(1, 0, 2)
        v = v.reshape(n, HEADS, HD).transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1) / math.sqrt(HD)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(n, D)
        o = o @ get(p + "self_attn.out_proj.weight").T + get(p + "self_attn.out_proj.bias")
        x = x + get(p + "ls_1.gamma") * o
        a = layernorm(x, get(p + "ln_2.weight"), get(p + "ln_2.bias"))
        f = gelu_erf(a @ get(p + "mlp.c_fc.weight").T + get(p + "mlp.c_fc.bias"))
        f = f @ get(p + "mlp.c_proj.weight").T + get(p + "mlp.c_proj.bias")
        x = x + get(p + "ls_2.gamma") * f
        if l == 0:
            x.astype(np.float32).tofile(prefix + ".l0")
        print(f"layer {l} done, |x|max={np.abs(x).max():.1f}", file=sys.stderr)
    x.astype(np.float32).tofile(prefix + ".enc")

    xp = x[1:].reshape(gh, gw, D)
    m = np.concatenate([xp[0::2, 0::2], xp[0::2, 1::2], xp[1::2, 0::2], xp[1::2, 1::2]], -1).reshape(-1, 4 * D)
    y = gelu_erf(m @ get("model.visual.proj_in.weight").T) @ get("model.visual.proj_out.weight").T
    y.astype(np.float32).tofile(prefix + ".emb")
    with open(prefix + ".json", "w") as f:
        json.dump({"w": w, "h": h, "gw": gw, "gh": gh, "n": n, "tokens": y.shape[0]}, f)
    print(json.dumps({"w": w, "h": h, "gw": gw, "gh": gh, "n": n, "tokens": y.shape[0], "emb_absmax": float(np.abs(y).max())}))


if __name__ == "__main__":
    main()
