# Build the squid5090 uncensored model dir from the official checkpoint + AtomicChat rank-1 LoRA:
#  * embed rows patched in place (on a reflink copy): E'[t] = E[t] + a[t] * b_embed   (BF16, round-nearest-even)
#  * uncensor.safetensors: layer_dirs F32 [24, 2304] = per-layer unit direction b_l (runtime projection,
#    applied before the short conv by `squid5090 ... --uncensor DST/uncensor.safetensors`)
# The LoRA is exact projection removal (verify_lora.py checks it), so nothing else needs patching.
#
# Sources: SRC = hf download thinkingmachines/Inkling-2-Mini (safetensors)
#          LORA = huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF -> Inkling-2-Mini-Uncensored-LoRA.gguf
# Prepare DST first (a reflink copy only stores the ~0.9 GB of changed embedding blocks):
#   mkdir DST && cp --reflink=always SRC/model-00001-of-00001.safetensors DST/
#   for f in SRC/*; do [ -e "DST/$(basename "$f")" ] || ln -s "$f" DST/; done
# usage: uv run --no-project --with numpy --with gguf python -I make_uncensored.py SRC DST LORA.gguf
#   (GGUF_PY=/path/to/llama.cpp/gguf-py uses llama.cpp's copy of gguf instead of the PyPI package)
import json, os, struct, sys
if os.environ.get("GGUF_PY"): sys.path.insert(0, os.environ["GGUF_PY"])  # e.g. llama.cpp/gguf-py
import numpy as np
from gguf import GGUFReader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_header import header

src_dir, dst_dir, lora_path = sys.argv[1], sys.argv[2], sys.argv[3]
st = os.path.join(dst_dir, "model-00001-of-00001.safetensors")
assert os.path.exists(st) and os.path.samefile(st, st)
H, base = header(st)
info = H["model.llm.embed.weight"]
assert info["dtype"] == "BF16" and info["shape"] == [201024, 2304]
off0 = base + info["data_offsets"][0]

r = GGUFReader(lora_path)
T = {t.name: np.asarray(t.data) for t in r.tensors}
a = T["token_embd.weight.lora_a"].reshape(-1).astype(np.float32)
be = T["token_embd.weight.lora_b"].reshape(-1).astype(np.float32)
assert a.shape == (201024,) and be.shape == (2304,)

rows, cols = info["shape"]
chunk = 8192
changed = 0
with open(st, "r+b") as f:
    for r0 in range(0, rows, chunk):
        n = min(chunk, rows - r0)
        f.seek(off0 + r0 * cols * 2)
        u16 = np.frombuffer(f.read(n * cols * 2), dtype=np.uint16).reshape(n, cols)
        x = (u16.astype(np.uint32) << 16).view(np.float32)
        y = x + a[r0:r0 + n, None] * be[None, :]
        u = y.view(np.uint32)
        out = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        changed += int((out != u16).sum())
        f.seek(off0 + r0 * cols * 2)
        f.write(out.tobytes())
print(f"embed patched: {changed} of {rows*cols} bf16 values changed")

dirs = np.stack([T[f"blk.{l}.attn_output.weight.lora_b"].reshape(2304) for l in range(24)]).astype(np.float32)
for l in range(24):
    eb = T[f"blk.{l}.ffn_down_exps.weight.lora_b"].reshape(128, 2304)
    assert np.abs(eb - dirs[l]).max() == 0.0, l
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
hdr = json.dumps({"__metadata__": {"source": "AtomicChat Inkling-2-Mini-Uncensored-LoRA (rank-1, alpha 1)", "apply": "y -= b (b.y) on o_proj and expert down outputs, before short conv"},
                  "layer_dirs": {"dtype": "F32", "shape": [24, 2304], "data_offsets": [0, dirs.nbytes]}}).encode()
hdr += b" " * ((8 - len(hdr) % 8) % 8)
with open(os.path.join(dst_dir, "uncensor.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hdr))); f.write(hdr); f.write(dirs.tobytes())
print("wrote uncensor.safetensors", dirs.shape, "row norms", float(np.linalg.norm(dirs, axis=1).min()), float(np.linalg.norm(dirs, axis=1).max()))

# read-back check on a few rows against the original
H0, base0 = header(os.path.join(src_dir, "model-00001-of-00001.safetensors"))
assert H0["model.llm.embed.weight"]["data_offsets"] == info["data_offsets"]
m_new = np.memmap(st, dtype=np.uint16, mode="r", offset=off0, shape=(rows, cols))
m_old = np.memmap(os.path.join(src_dir, "model-00001-of-00001.safetensors"), dtype=np.uint16, mode="r", offset=base0 + info["data_offsets"][0], shape=(rows, cols))
idx = np.array([0, 1, 1000, 77777, 200000, 201023])
f32 = lambda m: (m.astype(np.uint32) << 16).view(np.float32)
new, old = f32(m_new[idx]), f32(m_old[idx])
print("b.E  before:", np.round(old @ be, 4), "\nb.E  after: ", np.round(new @ be, 4))
