# squid5090-8gb

**A fork of [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090) (based on `3ee093c`) that runs Inkling-2-Mini on an 8 GB GPU.** Tested on an RTX 4060 Laptop GPU (`sm_89`, 8 GB VRAM, PCIe 4.0 x8) with 30 GB of RAM. Upstream's README follows below, unchanged.

## What this fork changes

- **Generation is ~4× faster when the CPU computes the experts that live in RAM** (opt-in, see [Fast generation](#fast-generation-cpu--gpu-experts)): 54–55 tok/s instead of 12.5 on a short prompt at 64K context. While one sequence generates, the CPU computes the routed experts held in host RAM straight from RAM and the GPU computes the rest. Up to ~690 of the most-used host experts are also copied into VRAM that sits idle during generation: the prefill buffers, the unused end of the KV cache, and the slots of the 300 least-used VRAM-layer experts, which the CPU computes instead.
- **Image input.** Chat requests can carry images in user and tool messages: OpenAI-style `image_url` parts with base64 `data:` URLs, in PNG, JPEG or WebP. Inkling-2-Mini's vision encoder runs on the GPU inside the prefill buffers. Its 609 MiB of weights stay in pinned host RAM and stream over PCIe one layer at a time, so it takes no VRAM and the context stays at 64K. See [Images](#images).
- **Experts that don't fit in VRAM are read from host RAM.** Upstream keeps every weight in VRAM (15.6 GiB at 8K context). Here, the expert banks beyond the VRAM budget go into one mapped, pinned host buffer that the GPU reads zero-copy over PCIe. VRAM goes to the looped core layers first, because layers 6–17 run three times per token. If everything fits, nothing moves.
- **`--uncensor PATH` applies the [AtomicChat uncensored LoRA](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) exactly.** That rank-1 LoRA is a projection removal, `y -= b·(b·y)`, on the attention output and on every expert's down projection, plus an edit to the embedding table. A new kernel applies the projection before the short conv. Upstream's `--ablation` hook runs after the conv, so it can't reproduce the LoRA. [`tools/uncensor-lora/`](tools/uncensor-lora/) checks the equivalence on the real weights and bakes the embedding edit into a reflinked copy of the checkpoint.
- **Loads in 16–18 s instead of ~2.5–3 min.** Layers 20–23 ship as MXFP8 and are converted to NVFP4 at load. For banks bound for host RAM, the conversion now runs in VRAM staging and the result is copied to the host by DMA; storing straight into mapped host memory ran at ~12 MB/s. Large safetensors files are read with `O_DIRECT` through a pinned bounce buffer.
- **Less fixed VRAM.** The DSpark drafter loads only when `--mtp` > 0, and `SQUID_PREFILL_CHUNK` caps the prefill batch. `[mem]` log lines report free VRAM after init and after a run.

## Build and run on an 8 GB card

```bash
git clone https://github.com/bigtimer-dev/squid5090-8gb.git && cd squid5090-8gb
zig build -Doptimize=ReleaseFast -Dcuda-arch=89 -Dcuda-path=/opt/cuda   # 89 = RTX 40xx (Ada)

# Weights: thinkingmachines/Inkling-2-Mini, see upstream step 3 below
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --mode nvfp4 --embed-host --mtp 0 --ctx 65536 --kv fp8 --host 127.0.0.1 --port 8080
```

Image input links the system libpng, libjpeg-turbo (2.x or 3.x) and libwebp: `libpng libjpeg-turbo libwebp` on Arch, `libpng-dev libturbojpeg0-dev libwebp-dev` on Debian and Ubuntu.

The load log shows the split, for example `[weights] Experts: 18/48 banks in VRAM, 30 in pinned host RAM (8.01 GiB, zero-copy) | host bank-visits/token=42/96`. On an 8 GB card about 10 GiB of RAM is pinned: the offloaded experts, the embedding table and the prompt cache.

For the uncensored model, build its folder with [`tools/uncensor-lora/make_uncensored.py`](tools/uncensor-lora/make_uncensored.py) (the steps are in its header), serve that folder, and add `--uncensor <folder>/uncensor.safetensors`.

## Fast generation: CPU + GPU experts

```bash
export SQUID_CPU_MOE=1 SQUID_CPU_MOE_GPU_SLOTS=1 SQUID_CPU_MOE_DOWN=kouter SQUID_CPU_MOE_L2PF=2 \
       SQUID_CPU_MOE_KEEPALIVE=20 SQUID_MOE_UP_PERSIST=1 SQUID_MOE_DOWN_PERSIST=1 SQUID_CTRL_SPLIT=1 \
       SQUID_DECODE_CACHE=tools/decode-cache/inkling-2-mini-routing-prior.bin SQUID_DECODE_CACHE_LAYERS=host \
       SQUID_DECODE_CACHE_PCIE=2 SQUID_DECODE_CACHE_ADAPT=16,16,0.9,4,reset SQUID_DECODE_CACHE_TAIL=256 \
       SQUID_DECODE_CACHE_DEMOTE=300 SQUID_DECODE_CACHE_LAZY=1
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --mode nvfp4 --embed-host --mtp 0 --ctx 65536 --kv fp8 --host 127.0.0.1 --port 8080
```

How one generated token runs at 64K context with fp8 KV on the 8 GB card:
- **9 layers keep their experts in VRAM.** They account for 27 of the 48 layer visits per token, because layers 6–17 run three times. Their experts run on the unchanged GPU kernels.
- **The other 15 layers keep their experts in pinned host RAM** and get 21 visits per token. In each of these visits:
  - The routed experts that are in the VRAM decode cache run on the GPU.
  - When at most 2 of the 8 are cached, the GPU also reads one more over PCIe.
  - Meanwhile the CPU computes the rest straight from RAM: 8 threads pinned to CPUs 0–7, AVX-512 VNNI on int16 activations, with work stealing. The GPU then adds the CPU's partial sum.
- **The decode cache lives in VRAM that sits idle while one sequence generates:**
  - The prefill buffers hold 169 experts (722 MiB) on this card.
  - The unused ends of the 16 global-layer KV arrays hold up to 224 more (957 MiB, `SQUID_DECODE_CACHE_TAIL`). In a conversation shorter than the context limit, most of each array is empty. A slot is switched off before the conversation reaches it: 224 are usable on a short prompt, 192 at 12.6K tokens, 112 at 32K and none past ~61K.
  - The 300 least-used experts of the 9 VRAM layers (4% of those layers' routing, by the prior) are handed to the CPU, which computes them from pinned RAM copies (1.28 GiB, `SQUID_DECODE_CACHE_DEMOTE`). Their VRAM holds 300 more host experts while a sequence generates. Prefill needs every expert on the GPU, so they are copied back after each answer and their slots are refilled for the next one (in the background with `SQUID_DECODE_CACHE_LAZY=1`).
  - Each request starts from the experts that a routing prior ranks highest ([`tools/decode-cache/`](tools/decode-cache/)).
  - Every 16 tokens, up to 16 cached experts are swapped for ones this request uses more.
  - Every prompt overwrites the prefill buffers and the demoted experts' slots, so that part of the cache is refilled for each request (~200 ms of copies with `SQUID_DECODE_CACHE_DEMOTE=300`). With `SQUID_DECODE_CACHE_LAZY=1` the refill runs in the background while the reply starts: chunks of 32 experts go over on a second stream, and each chunk's experts move from the CPU to the GPU at the next token boundary once it has landed. Nothing waits for it, so the second token comes 0.8 ms after the first instead of ~170 ms later; a 64-token reply runs at 52.3 tok/s after its first token instead of 47.0, and a 256-token one at 53.2 instead of 52.2. Copying the demoted experts back runs after the answer has gone out; it delays the next prompt by up to ~65 ms only when requests come back to back. The KV-tail part is put back after the answer, so it adds no wait.
- **`SQUID_CPU_MOE_KEEPALIVE=20` keeps the AMD fabric clock up.** While generating, one CPU thread reads a private 64 MB buffer at up to 20 GB/s. Without that traffic, the fabric clock (FCLK) drops from 1960 to 1400 MHz during the GPU-only stretches of each token, and the CPU then reads RAM up to 30% slower.

Requirements and costs:
- **CPU:** needs AVX-512 (F/BW/DQ/VBMI, plus VNNI for the int16 kernel). Without it the log prints `[cpu-moe] needs AVX-512 ...` and the engine stays GPU-only.
- **Scope:** only one-sequence generation uses this path; batched prefill and multi-sequence decode don't.
- **CPU load:** 8 cores work while a sequence generates and sleep when the server is idle.

**Numerical change:** greedy output is not bit-identical to the GPU-only path, because the CPU kernels round differently from the GPU's. It was compared with the GPU-only path on the uncensored model, teacher-forced:
- Mean NLL difference: −0.0043 ± 0.0024 nats on 1024 tokens of upstream's README, and +0.0026 ± 0.0017 on 256 tokens of a greedy answer.
- The top token agreed at 1003/1024 and 255/256 positions. Every disagreement was a near-tie (top-2 margin ≤ 0.23).
- With `SQUID_DECODE_CACHE_DEMOTE=300` (the demoted experts also move to the CPU's arithmetic): −0.0039 ± 0.0025 and +0.0018 ± 0.0018 nats, top token 1003/1024 and 253/256 (margins ≤ 0.23), and at positions 12.6K–12.9K +0.00001 ± 0.00001 nats with 256/256.
- With `SQUID_DECODE_CACHE_LAZY=1` as well (the first tokens of a reply compute more experts on the CPU while the refill lands): −0.0026 ± 0.0032 and +0.0023 ± 0.0019 nats, top token 1002/1024 and 254/256, and at 12.6K +0.00000 ± 0.00001 with 256/256.
- A fact placed at 50% depth of a 32K-token prompt and at 10% depth of a 61K-token prompt was still retrieved, and tool calls pass, plain and streamed.

**Numerical change in both modes:** decode attention over a long context now splits the context into 24 parts instead of 64, which fills the 4060's 24 SMs in two whole waves. This changes the order of an fp32 merge, so output also differs slightly from upstream's GPU-only path once the context passes 1024 tokens. On 1024 teacher-forced tokens at positions 12.6K–13.6K, GPU-only: mean NLL +0.00003 ± 0.00002 nats, top token 1024/1024, largest change in a token's log-probability 0.012. The split count is `KV_SPLIT_K` in `cuda/config.hpp`; 64 restores upstream's arithmetic.

## Images

Put images in the `content` parts of user or tool messages as `image_url` parts with a base64 data URL. The part types `input_image` and `image` also work. The server doesn't fetch `http(s)` URLs.

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}}]}]}'
```

- **Token cost:** one prompt token per 40×40 pixels. A 640×480 image is 192 tokens and 1920×1080 is 1296. An image above `SQUID_IMAGE_MAX_TOKENS` (default 1600) is downscaled by area averaging to fit, keeping its aspect ratio; with the default settings the prefill buffers fit at most 3548. Transparent images are composited onto white.
- **The encoder** is a 23-layer ViT with 2D RoPE, a 2×2 patch merge and a two-layer projector. It runs as FP16 tensor-core GEMMs with FP32 accumulation, an FP32 residual stream and FP32 flash-style attention. The image embeddings replace the placeholder tokens after the token embedding.
- **Encoder time on the RTX 4060 Laptop:** ~50 ms for a small image (the weights stream over PCIe each time), 70 ms for 640×640, 0.9 s for 1920×1080, and 1.3 s for 4000×3000 (downscaled to 1564 tokens). The first image also loads the weights from the checkpoint, which takes ~1 s.
- **Caching:** recent images' embeddings are cached by content hash. The prompt cache reuses an image's span across the turns of a conversation only when the same image sits at the same position.
- **Accuracy:** the encoder matches a float64 NumPy port of llama.cpp's Inkling 2 vision graph to a relative L2 error of 0.07–0.16% on the projector output ([`tools/vision/`](tools/vision/)). `squid5090 vision-test MODEL_DIR --image FILE` runs the encoder alone; `SQUID_VISION_DUMP=prefix` writes its intermediate tensors.
- **Text-only requests** run exactly as before.

## Results on the RTX 4060 Laptop (8 GB)

Both engines got the same prompts: 64K context, 8-bit KV cache, the uncensored model, temperature 0. llama.cpp runs the Q4_K GGUF with the experts on the CPU except for 6 loop layers (`-ub 64`, `q8_0` KV, 8 threads). "GPU experts" is this fork's default; "CPU + GPU experts" uses the settings above.

| | llama.cpp | GPU experts | CPU + GPU experts |
|---|---:|---:|---:|
| Generation, short prompt | 18.6 tok/s | 12.5 tok/s | **54–55 tok/s** |
| Generation after 12.6K tokens of context | 15.9 tok/s | 12.2 tok/s | **46.9 tok/s** |
| Generation after 32K tokens of context | — | — | **42.4 tok/s** |
| Generation after 46K tokens of context | — | 11.7 tok/s | — |
| Generation after 64K tokens of context | — | — | **37.1 tok/s** |
| Reading a 3.8K-token prompt | 76 tok/s (51 s) | **509 tok/s (7.5 s)** | same path |
| Reading a 12.6K-token prompt | 70 tok/s (181 s) | **495 tok/s (25.5 s)** | **500 tok/s (25.2 s)** |
| Reading a 46K-token prompt | — | **500 tok/s (93 s)** | same path |
| Reading a 64K-token prompt | — | — | **498 tok/s (128 s)** |

With CPU + GPU experts the short-prompt speed is ~2 tok/s lower at first than after a few minutes of steady load. The change follows the CPU's RAM bandwidth, which rises from ~48 to ~54 GB/s; the cause is unknown and is not temperature.

With GPU experts, generation is limited by PCIe: at 64K context, 42 of the 96 expert-bank visits per token are read from host RAM. Each bank that fits in VRAM adds about 0.5 tok/s, so a smaller context helps: `--ctx 32768` reaches 14.0 tok/s, and `--kv q4` reaches 13.6 tok/s at 64K (q4 quality not evaluated). With CPU + GPU experts, the limits are the CPU's RAM bandwidth and the many small GPU kernels per layer. On this Ryzen 9 8945HS, RAM reads top out at 62.7 GB/s at FCLK 1960 MHz, and the path reaches ~54. Tool calls (plain and streamed) and the prefix cache work in both modes: the second turn of a 12.6K-token conversation starts in 0.8 s.

What didn't help on this machine:
- MTP drafting: 13.6 tok/s with it versus 14.6 without, at 8K context, because verifying drafts pulls in more experts over PCIe.
- Write-combined host memory: no change.
- A 1024-token prefill chunk: generation gains ~1 tok/s, but prompts are read at only ~410 tok/s.
- Transparent huge pages for the host experts (`SQUID_HOST_THP=1`): no change.
- Keeping the decode cache from one request to the next instead of starting from the prior: ~3% slower when the topic changes.
- A second decode-cache bank (`SQUID_DECODE_CACHE_N2=128` with `SQUID_EXPERT_VRAM_MB=4380`): generation +1.4–1.9%, but short prompts take +130 ms. Left off.
- Demoting more VRAM-layer experts: 100, 200 and 300 give +2.0%, +4.5% and +6.5% on a short prompt; 384 is 1% slower than 300.

Only `--mode nvfp4` has been tested with the offload.

### Fork-only options

| | Default | Effect |
|---|---|---|
| `--uncensor PATH` | off | Projection directions to remove (`layer_dirs`, F32 `[24, 2304]`) |
| `SQUID_VRAM_MARGIN_MB` | `192` | VRAM left free after weights and runtime buffers |
| `SQUID_EXPERT_VRAM_MB` | auto | Fixed VRAM budget for expert banks |
| `SQUID_PREFILL_CHUNK` | `2048` | Prefill batch size (32–2048, multiple of 8) |
| `SQUID_HOST_WC` | `0` | `1` = write-combined host expert buffer |
| `SQUID_LOAD_TRACE` | `0` | `1` = per-phase and per-bank load timings |
| `SQUID_CPU_MOE` | `0` | `1` = the CPU computes host-RAM experts in one-sequence decode |
| `SQUID_CPU_MOE_THREADS` | `8` | CPU worker threads, pinned to CPUs 0..n-1 (`SQUID_CPU_MOE_PIN=0` unpins, `SQUID_CPU_MOE_CPUS=a,b,...` picks CPUs) |
| `SQUID_CPU_MOE_GPU_SLOTS` | `0` | The GPU computes this many of a host visit's experts over PCIe while the CPU works (`1` is best here) |
| `SQUID_CPU_MOE_DOWN` | row order | `kouter` = down projection in k-outer order (faster here, bit-identical) |
| `SQUID_CPU_MOE_L2PF` | off | While the GPU waits for the CPU, read the next visit's dense weights (QKVRG + o_proj) into its L2: `1` plain loads, `2` evict-last priority |
| `SQUID_CPU_MOE_KEEPALIVE` | off | GB/s of background RAM reads while decoding, to hold the fabric clock up |
| `SQUID_MOE_UP_PERSIST`, `SQUID_MOE_DOWN_PERSIST` | `0` | `1` = persistent-grid GPU expert kernels (one wave of CTAs loops over the work) |
| `SQUID_CTRL_SPLIT` | `0` | `1` = split the controller kernel in two for one-sequence decode (more parallel) |
| `SQUID_DECODE_CACHE` | off | Routing prior (`[24][128]` u32 counts) that seeds the VRAM decode cache |
| `SQUID_DECODE_CACHE_LAYERS` | `loop` | Layers the cache may hold: `loop`, `host` (all CPU-served layers) or a hex mask |
| `SQUID_DECODE_CACHE_PCIE` | `-1` | On a visit with at most this many experts cached, the GPU also reads one uncached expert over PCIe |
| `SQUID_DECODE_CACHE_ADAPT` | off | `N,M,decay,W[,reset]`: every N tokens swap up to M experts by score W·prior + decayed session uses; `reset` starts each request from the prior |
| `SQUID_DECODE_CACHE_N`, `SQUID_DECODE_CACHE_N2` | all that fit, `0` | Cap on cached experts; extra slots in a dedicated VRAM bank |
| `SQUID_DECODE_CACHE_TAIL` | `0` | Up to this many more cached experts in the unused ends of the global-layer KV arrays (224 fit with `--ctx 65536 --kv fp8`). A slot is switched off 64 positions before the sequence reaches it |
| `SQUID_DECODE_CACHE_DEMOTE` | `0` | The CPU computes this many of the least-used VRAM-layer experts (by the prior) from pinned RAM copies, and their VRAM slots cache that many more host experts while generating. Needs `--mtp 0` |
| `SQUID_DECODE_CACHE_LAZY` | `0` | `1` = refill each request's cache slots in the background (chunks of 32 experts on a second stream) instead of before the second token; experts not landed yet run on the CPU |
| `SQUID_LOCAL_KV` | `bf16` | `fp8` = K/V of the 40 sliding-window visits as fp8 E4M3 with a per-(token, head) scale: generation +1.9% at 12.6K and +1.5% at 32K. **Lossy**: teacher-forced NLL vs the GPU-only path −0.0048 ± 0.0043 and +0.0067 ± 0.0026 nats (default −0.0039 and +0.0018), top token 995/1024 and 251/256, so it is off by default |
| `SQUID_VISION` | `1` | `0` = no image input (requests with images get an error) |
| `SQUID_IMAGE_MAX_TOKENS` | `1600` | Prompt tokens per image; larger images are downscaled |
| `SQUID_ROUTE_TRACE` | off | Path: write every decode token's top-8 experts per layer visit at exit (input for `tools/decode-cache/route_prior.py`) |

The other `SQUID_CPU_MOE_*` variables and `SQUID_OPROJ_POST_FUSED` (on: o_proj runs inside the post-attention kernel for one token), `SQUID_DECODE_BANK`, `SQUID_DECODE_CACHE_TAIL_CHECK`, `SQUID_DECODE_CACHE_HOLE_CHECK`, `SQUID_E2M1_LUT`, `SQUID_HOST_THP`, `SQUID_BLOCKING_SYNC`, `SQUID_ENGINE_CPU`, `SQUID_ATTN_NO_PREFETCH` and `SQUID_SCORE_*` are experiment or diagnostic switches. Their defaults are the measured best.

---

# squid5090

**A single-binary, zero-Python TigerStyle Zig + Blackwell CUDA (`sm_120a`) inference engine and OpenAI-compatible API server for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) — hitting `589.5 tok/s` code decode (`771.6 tok/s` on code-copy/RAG), `9,181 tok/s` prefill, and `1,048,576` (1M) context on a single RTX 5090.**

![squid5090 RTX 5090 Benchmark & 1M Context Scaling](assets/benchmark.png)

> **Want to understand how it works?**
> - **[`LITE.md`](LITE.md)** — Plain-English, 3Blue1Brown-style visual guide to every optimization and how runtime-gated abliteration works step by step.
> - **[`LEARNINGS.md`](LEARNINGS.md)** — Complete technical reference: every Blackwell kernel optimization, roofline audit vs. [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill) (including what's still left on the table), and the `uv` abliteration pipeline.

---

## Why `squid5090` on an RTX 5090?

Single-user (`c=1`) comparison for **Inkling-2-Mini** on one **NVIDIA RTX 5090 (32 GiB GDDR7)**:

| Inference Engine | Runtime Stack | Decode Speed (`c=1`) | Max Context (32 GiB) | VRAM / Host Spill | Key Benefits & Tradeoffs |
|---|---|---:|---:|---|---|
| **PyTorch / HF** | Python + PyTorch | ~10–15 tok/s | ~64K | >24 GiB | Reference baseline; eager Python overhead |
| **`llama.cpp` / Ollama** | C++ (GGUF) | ~90–97 tok/s | ~256K | ~18–22 GiB | No native NVFP4 Blackwell PTX or DSpark MTP |
| **vLLM / SGLang** | Python + CUDA | ~190–350 tok/s | ~256K–512K | ~22–28 GiB | Built for multi-user batching; Python overhead at `c=1` |
| **Basalt** *(same 5090)* | Native NVFP4 + Q8 | 261–325 tok/s | 256K | 22.9 GiB + **8.9 GiB RAM spill** | Spills weights to host RAM; capped at 256K |
| **`ingine`** *(published ref)* | C++ + CUDA | 561 tok/s *(chat: 325)* | 256K | ~18–22 GiB | Adaptive `{2,4,8}` DSpark; capped at 256K |
| **`squid5090` (this repo)** | **1 Zig binary (`0` Python)** | **589.5 tok/s** code<br>**771.6 tok/s** code-copy/RAG<br>**332.5 tok/s** chat<br>**541.1 tok/s** ablated | **1,048,576 (1M)** | **15.6 GiB** (8K)<br>**23.2 GiB** (1M)<br>**0 GiB RAM spill** | **2 `cudaMalloc` arenas**, zero post-init `malloc`, **9,181 tok/s** prefill (`8,755` at 20K, `7,742` at 64K), **62–72 ms** warm prefix cache, optional 1.8 MB `--ablation` |

---

## Quickstart (Linux Build & Install)

### 1. Linux Requirements

- **OS:** Linux x86_64 (tested on Ubuntu 24.04 / Debian 12–13)
- **GPU:** NVIDIA Blackwell GPU (`sm_120a`, e.g., RTX 5090) with ≥ 16 GiB free VRAM (≥ 23.5 GiB for full 1M context)
- **NVIDIA Driver:** `580+` (tested on `590.48.01`)
- **CUDA Toolkit:** CUDA `13.0+` with `nvcc` and `cuBLAS` (tested on CUDA `13.1` at `/usr/local/cuda`)
- **Build Tools:** `g++`, `git`, `curl`, and **Zig `0.17.1+`**

```bash
# Install host C++ compiler (required by nvcc) and basics on Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y build-essential git curl

# Verify CUDA 13+ and Zig 0.17.1+ are on PATH
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
zig version
```

*Runtime links only against `libc`, `libstdc++`, `libcudart`, `libcublas`, and `libpthread` — zero Python, CMake, Make, or external packages.*

### 2. Clone & Build

```bash
git clone git@github.com:tensorfish/squid5090.git
cd squid5090

# Compiles all sm_120a CUDA kernels in parallel (content-hashed cache) + ReleaseFast Zig
zig build -Doptimize=ReleaseFast
```

The single binary is built at `zig-out/bin/squid5090` (~30 s clean build, ~24 ms incremental).

### 3. Download Inkling-2-Mini Weights

Download [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (~20.2 GB) into a local folder (e.g., `/opt/models/inkling-2-mini`):

```bash
uvx --from huggingface-hub hf download thinkingmachines/Inkling-2-Mini \
  --local-dir /opt/models/inkling-2-mini
```

### 4. Run the OpenAI API Server or CLI

```bash
# Full 1M-context OpenAI HTTP/SSE server (~23.2 GiB VRAM, adaptive {2,4,8} DSpark + prompt-lookup)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode nvfp4 --kv q4 --ctx 1048576 --mtp 7 --spec-threshold 0.80

# Optional: enable 1.8 MB runtime-gated abliteration (zero weight modification)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode nvfp4 --kv q4 --ctx 1048576 --mtp 7 \
  --ablation /opt/models/inkling-2-mini/inkling-2-mini-abliterated.safetensors

# One-shot CLI generation
./zig-out/bin/squid5090 generate /opt/models/inkling-2-mini \
  --prompt "Write a Python LRU cache with tests." \
  --mode nvfp4 --kv q4 --mtp 7 --max-tokens 1024

# Benchmark & 1M long-context scaling sweep
./zig-out/bin/squid5090 bench /opt/models/inkling-2-mini \
  --prompt "Write a Python LRU cache with tests." --mode nvfp4 --kv q4 --mtp 7 --max-tokens 1024 --runs 3
./zig-out/bin/squid5090 longctx /opt/models/inkling-2-mini --mode nvfp4 --kv q4 --ctx 1048576
```

**Test the server (`curl` or OpenAI Python SDK):**
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Write a 1-line Python function to reverse a string."}],
    "max_tokens": 256,
    "stream": true
  }'
```

**Install as a persistent `systemd` service:**
```bash
sudo cp deploy/squid5090.service /etc/systemd/system/squid5090.service
sudo systemctl daemon-reload && sudo systemctl enable --now squid5090.service
curl -s http://localhost:8000/health
```

### Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `nvfp4` (`serve`) | Weight precision: `nvfp4` \| `fp8` \| `exact` |
| `--kv` | `auto` | Global KV cache format: `bf16` \| `fp8` (up to 432K) \| `q4` (up to **1M**) |
| `--ctx` | `65536` (`serve`) | Context window size up to `1048576` |
| `--mtp` | `7` (`serve`) | Built-in DSpark drafter (`0` = off, `7` = adaptive `{2,4,8}` CUDA Graph switch) |
| `--spec-threshold` | `0.80` | Markov confidence threshold for `{2,4,8}` verify width (`0.0` = fixed `B=8`) |
| `--no-prompt-lookup` | off | Disable hybrid O(1) prompt-lookup drafter (enabled by default) |
| `--ablation` | none | Optional 1.8 MB runtime-gated Fisher-LDA `.safetensors` (see [`tools/ablate.py`](tools/ablate.py)) |
| `--api-key` | `$SQUID_API_KEY` | Optional `Authorization: Bearer <key>` auth for `serve` |

---

## Documentation & Attribution

- **[`LITE.md`](LITE.md)** — High-level, 3Blue1Brown-style plain-English guide to LLM optimization and abliteration.
- **[`LEARNINGS.md`](LEARNINGS.md)** — Consolidated technical reference: GPU kernels, roofline audit vs. [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill), unexploited future levers, and the `uv` abliteration pipeline.
- **[`NOTICE`](NOTICE)** — Open-source attributions (`satellitedown/inkling-2-mini-abliterated-ingine`, `AtomicBot-ai/atomic-engine`, `sybil-solutions/inference-optimisation-skill`, `thinkingmachines/Inkling-2-Mini`).
