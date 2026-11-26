# squid5090-8gb

**A fork of [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090) (based on `3ee093c`) that runs Inkling-2-Mini on an 8 GB GPU.** Tested on an RTX 4060 Laptop GPU (`sm_89`, 8 GB VRAM, PCIe 4.0 x8) with 30 GB of RAM. Upstream's README follows below, unchanged.

## What this fork changes

- **Generation is ~4× faster when the CPU computes the experts that live in RAM** (opt-in, see [Fast generation](#fast-generation-cpu--gpu-experts)): 54–55 tok/s instead of 12.5 on a short prompt at 64K context. While one sequence generates, the CPU computes the routed experts held in host RAM straight from RAM and the GPU computes the rest. Up to ~690 of the most-used host experts are also copied into VRAM that sits idle during generation: the prefill buffers, the unused end of the KV cache, and the slots of the 300 least-used VRAM-layer experts, which the CPU computes instead.
- **Speculative drafting works with CPU + GPU experts** (`--mtp 7`, see [Drafting](#drafting-on-the-cpu--gpu-path)): ~67 tok/s on a short prompt instead of 54–55, and 74 instead of 48 tok/s over replayed OpenCode requests. The built-in DSpark drafter proposes 7 tokens and one target pass checks 1, 3 or 7 of them, depending on the drafter's confidence. In that pass the CPU computes the RAM experts of all the tokens in one request, and the GPU reads each cached expert once for all of them. Prompts take ~10% longer with the drafter on.
- **Prompts of up to ~80 tokens are read 1.7–2.9× faster with CPU + GPU experts** (see [Reading prompts](#reading-prompts-on-the-cpu--gpu-path)): the last ≤ 160 tokens of a prompt go through the generation path instead of a batched chunk that streams the host experts over PCIe. A 41-token prompt takes 0.50 s instead of 1.22 s. Prefill chunks also use 128-token tiles at every size when experts live in host RAM, which reads them fewer times: a 500-token prompt takes 2.2 s instead of 3.3, and 2.0 s instead of 2.9 with GPU experts. Replayed OpenCode requests spend 18% less time reading prompts.
- **Prompts longer than 160 tokens are read 1.3–1.8× faster when experts live in host RAM** (see [Reading prompts](#reading-prompts-on-the-cpu--gpu-path)). Prefill now copies the host experts into VRAM with the copy engine, a group at a time, and multiplies them there. Before, each 128-token tile read its experts over PCIe again. With CPU + GPU experts and `--mtp 7`, a 32K-token prompt takes 37.7 s instead of 66.9 s, and a 500-token prompt 1.44 s instead of 2.17 s. With GPU experts, a 12.6K-token prompt takes 15.3 s instead of 24.5 s. The output is bit-identical. Replayed OpenCode requests spend 35% less time reading prompts.
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
  --mode nvfp4 --embed-host --mtp 7 --ctx 65536 --kv fp8 --host 127.0.0.1 --port 8080
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

### Drafting on the CPU + GPU path

With `--mtp 7`, the DSpark drafter drafts 7 tokens after each target pass, and the next target pass checks drafts as a batch of 2, 4 or 8 tokens. Upstream's adaptive switch picks the width from the number of leading drafts whose confidence is at least `--spec-threshold` (0.80). In the batched pass:
- **The CPU gets one request for all the tokens.** It computes each token's RAM experts, reading an expert that several tokens share once.
- **The GPU runs the cached experts of all the tokens in one up and one down launch** and reads each distinct expert once: each token is one column of the tensor-core tile, with the same per-token sums as the one-token kernels.
- **Sampling stays on the GPU,** so the output follows the target's distribution. Greedy text matches `--mtp 0` until the first near-tie, where the batched arithmetic can pick the other token. On three of four test prompts that happened after 202, 81 and 39 tokens; the fourth matched to the end.

A batched pass reads the union of its tokens' experts. Two tokens cost ~1.5 one-token passes, four ~2 and eight ~2.9. On chat and code text, acceptance falls quickly with draft depth, so the switch mostly checks one draft: on a code prompt it ran 81 passes of width 2, 25 of width 4 and 10 of width 8. Always checking all 7 drafts (`SQUID_SPEC_ADAPT=0`) is slower than not drafting on such text (44 vs 54 tok/s), but it is the fastest option on text that copies its context.

Measured with the settings above, between `--mtp 0` runs before and after (median of 6 runs, 3 for 12.6K; temperature 0 unless noted):

| | `--mtp 0` | `--mtp 7` |
|---|---:|---:|
| Generation, code prompt | 54.4 / 55.1 tok/s | **67.6 tok/s** |
| Same, temperature 1.0 | 52.9 / 52.7 tok/s | **70.6 tok/s** |
| Generation, chat prompt | 55.7 / 55.7 tok/s | **67.4 tok/s** |
| Generation, editing a file (copies its context) | 52.2 / 52.2 tok/s | **102.4 tok/s** |
| Generation after a 12.6K-token prompt (repeats its context) | 46.6 / 46.7 tok/s | **95.2 tok/s** |
| Reading that 12.6K-token prompt (before [the prompt changes](#reading-prompts-on-the-cpu--gpu-path)) | **25.1 s** | 27.6 s |
| 42 replayed OpenCode requests: generation | 47.8 / 47.8 tok/s | **74.2 tok/s** |
| Same: prompt reading, total (before the prompt changes) | **165.3 s** | 183.1 s |
| Same: whole replay | 308 / 311 s | **278 s** |

Prompts take ~10% longer because the drafter also runs over every prompt token to fill its own KV cache. Tool calls, the needle tests at 32K and 62K, the KV-tail and hole checks, and image input pass with `--mtp 7`.

### Reading prompts on the CPU + GPU path

Upstream reads a prompt in batched chunks and ends with passes of up to 8 tokens. With experts in host RAM, both read those experts over PCIe: a chunk reads every host expert its tokens route to, which costs ~1.1 s even for 32 tokens, and an 8-token pass reads its tokens' experts in ~250 ms. Three changes:
- **The last ≤ 160 tokens of a prompt run on the generation path,** 8 per pass, like the passes that check drafts: the CPU computes the RAM experts and the GPU computes the cached ones, in ~75 ms per pass. The decode cache is set up first, so its background refill runs while the prompt is read, and the first tokens of the reply come faster too. Longer prompts are still chunked, down to their last ≤ 8 tokens. 160 is about the break-even point on this machine (`SQUID_PREFILL_CPU_MAX`; `0` restores upstream).
- **Prefill chunks use 128-token tiles at every size** when any expert bank is in host RAM. Upstream uses 32-token tiles for chunks of up to 512 tokens. Each tile reads its experts' weights, so with small tiles a 500-token chunk read the host experts several times and took longer than an 800-token chunk. The output is bit-identical (`SQUID_PREFILL_TM32_MAX`). This also applies with GPU experts. When every bank fits in VRAM, upstream's tiles are kept.
- **Prefill copies the host experts into VRAM before multiplying them.** A chunk's expert GEMMs read each host expert from RAM once per 128-token tile that routes to it, in 64-byte pieces, and the GPU's L2 cache doesn't keep those reads. A 2048-token chunk got ~5 GB/s of weights out of a link that does 13.3. Now the copy engine copies up to 22 experts at a time into one of two VRAM slots, on its own stream. The GEMM for that group reads them from VRAM while the next group is copied. The tiles and their arithmetic don't change, so the output is bit-identical. This also applies with GPU experts. The slots take 128 MiB of VRAM that was free (free after init: 338 → 210 MiB). They get less if fewer than 192 MiB would stay free. With GPU experts, whose banks fill VRAM, they get 16 MiB, which still gives most of the gain (`SQUID_PREFILL_STAGE_MB`; `0` reads host RAM directly).

The first two changes, measured with the settings above and `--mtp 7` against the build before them, run before and after (median of 3 runs, 32-token replies):

| Prompt | Before | Now | Generation of the reply, before → now |
|---|---:|---:|---:|
| 41 tokens | 1.22 s | **0.50 s** | 62.9 → 69.9 tok/s |
| 81 tokens | 1.43 s | **0.83 s** | 63.8 → 73.1 tok/s |
| 130 tokens | 1.74 s | **1.28 s** | 63.7 → 76.4 tok/s |
| 201 tokens | 1.94 s | **1.67 s** | 63.3 → 67.3 tok/s |
| 300 tokens | 2.48 s | **1.81 s** | 69.8 → 73.5 tok/s |
| 500 tokens | 3.33 s | **2.17 s** | 63.9 → 68.4 tok/s |
| 800 tokens | 2.79 s | **2.54 s** | 53.6 → 56.4 tok/s |
| 12.6K tokens | 27.6 s | **26.9 s** | |
| 32K tokens | 67.2 s | **66.9 s** | |
| 42 replayed OpenCode requests: prompt reading, total | 183.2 / 183.1 s | **150.9 s** | 76.1 / 74.3 → **79.8 tok/s** |

With GPU experts (`--mtp 0`) the tiles alone take a 130-token prompt from 1.48 to 1.36 s, 300 tokens from 2.14 to 1.68 s and 500 tokens from 2.91 to 2.03 s.

The expert stage, measured the same way against the build before it, run before and after (median of 3 runs up to 3K tokens, one run each for longer prompts):

| Prompt | Before | Now |
|---|---:|---:|
| 41 tokens | 0.49 s | 0.49 s |
| 201 tokens | 1.68 / 1.68 s | **1.29 s** |
| 300 tokens | 1.82 / 1.81 s | **1.36 s** |
| 500 tokens | 2.17 / 2.16 s | **1.44 s** |
| 800 tokens | 2.56 / 2.57 s | **1.63 s** |
| 1.5K tokens | 3.49 / 3.50 s | **2.01 s** |
| 3K tokens | 6.84 / 6.86 s | **3.89 s** |
| 12.6K tokens | 26.9 / 26.8 s | **15.1 s** |
| 32K tokens | 66.9 / 66.9 / 66.9 s | **37.7 / 37.8 s** |
| 63.6K tokens | 139.8 s ¹ | **82.3 s** |
| 42 replayed OpenCode requests: prompt reading, total | 150.3 / 150.3 s | **97.2 s** |

Prompts of up to 160 tokens take the generation path, so they don't change. Generation speed doesn't change either: 100.0 / 100.4 → 100.0 tok/s after the 12.6K-token prompt, 79.6 / 81.4 → 80.0 tok/s over the OpenCode replay. With GPU experts (`--mtp 0`), 500 tokens go from 2.03 / 2.01 to 1.47 s, 12.6K tokens from 24.5 / 24.5 to 15.3 s, and 32K tokens from 61.0 to 38.1 s ¹. Greedy output matched the build before it on 201-, 800-, 1.5K- and 12.6K-token prompts at `--mtp 0` and `--mtp 7`, and with GPU experts. Tool calls, the needle tests at 32K and 62K, the KV-tail and hole checks, and image input pass. ¹ The new build with `SQUID_PREFILL_STAGE_MB=0`, which reads prompts as fast as the build before it.

Over a 256-token reply to a 25-token code prompt the gain in generation is gone: 66.4 tok/s against 67.4 and 67.3. The prompt is computed differently, so the reply's text changes, and with it the share of accepted drafts (59.1% instead of 61.5%).

**Numerical change:** the prompt tokens that take the generation path get its arithmetic, including the CPU kernels' rounding. In 8 greedy test runs the text matched the previous build for the first 9 to 222 tokens, then diverged. Teacher-forced on five texts, the mean NLL difference from the previous build is +0.0001 ± 0.0010 nats, pooled. Against the GPU-only path it is −0.0007 ± 0.0028 nats on 1024 tokens of upstream's README (top token 997/1024), and between −0.0033 and +0.0062 nats on four 256-token answers (top token 251–255/256). Of the 40 top-token disagreements, 37 were near-ties (top-2 margin ≤ 0.3); the other three had margins of 0.37, 0.44 and 0.97. The previous build had 34 disagreements on the same texts, all ≤ 0.3. With the tile change alone the output is bit-identical. Tool calls, the needle tests at 32K and 62K, the KV-tail and hole checks, and image input pass. Images in a short prompt are injected on the generation path.

Requirements and costs:
- **CPU:** needs AVX-512 (F/BW/DQ/VBMI, plus VNNI for the int16 kernel). Without it the log prints `[cpu-moe] needs AVX-512 ...` and the engine stays GPU-only.
- **Scope:** one-sequence generation uses this path, including the batched passes that check drafts, and so do the last ≤ 160 tokens of a prompt. Larger prefill chunks and multi-sequence decode don't.
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
| Reading a 3.8K-token prompt † | 76 tok/s (51 s) | **509 tok/s (7.5 s)** | same path |
| Reading a 12.6K-token prompt | 70 tok/s (181 s) | **824 tok/s (15.3 s)** | **837 tok/s (15.1 s)** ‡ |
| Reading a 32K-token prompt | — | **835 tok/s (38.1 s)** | **843 tok/s (37.7 s)** ‡ |
| Reading a 46K-token prompt † | — | **500 tok/s (93 s)** | same path |
| Reading a 64K-token prompt | — | — | **773 tok/s (82.3 s)** ‡ |

† Measured before the expert stage (see [Reading prompts](#reading-prompts-on-the-cpu--gpu-path)), which reads prompts 1.3–1.8× faster in both modes. ‡ With `--mtp 7`; the drafter adds ~10% to prompt reading.

With CPU + GPU experts the short-prompt speed is ~2 tok/s lower at first than after a few minutes of steady load. The change follows the CPU's RAM bandwidth, which rises from ~48 to ~54 GB/s; the cause is unknown and is not temperature.

With GPU experts, generation is limited by PCIe: at 64K context, 42 of the 96 expert-bank visits per token are read from host RAM. Each bank that fits in VRAM adds about 0.5 tok/s, so a smaller context helps: `--ctx 32768` reaches 14.0 tok/s, and `--kv q4` reaches 13.6 tok/s at 64K (q4 quality not evaluated). With CPU + GPU experts, the limits are the CPU's RAM bandwidth and the many small GPU kernels per layer. On this Ryzen 9 8945HS, RAM reads top out at 62.7 GB/s at FCLK 1960 MHz, and the path reaches ~54. Tool calls (plain and streamed) and the prefix cache work in both modes: the second turn of a 12.6K-token conversation starts in 0.8 s.

What didn't help on this machine:
- MTP drafting with GPU experts: 13.6 tok/s with it versus 14.6 without, at 8K context, because verifying drafts pulls in more experts over PCIe. With CPU + GPU experts drafting helps (see [Drafting](#drafting-on-the-cpu--gpu-path)).
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
| `SQUID_DECODE_CACHE_DEMOTE` | `0` | The CPU computes this many of the least-used VRAM-layer experts (by the prior) from pinned RAM copies, and their VRAM slots cache that many more host experts while generating |
| `SQUID_DECODE_CACHE_LAZY` | `0` | `1` = refill each request's cache slots in the background (chunks of 32 experts on a second stream) instead of before the second token; experts not landed yet run on the CPU |
| `SQUID_SPEC_ADAPT` | `1` | With CPU + GPU experts and `--mtp 7`: `0` = always check all 7 drafts instead of the adaptive 2 / 4 / 8 switch |
| `SQUID_SPEC_MULTI` | `1` | `0` = one CPU request per token in a batched pass instead of one for all the tokens |
| `SQUID_SPEC_MTK` | `1` | `0` = run the GPU's cached experts one token at a time in a batched pass |
| `SQUID_PREFILL_CPU_MAX` | `160` | With CPU + GPU experts: a prompt remainder of at most this many tokens runs on the generation path, 8 tokens per pass, instead of a batched chunk; `0` = upstream's passes |
| `SQUID_PREFILL_TM32_MAX` | `0` with host banks, else `512` | Prefill chunks of at most this many tokens use 32-token tiles in the expert GEMMs, larger ones 128 (bit-identical; at most 512) |
| `SQUID_PREFILL_STAGE_MB` | `128` | VRAM for the two slots that prefill copies host-RAM experts into, a group at a time (less if fewer than 192 MiB would stay free; bit-identical); `0` = the expert GEMMs read host RAM directly |
| `SQUID_PREFILL_STAGE_MIN_B` | `0` | Smallest prefill chunk that uses the stage |
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
