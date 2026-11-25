# Vision encoder check

`vit_ref.py` is a float64 NumPy port of llama.cpp's Inkling 2 vision graph (`tools/mtmd`). `cmp.py` compares squid's encoder dumps with it stage by stage: after `ln_pre`, after layer 0, the encoder output and the projector output (the text-side image embeddings). It needs numpy and Pillow.

```bash
M=/opt/models/inkling-2-mini
python3 vit_ref.py $M/model-00001-of-00001.safetensors image.png /tmp/ref
SQUID_VISION_DUMP=/tmp/sq ../../zig-out/bin/squid5090 vision-test $M --image image.png
python3 cmp.py /tmp/ref /tmp/sq
```

The reference uses the image at its own size, so a test image must fit squid's token cap, one token per 40×40 pixels.

Results on the RTX 4060 Laptop, projector output (relative L2, lowest per-token cosine):

| Image | Tokens | rel. L2 | min cos |
|---|---|---|---|
| 120×80 | 6 | 7.4e-4 | 0.999999 |
| 263×181 (padded, position grid resized) | 35 | 7.2e-4 | 0.999999 |
| 640×640 (32×32 grid, no resize) | 256 | 1.31e-3 | 0.999965 |
| 640×488 JPEG photo (decoded by TurboJPEG here, by PIL in the reference) | 208 | 1.60e-3 | 0.999966 |
