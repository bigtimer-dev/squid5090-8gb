#!/usr/bin/env python3
"""cmp.py REF_PREFIX SQUID_PREFIX: compare squid's encoder dumps with the reference, stage by stage."""
import json, sys
import numpy as np
ref, sq = sys.argv[1:3]
d = json.load(open(ref + ".json"))
for ext, cols in ((".lnpre", 1024), (".l0", 1024), (".enc", 1024), (".emb", 2304)):
    a = np.fromfile(ref + ext, dtype=np.float32).reshape(-1, cols).astype(np.float64)
    b = np.fromfile(sq + ext, dtype=np.float32).reshape(-1, cols).astype(np.float64)
    if a.shape != b.shape:
        print(ext, "shape mismatch", a.shape, b.shape); continue
    rel = np.linalg.norm(a - b) / np.linalg.norm(a)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print(f"{ext:7s} rows={a.shape[0]:5d} rel_l2={rel:.2e} max_abs={np.abs(a-b).max():.3e} (|ref|max {np.abs(a).max():.2f}) min_row_cos={cos.min():.6f}")
