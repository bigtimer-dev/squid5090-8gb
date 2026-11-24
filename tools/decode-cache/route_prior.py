#!/usr/bin/env python3
"""Routing prior for SQUID_DECODE_CACHE: [24][128] u32 little-endian counts of (layer, expert) uses over every decode
visit of the given SQUID_ROUTE_TRACE files (a loop layer counts once per visit, i.e. 3 x per token).
usage: route_prior.py OUT TRACE [TRACE ...]   (pure Python)"""
import struct, sys

VISITS = 48
SCHED = list(range(0, 6)) + list(range(6, 18)) * 3 + list(range(18, 24))
out, paths = sys.argv[1], sys.argv[2:]
c = [0] * (24 * 128)
ntok = 0
for p in paths:
    b = open(p, 'rb').read()
    n = struct.unpack_from('<I', b, 0)[0]
    flat = struct.unpack_from('<%di' % (n * VISITS * 8), b, 4)
    ntok += n
    for t in range(n):
        for v in range(VISITS):
            base = SCHED[v] * 128
            for e in flat[(t * VISITS + v) * 8:(t * VISITS + v) * 8 + 8]:
                c[base + e] += 1
open(out, 'wb').write(struct.pack('<%dI' % len(c), *c))
print(f'{out}: {len(paths)} traces, {ntok} tokens, {sum(c)} expert uses')
