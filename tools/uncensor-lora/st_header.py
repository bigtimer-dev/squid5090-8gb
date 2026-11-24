import json, struct, sys
def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n
if __name__ == '__main__':
    h, base = header(sys.argv[1])
    pats = sys.argv[2:]
    for k, v in h.items():
        if k == '__metadata__': continue
        if any(p in k for p in pats):
            print(k, v['dtype'], v['shape'], v['data_offsets'][1] - v['data_offsets'][0])
