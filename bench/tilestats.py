#!/usr/bin/env python3
"""Quad statistics of a rendered map: counts per level, group and face direction,
heights and table sizes.

    bench/tilestats.py <map dir> [--sample N]
"""
import gzip, os, random, struct, sys, collections, argparse

W = 66
GROUPS = ['opaque', 'cutout', 'leafint', 'transl']
DIRS = ['down', 'up', 'north', 'south', 'west', 'east', 'other']

def direction(t):
    # Corners in 1/64 block +64. Axis-aligned when one coordinate is constant.
    xs, ys, zs = t[0::3][:4], t[1::3][:4], t[2::3][:4]
    ax, ay, az = t[3]-t[0], t[4]-t[1], t[5]-t[2]
    bx, by, bz = t[6]-t[0], t[7]-t[1], t[8]-t[2]
    nx, ny, nz = ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx
    if ny and not nx and not nz: return 0 if ny < 0 else 1
    if nz and not nx and not ny: return 2 if nz < 0 else 3
    if nx and not ny and not nz: return 4 if nx < 0 else 5
    return 6

def read(path):
    with open(path, 'rb') as f: raw = f.read()
    b = gzip.decompress(raw)
    if b[:4] == b'TMH2': return read2(raw, b)
    assert b[:4] == b'TMH3'
    ox, oy, oz = struct.unpack_from('<iii', b, 4)
    streams = struct.unpack_from('<28I', b, 20)
    nt, np_ = struct.unpack_from('<II', b, 132)
    o = 140
    field = lambda k, i: b[o+k*2*nt+i] | b[o+k*2*nt+nt+i] << 8
    templates = [tuple(field(k, i) for k in range(22)) for i in range(nt)]
    o += nt*44
    colors = [b[o+i] | b[o+np_+i] << 8 | b[o+2*np_+i] << 16 for i in range(np_)]
    o += np_*3
    quads = []
    counts = [0, 0, 0, 0]
    for s_, n in enumerate(streams):
        g, d = s_ // 7, s_ % 7
        counts[g] += n
        pl, vl, ul, wl, hl, tl, cl = o, o+2*n, o+4*n, o+5*n, o+6*n, o+7*n, o+9*n
        o += 11*n
        p = v = u = 0
        for i in range(n):
            dp = b[pl+i] | b[pl+n+i] << 8
            dv = b[vl+i] | b[vl+n+i] << 8
            du = b[ul+i]
            new_plane = dp != 0 or i == 0
            p += dp
            v = dv if new_plane else v + dv
            u = du if (new_plane or dv != 0) else u + du
            if d in (2, 3): x, y, z = u, v, p
            elif d in (4, 5): x, y, z = p, v, u
            else: x, y, z = u, p, v
            quads.append((g, x, y, z, b[tl+i] | b[tl+n+i] << 8, b[cl+i] | b[cl+n+i] << 8))
    return dict(raw=len(raw), size=len(b), counts=counts, nt=nt, np=np_, templates=templates, colors=colors, quads=quads, origin=(ox, oy, oz))

def read2(raw, b):
    ox, oy, oz = struct.unpack_from('<iii', b, 4)
    counts = struct.unpack_from('<4I', b, 20)
    nt, np_ = struct.unpack_from('<II', b, 36)
    o = 44
    templates = [struct.unpack_from('<21H', b, o + i*42) for i in range(nt)]
    o += nt*42
    colors = struct.unpack_from(f'<{np_}I', b, o)
    o += np_*4
    quads = []
    for g, n in enumerate(counts):
        cl, ch = o, o+W*W
        o += 2*W*W
        sl, sh, tl, th, pl, ph = o, o+n, o+2*n, o+3*n, o+4*n, o+5*n
        o += 6*n
        q = 0
        for z in range(W):
            for x in range(W):
                c = b[cl+z*W+x] | b[ch+z*W+x] << 8
                y = 1023
                for _ in range(c):
                    y -= b[sl+q] | b[sh+q] << 8
                    quads.append((g, x, y, z, b[tl+q] | b[th+q] << 8, b[pl+q] | b[ph+q] << 8))
                    q += 1
    return dict(raw=len(raw), size=len(b), counts=counts, nt=nt, np=np_, templates=templates, colors=colors, quads=quads, origin=(ox, oy, oz))

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('--sample', type=int, default=150)
a = ap.parse_args()
random.seed(7)
levels = {'0': []}
for r, _, fs in os.walk(os.path.join(a.dir, 'hires')):
    levels['0'] += [os.path.join(r, f) for f in fs if f.endswith('.tmh')]
for l in os.listdir(os.path.join(a.dir, 'lod')):
    if l.startswith('src'): continue
    levels[l] = [os.path.join(a.dir, 'lod', l, f) for f in os.listdir(os.path.join(a.dir, 'lod', l)) if f.endswith('.tmh')]

for l in sorted(levels):
    files = levels[l]
    sample = random.sample(files, min(a.sample, len(files)))
    tiles = [read(f) for f in sample]
    n = len(tiles) or 1
    total = sum(sum(t['counts']) for t in tiles)
    print(f"\nlevel {l}: {len(files)} tiles, sampled {len(tiles)}")
    print(f"  quads/tile {total/n:.0f}  gz/tile {sum(t['raw'] for t in tiles)/n/1e3:.1f} KB  raw/tile {sum(t['size'] for t in tiles)/n/1e3:.1f} KB  gz bytes/quad {sum(t['raw'] for t in tiles)/max(total,1):.2f}")
    print(f"  templates/tile {sum(t['nt'] for t in tiles)/n:.0f} ({sum(t['nt'] for t in tiles)/n*42/1e3:.1f} KB raw)  colors/tile {sum(t['np'] for t in tiles)/n:.0f}  distinct rgb/tile {sum(len({c & 0xffffff for c in t['colors']}) for t in tiles)/n:.0f}")
    by_g = collections.Counter(); by_d = collections.Counter(); by_gd = collections.Counter(); ys = collections.Counter()
    for t in tiles:
        for (g, x, y, z, ti, ci) in t['quads']:
            d = direction(t['templates'][ti])
            by_g[g] += 1; by_d[d] += 1; by_gd[(g, d)] += 1
            ys[(y + t['origin'][1]) * (1 << int(l)) // 32 * 32] += 1
    print('  by group: ' + '  '.join(f"{GROUPS[g]} {by_g[g]/max(total,1)*100:.1f}%" for g in range(4)))
    print('  by dir:   ' + '  '.join(f"{DIRS[d]} {by_d[d]/max(total,1)*100:.1f}%" for d in range(7)))
    print('  by y:     ' + '  '.join(f"{y}:{c/max(total,1)*100:.0f}%" for y, c in sorted(ys.items())))
