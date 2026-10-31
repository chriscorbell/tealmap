#!/usr/bin/env python3
"""Quad statistics of a rendered map: counts per level, group and face direction,
heights, table sizes and how many quads greedy merging could remove.

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

def full_face(t):
    xs, ys, zs = t[0::3][:4], t[1::3][:4], t[2::3][:4]
    for c in (xs, ys, zs):
        if not (min(c) in (64, 128) and max(c) in (64, 128)): return False
    return True

def read(path):
    with open(path, 'rb') as f: raw = f.read()
    b = gzip.decompress(raw)
    assert b[:4] == b'TMH2'
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

def greedy(quads, templates, colors, ignore_rotation):
    """Greedy-merges full cube faces in each plane. Returns merged quad count."""
    planes = collections.defaultdict(dict)
    rest = 0
    for (g, x, y, z, ti, ci) in quads:
        t = templates[ti]
        d = direction(t)
        if d == 6 or not full_face(t):
            rest += 1
            continue
        c = colors[ci] & 0xffffff
        layer = t[20]
        key = (layer, c, g) if ignore_rotation else (layer, c, g, t[12:20])
        if d < 2: p, u, v = y, x, z
        elif d < 4: p, u, v = z, x, y
        else: p, u, v = x, z, y
        planes[(d, p)][(u, v)] = key
    merged = rest
    for cells in planes.values():
        seen = set()
        for (u, v) in sorted(cells):
            if (u, v) in seen: continue
            k = cells[(u, v)]
            w = 1
            while (u+w, v) in cells and (u+w, v) not in seen and cells[(u+w, v)] == k: w += 1
            h = 1
            while all((u+i, v+h) in cells and (u+i, v+h) not in seen and cells[(u+i, v+h)] == k for i in range(w)): h += 1
            for i in range(w):
                for j in range(h): seen.add((u+i, v+j))
            merged += 1
    return merged

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('--sample', type=int, default=150)
ap.add_argument('--greedy', type=int, default=30)
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
    gt = tiles[:a.greedy]
    q = sum(len(t['quads']) for t in gt)
    if q:
        g1 = sum(greedy(t['quads'], t['templates'], t['colors'], False) for t in gt)
        g2 = sum(greedy(t['quads'], t['templates'], t['colors'], True) for t in gt)
        print(f"  greedy over {len(gt)} tiles: exact template {g1/q*100:.0f}% of quads remain; ignoring rotation {g2/q*100:.0f}% remain")
