#!/usr/bin/env python3
"""Writes a small made-up Nether: region/r.0.0.mca, chunks 0-3 x 0-3.

Like the real thing it has a bedrock floor and roof, no sky light saved, and
caves and lava between them. Nobody has played this world; there just wasn't a
real Nether to cut one from.
"""
import math
import os
import struct
import zlib


def s(t):
    b = t.encode()
    return struct.pack(">H", len(b)) + b


def tag(kind, name, payload):
    return bytes([kind]) + s(name) + payload


def compound(*entries):
    return b"".join(entries) + b"\0"


def blocks_at(x, y, z):
    if y == 0 or y >= 127:
        return "bedrock"
    # Caves: where two slow waves both run high.
    a = math.sin(x / 9) + math.sin(z / 7) + math.sin(y / 6)
    b = math.cos(x / 11 + z / 13) + math.cos(y / 8)
    if a + b > 1.2:
        return "lava" if y < 32 else "air"
    return "soul_sand" if (x // 5 + z // 5) % 7 == 0 else "netherrack"


def section(cx, cz, sy):
    states = [
        blocks_at(cx * 16 + x, sy * 16 + y, cz * 16 + z)
        for y in range(16)
        for z in range(16)
        for x in range(16)
    ]
    palette = sorted(set(states))
    index = {n: i for i, n in enumerate(palette)}
    entries = b"".join(compound(tag(8, "Name", s("minecraft:" + n))) for n in palette)
    block_states = [tag(9, "palette", bytes([10]) + struct.pack(">i", len(palette)) + entries)]
    if len(palette) > 1:
        bits = max(4, math.ceil(math.log2(len(palette))))
        per = 64 // bits
        longs = []
        for i in range(0, 4096, per):
            v = 0
            for j, n in enumerate(states[i : i + per]):
                v |= index[n] << (j * bits)
            longs.append(v)
        block_states.append(tag(12, "data", struct.pack(">i", len(longs)) + struct.pack(f">{len(longs)}Q", *longs)))
    return compound(tag(1, "Y", bytes([sy & 0xff])), tag(10, "block_states", compound(*block_states)))


def chunk(cx, cz):
    sections = [section(cx, cz, sy) for sy in range(8)]
    # 256 9-bit values, 7 to a long: the first free height over the roof, above y=0.
    longs = []
    for i in range(0, 256, 7):
        v = 0
        for j in range(len(range(i, min(i + 7, 256)))):
            v |= 128 << (j * 9)
        longs.append(v)
    return compound(
        tag(8, "Status", s("minecraft:full")),
        tag(1, "isLightOn", b"\x01"),
        tag(3, "xPos", struct.pack(">i", cx)),
        tag(3, "zPos", struct.pack(">i", cz)),
        tag(3, "yPos", struct.pack(">i", 0)),
        tag(10, "Heightmaps", compound(tag(12, "OCEAN_FLOOR", struct.pack(">i", len(longs)) + struct.pack(f">{len(longs)}Q", *longs)))),
        tag(9, "sections", bytes([10]) + struct.pack(">i", len(sections)) + b"".join(sections)),
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    locations = bytearray(4096)
    body = bytearray()
    sector = 2
    for cz in range(4):
        for cx in range(4):
            data = zlib.compress(b"\x0a\0\0" + chunk(cx, cz), 9)
            framed = struct.pack(">IB", len(data) + 1, 2) + data
            framed += b"\0" * (-len(framed) % 4096)
            count = len(framed) // 4096
            struct.pack_into(">I", locations, (cx + cz * 32) * 4, (sector << 8) | count)
            body += framed
            sector += count
    with open(os.path.join(here, "region", "r.0.0.mca"), "wb") as f:
        f.write(bytes(locations) + bytes(4096) + bytes(body))


main()
