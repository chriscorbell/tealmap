//! Chunk decoding: palettes expanded to global block ids, plus light,
//! biomes and the ocean-floor heightmap.

use crate::model::{AIR, Registry};
use crate::nbt::{self, Value};
use anyhow::Result;

pub struct Section {
    pub y: i32,
    /// 4096 block ids in YZX order, or None if the section is a single state.
    pub blocks: Option<Box<[u16; 4096]>>,
    pub single: u16,
    /// 64 biome ids in 4×4×4 YZX order.
    pub biomes: [u16; 64],
    pub sky: Option<Box<[u8; 2048]>>,
    pub block_light: Option<Box<[u8; 2048]>>,
}

impl Section {
    #[inline]
    pub fn block(&self, i: usize) -> u16 {
        match &self.blocks {
            Some(b) => b[i],
            None => self.single,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.blocks.is_none() && self.single == AIR
    }
}

pub struct Chunk {
    pub sections: Vec<Section>,
    pub has_light: bool,
    /// Ocean-floor height (first free y) per column in ZX order, absolute.
    pub ocean_floor: Option<Box<[i32; 256]>>,
}

fn keep(k: &[u8]) -> bool {
    matches!(
        k,
        b"sections" | b"Status" | b"isLightOn" | b"Heightmaps" | b"yPos" | b"Level"
    )
}

/// Decodes chunk NBT. Returns None for chunks that aren't fully generated.
pub fn decode_chunk(data: &[u8], reg: &Registry, min_y: i32) -> Result<Option<Chunk>> {
    let root = nbt::read_root_filtered(data, keep)?;
    // Pre-1.18 chunks nest everything under "Level"; they aren't supported.
    if root.get("Level").is_some() {
        return Ok(None);
    }
    let status = root.get("Status").and_then(|v| v.as_str()).unwrap_or("");
    if !matches!(status, "minecraft:full" | "full") {
        return Ok(None);
    }
    let has_light = root.get("isLightOn").and_then(|v| v.as_i64()).unwrap_or(0) != 0;

    let mut sections = Vec::new();
    let mut props = Vec::new();
    if let Some(list) = root.get("sections").and_then(|v| v.as_list()) {
        for s in list {
            let Some(y) = s.get("Y").and_then(|v| v.as_i64()) else {
                continue;
            };
            let y = y as i32;
            let (blocks, single) = match s.get("block_states") {
                Some(bs) => decode_blocks(bs, reg, &mut props),
                None => (None, AIR),
            };
            let biomes = match s.get("biomes") {
                Some(b) => decode_biomes(b, reg),
                None => [0; 64],
            };
            let nibbles = |key: &str| {
                s.get(key)
                    .and_then(|v| v.as_bytes())
                    .filter(|b| b.len() == 2048)
                    .map(|b| Box::new(<[u8; 2048]>::try_from(b).unwrap()))
            };
            sections.push(Section {
                y,
                blocks,
                single,
                biomes,
                sky: nibbles("SkyLight"),
                block_light: nibbles("BlockLight"),
            });
        }
    }
    sections.sort_by_key(|s| s.y);

    let ocean_floor = root
        .get("Heightmaps")
        .and_then(|h| h.get("OCEAN_FLOOR"))
        .map(|v| v.long_array())
        .filter(|l| !l.is_empty())
        .map(|longs| {
            let mut out = Box::new([min_y; 256]);
            unpack(&longs, 9, 256, |i, v| out[i] = min_y + v as i32);
            out
        });

    Ok(Some(Chunk {
        sections,
        has_light,
        ocean_floor,
    }))
}

fn decode_blocks(
    bs: &Value,
    reg: &Registry,
    props: &mut Vec<(String, String)>,
) -> (Option<Box<[u16; 4096]>>, u16) {
    let Some(palette) = bs.get("palette").and_then(|v| v.as_list()) else {
        return (None, AIR);
    };
    let ids: Vec<u16> = palette
        .iter()
        .map(|p| {
            let name = p
                .get("Name")
                .and_then(|v| v.as_str())
                .unwrap_or("minecraft:air");
            props.clear();
            if let Some(c) = p.get("Properties").and_then(|v| v.as_compound()) {
                for (k, v) in c {
                    if let Some(v) = v.as_str() {
                        props.push((String::from_utf8_lossy(k).into_owned(), v.to_owned()));
                    }
                }
            }
            reg.id(name, props)
        })
        .collect();
    if ids.len() <= 1 {
        return (None, ids.first().copied().unwrap_or(AIR));
    }
    let longs = bs.get("data").map(|v| v.long_array()).unwrap_or_default();
    if longs.is_empty() {
        return (None, ids[0]);
    }
    let bits = (usize::BITS - (ids.len() - 1).leading_zeros()).max(4) as usize;
    let mut out = Box::new([ids[0]; 4096]);
    unpack(&longs, bits, 4096, |i, v| {
        out[i] = ids.get(v as usize).copied().unwrap_or(AIR);
    });
    (Some(out), ids[0])
}

fn decode_biomes(b: &Value, reg: &Registry) -> [u16; 64] {
    let biomes = &reg.baker.biomes;
    let Some(palette) = b.get("palette").and_then(|v| v.as_list()) else {
        return [0; 64];
    };
    let ids: Vec<u16> = palette
        .iter()
        .map(|p| biomes.id(p.as_str().unwrap_or("minecraft:plains")))
        .collect();
    if ids.len() <= 1 {
        return [ids.first().copied().unwrap_or(0); 64];
    }
    let longs = b.get("data").map(|v| v.long_array()).unwrap_or_default();
    let bits = (usize::BITS - (ids.len() - 1).leading_zeros()) as usize;
    let mut out = [ids[0]; 64];
    unpack(&longs, bits, 64, |i, v| {
        out[i] = ids.get(v as usize).copied().unwrap_or(ids[0])
    });
    out
}

/// Unpacks `count` values of `bits` each; values never span two longs.
#[inline]
fn unpack(longs: &[u64], bits: usize, count: usize, mut f: impl FnMut(usize, u64)) {
    let per = 64 / bits;
    let mask = (1u64 << bits) - 1;
    let mut i = 0;
    for &l in longs {
        let mut l = l;
        for _ in 0..per {
            if i >= count {
                return;
            }
            f(i, l & mask);
            l >>= bits;
            i += 1;
        }
    }
}

#[inline]
pub fn nibble(arr: &[u8; 2048], i: usize) -> u8 {
    (arr[i >> 1] >> ((i & 1) * 4)) & 0xf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_no_span() {
        // 5 bits per value: 12 per long, the top 4 bits unused.
        let mut l = 0u64;
        for i in 0..12u64 {
            l |= (i + 1) << (i * 5);
        }
        let mut out = vec![];
        unpack(&[l, 31], 5, 13, |_, v| out.push(v));
        assert_eq!(out, (1..=12).chain([31]).collect::<Vec<_>>());
    }

    #[test]
    fn nibbles() {
        let mut a = [0u8; 2048];
        a[0] = 0xa5;
        assert_eq!(nibble(&a, 0), 5);
        assert_eq!(nibble(&a, 1), 10);
    }
}
