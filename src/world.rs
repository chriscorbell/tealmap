//! Chunk decoding: palettes expanded to global block ids, plus light,
//! biomes and the ocean-floor heightmap.

use crate::model::{AIR, Registry};
use crate::nbt::Reader;
use anyhow::Result;
use rustc_hash::FxHashMap;

pub struct Section {
    pub y: i32,
    /// 4096 block ids in YZX order, or None if the section is a single state.
    pub blocks: Option<Box<[u16; 4096]>>,
    pub single: u16,
    /// 64 biome ids in 4×4×4 YZX order.
    pub biomes: [u16; 64],
    /// `sky << 4 | block light` per block in YZX order, or None if it's all
    /// `uniform_light`.
    pub light: Option<Box<[u8; 4096]>>,
    pub uniform_light: u8,
    /// Whether any block in it holds water.
    pub has_water: bool,
}

impl Section {
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

/// Remembers what palette entries resolved to, by their raw NBT bytes, so most
/// entries cost a hash instead of building a state key.
#[derive(Default)]
pub struct DecodeCache {
    states: FxHashMap<Box<[u8]>, u16>,
    biomes: FxHashMap<Box<[u8]>, u16>,
    ids: Vec<u16>,
    props: Vec<(String, String)>,
}

/// A section as read from NBT, before its light is resolved.
struct RawSection<'a> {
    y: i32,
    blocks: Option<Box<[u16; 4096]>>,
    single: u16,
    has_water: bool,
    biomes: [u16; 64],
    sky: Option<&'a [u8]>,
    block_light: Option<&'a [u8]>,
}

/// Decodes chunk NBT. Returns None for chunks that aren't fully generated.
pub fn decode_chunk(
    data: &[u8],
    reg: &Registry,
    min_y: i32,
    cache: &mut DecodeCache,
) -> Result<Option<Chunk>> {
    let mut r = Reader::new(data);
    r.enter_root()?;
    let (mut full, mut legacy, mut has_light) = (false, false, false);
    let mut sections: Vec<RawSection> = Vec::new();
    let mut ocean_floor = None;
    while let Some((tag, name)) = r.next_entry()? {
        match name {
            b"sections" if tag == 9 => {
                let (elem, n) = r.list_header()?;
                if elem == 10 {
                    sections.reserve(n.min(64));
                    for _ in 0..n {
                        if let Some(s) = read_section(&mut r, reg, cache)? {
                            sections.push(s);
                        }
                    }
                } else {
                    for _ in 0..n {
                        r.skip_payload(elem)?;
                    }
                }
            }
            b"Status" => {
                let v = r.value(tag)?;
                full = matches!(v.as_str(), Some("minecraft:full" | "full"));
            }
            b"isLightOn" => has_light = r.value(tag)?.as_i64().unwrap_or(0) != 0,
            // Pre-1.18 chunks nest everything under "Level"; they aren't supported.
            b"Level" => {
                legacy = true;
                r.skip_payload(tag)?;
            }
            b"Heightmaps" if tag == 10 => {
                while let Some((t, k)) = r.next_entry()? {
                    if k == b"OCEAN_FLOOR" && t == 12 {
                        let longs = r.array(t)?;
                        if !longs.is_empty() {
                            let mut out = Box::new([min_y; 256]);
                            unpack(longs, 9, 256, |i, v| out[i] = min_y + v as i32);
                            ocean_floor = Some(out);
                        }
                    } else {
                        r.skip_payload(t)?;
                    }
                }
            }
            _ => r.skip_payload(tag)?,
        }
    }
    if legacy || !full {
        return Ok(None);
    }
    sections.sort_by_key(|s| s.y);

    // Sky light for sections without saved data comes from the bottom row of
    // the nearest section above that has it (15 if none), matching how the
    // game reads missing sky light. So light resolves from the top down.
    let mut carry = [15u8; 256];
    let mut out: Vec<Section> = Vec::with_capacity(sections.len());
    for s in sections.into_iter().rev() {
        let (light, uniform_light) = resolve_light(s.sky, s.block_light, &carry);
        if let Some(a) = s.sky {
            for (i, v) in carry.iter_mut().enumerate() {
                *v = (a[i >> 1] >> ((i & 1) * 4)) & 0xf;
            }
        }
        out.push(Section {
            y: s.y,
            blocks: s.blocks,
            single: s.single,
            biomes: s.biomes,
            light,
            uniform_light,
            has_water: s.has_water,
        });
    }
    out.reverse();

    Ok(Some(Chunk {
        sections: out,
        has_light,
        ocean_floor,
    }))
}

/// Merges a section's nibble arrays into one `sky << 4 | block` byte per block.
fn resolve_light(
    sky: Option<&[u8]>,
    block: Option<&[u8]>,
    carry: &[u8; 256],
) -> (Option<Box<[u8; 4096]>>, u8) {
    if sky.is_none() && block.is_none() {
        if carry.iter().all(|&v| v == carry[0]) {
            return (None, carry[0] << 4);
        }
        let mut out = Box::new([0u8; 4096]);
        for layer in out.chunks_exact_mut(256) {
            for (o, &c) in layer.iter_mut().zip(carry) {
                *o = c << 4;
            }
        }
        return (Some(out), 0);
    }
    let mut out = Box::new([0u8; 4096]);
    match sky {
        Some(a) => {
            for (o, &v) in out.chunks_exact_mut(2).zip(a) {
                o[0] = v << 4;
                o[1] = v & 0xf0;
            }
        }
        None => {
            for layer in out.chunks_exact_mut(256) {
                for (o, &c) in layer.iter_mut().zip(carry) {
                    *o = c << 4;
                }
            }
        }
    }
    if let Some(b) = block {
        for (o, &v) in out.chunks_exact_mut(2).zip(b) {
            o[0] |= v & 0xf;
            o[1] |= v >> 4;
        }
    }
    (Some(out), 0)
}

fn read_section<'a>(
    r: &mut Reader<'a>,
    reg: &Registry,
    cache: &mut DecodeCache,
) -> Result<Option<RawSection<'a>>> {
    let mut y = None;
    let (mut blocks, mut single, mut has_water) = (None, AIR, false);
    let mut biomes = [0u16; 64];
    let (mut sky, mut block_light) = (None, None);
    while let Some((tag, name)) = r.next_entry()? {
        match name {
            b"Y" => y = r.value(tag)?.as_i64(),
            b"block_states" if tag == 10 => {
                (blocks, single) = read_blocks(r, reg, cache)?;
                // What read_blocks left in the cache is this section's palette.
                has_water = cache.ids.iter().any(|&id| reg.get(id).has_water());
            }
            b"biomes" if tag == 10 => biomes = read_biomes(r, reg, cache)?,
            b"SkyLight" if tag == 7 => sky = Some(r.array(tag)?).filter(|a| a.len() == 2048),
            b"BlockLight" if tag == 7 => {
                block_light = Some(r.array(tag)?).filter(|a| a.len() == 2048)
            }
            _ => r.skip_payload(tag)?,
        }
    }
    Ok(y.map(|y| RawSection {
        y: y as i32,
        blocks,
        single,
        has_water,
        biomes,
        sky,
        block_light,
    }))
}

fn read_blocks(
    r: &mut Reader,
    reg: &Registry,
    cache: &mut DecodeCache,
) -> Result<(Option<Box<[u16; 4096]>>, u16)> {
    let mut data: &[u8] = &[];
    let mut has_palette = false;
    cache.ids.clear();
    while let Some((tag, name)) = r.next_entry()? {
        match name {
            b"palette" if tag == 9 => {
                has_palette = true;
                let (elem, n) = r.list_header()?;
                cache.ids.reserve(n.min(4096));
                for _ in 0..n {
                    let raw = r.raw(elem)?;
                    let id = match cache.states.get(raw) {
                        Some(&id) => id,
                        None => {
                            let id = resolve_state(elem, raw, reg, &mut cache.props)?;
                            cache.states.insert(raw.into(), id);
                            id
                        }
                    };
                    cache.ids.push(id);
                }
            }
            b"data" if tag == 12 => data = r.array(tag)?,
            _ => r.skip_payload(tag)?,
        }
    }
    let ids = &cache.ids;
    if !has_palette {
        return Ok((None, AIR));
    }
    if ids.len() <= 1 {
        return Ok((None, ids.first().copied().unwrap_or(AIR)));
    }
    if data.is_empty() {
        return Ok((None, ids[0]));
    }
    let bits = (usize::BITS - (ids.len() - 1).leading_zeros()).max(4) as usize;
    let mut out = Box::new([ids[0]; 4096]);
    unpack(data, bits, 4096, |i, v| {
        out[i] = ids.get(v as usize).copied().unwrap_or(AIR);
    });
    Ok((Some(out), ids[0]))
}

/// A palette entry's global id, from its raw NBT.
fn resolve_state(
    tag: u8,
    raw: &[u8],
    reg: &Registry,
    props: &mut Vec<(String, String)>,
) -> Result<u16> {
    let p = Reader::new(raw).value(tag)?;
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
    Ok(reg.id(name, props))
}

fn read_biomes(r: &mut Reader, reg: &Registry, cache: &mut DecodeCache) -> Result<[u16; 64]> {
    let mut data: &[u8] = &[];
    let mut has_palette = false;
    cache.ids.clear();
    while let Some((tag, name)) = r.next_entry()? {
        match name {
            b"palette" if tag == 9 => {
                has_palette = true;
                let (elem, n) = r.list_header()?;
                for _ in 0..n {
                    let raw = r.raw(elem)?;
                    let id = match cache.biomes.get(raw) {
                        Some(&id) => id,
                        None => {
                            let v = Reader::new(raw).value(elem)?;
                            let id = reg
                                .baker
                                .biomes
                                .id(v.as_str().unwrap_or("minecraft:plains"));
                            cache.biomes.insert(raw.into(), id);
                            id
                        }
                    };
                    cache.ids.push(id);
                }
            }
            b"data" if tag == 12 => data = r.array(tag)?,
            _ => r.skip_payload(tag)?,
        }
    }
    let ids = &cache.ids;
    if !has_palette {
        return Ok([0; 64]);
    }
    if ids.len() <= 1 {
        return Ok([ids.first().copied().unwrap_or(0); 64]);
    }
    let bits = (usize::BITS - (ids.len() - 1).leading_zeros()) as usize;
    let mut out = [ids[0]; 64];
    unpack(data, bits, 64, |i, v| {
        out[i] = ids.get(v as usize).copied().unwrap_or(ids[0])
    });
    Ok(out)
}

/// Unpacks `count` values of `bits` each from big-endian longs; values never
/// span two longs.
#[inline]
fn unpack(longs: &[u8], bits: usize, count: usize, mut f: impl FnMut(usize, u64)) {
    let per = 64 / bits;
    let mask = (1u64 << bits) - 1;
    let mut i = 0;
    for l in longs.chunks_exact(8) {
        let mut l = u64::from_be_bytes(l.try_into().unwrap());
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
        let raw: Vec<u8> = [l, 31].iter().flat_map(|v: &u64| v.to_be_bytes()).collect();
        unpack(&raw, 5, 13, |_, v| out.push(v));
        assert_eq!(out, (1..=12).chain([31]).collect::<Vec<_>>());
    }

    #[test]
    fn light_merges_nibbles_and_carries_sky() {
        let (mut sky, mut block) = ([0u8; 2048], [0u8; 2048]);
        sky[0] = 0xa5;
        block[0] = 0x3c;
        let (light, _) = resolve_light(Some(&sky), Some(&block), &[15; 256]);
        let light = light.unwrap();
        assert_eq!(light[0], 0x5c);
        assert_eq!(light[1], 0xa3);
        // No saved light: every layer takes the row carried from above.
        let mut carry = [15u8; 256];
        assert_eq!(resolve_light(None, None, &carry).1, 0xf0);
        carry[7] = 2;
        let (light, _) = resolve_light(None, None, &carry);
        let light = light.unwrap();
        assert_eq!(light[7], 0x20);
        assert_eq!(light[256 * 15 + 7], 0x20);
        assert_eq!(light[8], 0xf0);
    }
}
