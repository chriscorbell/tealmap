//! Voxel LOD pyramid ("block mipmaps").
//!
//! Level k uses voxels of 2^k blocks. A LOD tile is 64×64 voxel columns, so a
//! level-k tile covers 64·2^k blocks: level 1 covers 2×2 hires tiles, and each
//! level above covers 2×2 tiles of the one below.
//!
//! Each voxel holds a real block state, picked from the 2×2×2 voxels (or
//! blocks) under it, so a LOD tile is meshed by the hires mesher at a larger
//! scale: textures, tints, models, leaves and water all match hires.
//!
//! Every tile also leaves its own downsampled grid (a "source") for its parent:
//! hires tiles write level 1 sources, level 1 tiles write level 2 sources, and
//! so on. A change only rebuilds the tile's ancestors.

use crate::mesh::{Settings, TILE, Volume};
use crate::model::{AIR, BlockInfo, Registry};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const LEVELS: u32 = 4;
/// Minimum solid children (of 8) for a voxel to be solid, per level. A low
/// bar keeps thin things (trunks, spikes, walls) from vanishing as levels
/// compound; the screen-space test keeps the slight fattening to a few pixels.
const MIN_SOLID: [u8; LEVELS as usize + 1] = [0, 2, 2, 2, 2];
/// Columns per LOD tile side.
pub const N: usize = TILE as usize;

/// A dense grid of voxel columns: `states[column * h + y]`, with column =
/// z * w + x. `y0` is the voxel y of index 0.
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub y0: i32,
    pub states: Vec<u16>,
    pub biomes: Vec<u16>,
}

impl Grid {
    pub fn empty(w: usize, h: usize, y0: i32) -> Self {
        Grid {
            w,
            h,
            y0,
            states: vec![AIR; w * w * h],
            biomes: vec![0; w * w],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.states.iter().all(|&s| s == AIR)
    }

    /// The lowest and highest non-air y index.
    pub fn y_range(&self) -> Option<(usize, usize)> {
        let (mut lo, mut hi) = (usize::MAX, 0);
        for (i, &s) in self.states.iter().enumerate() {
            if s != AIR {
                let y = i % self.h;
                lo = lo.min(y);
                hi = hi.max(y);
            }
        }
        (lo <= hi).then_some((lo, hi))
    }

    /// Copies `src` with its corner at column (ox, oz), clipped to this grid.
    fn place_at(&mut self, src: &Grid, ox: isize, oz: isize) {
        debug_assert_eq!(src.h, self.h);
        for z in 0..src.w {
            for x in 0..src.w {
                let (tx, tz) = (ox + x as isize, oz + z as isize);
                if tx < 0 || tz < 0 || tx >= self.w as isize || tz >= self.w as isize {
                    continue;
                }
                let to = tz as usize * self.w + tx as usize;
                let from = z * src.w + x;
                self.states[to * self.h..(to + 1) * self.h]
                    .copy_from_slice(&src.states[from * src.h..(from + 1) * src.h]);
                self.biomes[to] = src.biomes[from];
            }
        }
    }
}

/// The world's vertical range padded to a multiple of the coarsest voxel, so
/// every level divides it exactly.
pub fn bounds(min_y: i32, max_y: i32) -> (i32, i32) {
    let step = 1 << LEVELS;
    (
        min_y.div_euclid(step) * step,
        (max_y + step - 1).div_euclid(step) * step,
    )
}

/// The level-k tiles whose mesh reads level-k source (cx, cz): its parent,
/// plus the neighbors whose one-voxel border ring reaches into it.
pub fn tiles_using_source(cx: i32, cz: i32) -> impl Iterator<Item = (i32, i32)> {
    let pair = |c: i32| {
        let core = c.div_euclid(2);
        [
            core,
            if c.rem_euclid(2) == 0 {
                core - 1
            } else {
                core + 1
            },
        ]
    };
    let zs = pair(cz);
    pair(cx)
        .into_iter()
        .flat_map(move |x| zs.map(move |z| (x, z)))
}

/// The block-level grid of a hires tile's own 64×64 columns, with removed
/// caves turned to air. Blocks outside the volume's range are air.
pub fn from_volume(v: &Volume, reg: &Registry, min_y: i32, max_y: i32) -> Grid {
    let (min_y, max_y) = bounds(min_y, max_y);
    let h = (max_y - min_y) as usize;
    let mut g = Grid::empty(N, h, min_y);
    let infos = reg.snapshot();
    // Bottom up, so each column ends with the biome of its top block.
    for vy in 1..v.h.saturating_sub(1) {
        let wy = v.y0 + vy as i32;
        if wy < min_y || wy >= max_y {
            continue;
        }
        let y = (wy - min_y) as usize;
        for z in 0..N {
            for x in 0..N {
                let (vx, vz) = (x + 1, z + 1);
                let id = v.block(vx, vy, vz);
                if id == AIR || !infos[id as usize].lod_counts || v.cave(vx, vy, vz) {
                    continue;
                }
                let col = z * N + x;
                g.states[col * h + y] = id;
                g.biomes[col] = v.biome(vx, vy, vz);
            }
        }
    }
    g
}

/// Halves a grid in every direction. A voxel is solid when at least
/// `min_solid` of its 8 children are, and takes the state that's most common
/// among them, counting children on the surface (open above) three times so
/// the top of the terrain wins over what's under it.
pub fn downsample(src: &Grid, reg: &Registry, min_solid: u8) -> Grid {
    let (w, h) = (src.w / 2, src.h.div_ceil(2));
    let mut out = Grid::empty(w, h, src.y0.div_euclid(2));
    let infos = reg.snapshot();
    let mut votes: Vec<(u16, u32)> = Vec::with_capacity(8);
    for z in 0..w {
        for x in 0..w {
            let col = z * w + x;
            let columns = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dz)| {
                let c = (z * 2 + dz) * src.w + x * 2 + dx;
                &src.states[c * src.h..(c + 1) * src.h]
            });
            for y in 0..h {
                let (cy0, cy1) = (y * 2, y * 2 + 1);
                // Most voxels are empty; settle those without voting.
                let bottom = columns.map(|c| c[cy0]);
                let top = columns.map(|c| c.get(cy1).copied().unwrap_or(AIR));
                if bottom == [AIR; 4] && top == [AIR; 4] {
                    continue;
                }
                votes.clear();
                let mut solid = 0;
                for (dx, dz, dy) in CHILDREN {
                    let (column, cy) = (columns[dz * 2 + dx], y * 2 + dy);
                    let s = column.get(cy).copied().unwrap_or(AIR);
                    let info = infos[s as usize];
                    if s == AIR || !info.lod_counts {
                        continue;
                    }
                    solid += 1;
                    // Votes count the space a block fills, doubled on the surface.
                    let above = column.get(cy + 1).copied().unwrap_or(AIR);
                    let surface = if above == AIR { 2 } else { 1 };
                    let weight = info.lod_volume as u32 * surface;
                    match votes.iter_mut().find(|v| v.0 == s) {
                        Some(v) => v.1 += weight,
                        None => votes.push((s, weight)),
                    }
                }
                if solid >= min_solid {
                    out.states[col * h + y] = pick(&votes, &infos, reg);
                }
            }
            // The biome of the child column with the most solid voxels.
            let kids =
                [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dz)| (z * 2 + dz) * src.w + x * 2 + dx);
            let best = kids
                .iter()
                .max_by_key(|&&c| {
                    let n = src.states[c * src.h..(c + 1) * src.h]
                        .iter()
                        .filter(|&&s| s != AIR)
                        .count();
                    (n, std::cmp::Reverse(c))
                })
                .unwrap();
            out.biomes[col] = src.biomes[*best];
        }
    }
    out
}

const CHILDREN: [(usize, usize, usize); 8] = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
];

/// Highest vote wins; ties go to the smaller registry key, which keeps output
/// independent of the order states were first seen in. A slab is drawn as a
/// double slab, since a voxel stands for the whole space.
fn pick(votes: &[(u16, u32)], infos: &[&BlockInfo], reg: &Registry) -> u16 {
    let mut best = votes[0];
    for &v in &votes[1..] {
        if v.1 > best.1 || (v.1 == best.1 && infos[v.0 as usize].key < infos[best.0 as usize].key) {
            best = v;
        }
    }
    let info = infos[best.0 as usize];
    if info.occludes == 0x3f {
        return best.0;
    }
    *info.lod_full_form.get_or_init(|| full_form(best.0, reg))
}

fn full_form(id: u16, reg: &Registry) -> u16 {
    let b = reg.get(id);
    let key: String = b
        .key
        .split('|')
        .map(|part| match part {
            "type=bottom" | "type=top" => "type=double",
            p => p,
        })
        .collect::<Vec<_>>()
        .join("|");
    if key == b.key {
        return id;
    }
    let full = reg.id_for_key(&key);
    if reg.get(full).occludes == 0x3f {
        full
    } else {
        id
    }
}

/// Paths for level-k tile files under a map's output directory.
pub fn tile_path(out: &Path, level: u32, x: i32, z: i32) -> PathBuf {
    out.join("lod")
        .join(level.to_string())
        .join(format!("{x}_{z}.tmh"))
}

/// The source a level-(k-1) tile (or a hires tile, for k = 1) leaves for level k.
pub fn source_path(out: &Path, level: u32, x: i32, z: i32) -> PathBuf {
    out.join("lod")
        .join(format!("src{level}"))
        .join(format!("{x}_{z}.bin"))
}

const SRC_MAGIC: &[u8; 4] = b"TLS1";

/// Saves a source grid as column runs, with state keys and biome names in
/// per-file tables, so it can be read back by another process.
pub fn save_source(path: &Path, g: &Grid, reg: &Registry) -> Result<()> {
    let mut keys: Vec<&str> = Vec::new();
    let mut key_ids: HashMap<u16, u16> = HashMap::new();
    let mut names: Vec<&str> = Vec::new();
    let mut name_ids: HashMap<u16, u16> = HashMap::new();
    let mut body = Vec::new();
    for col in 0..g.w * g.w {
        let b = *name_ids.entry(g.biomes[col]).or_insert_with(|| {
            names.push(reg.baker.biomes.name(g.biomes[col]));
            (names.len() - 1) as u16
        });
        body.extend_from_slice(&b.to_le_bytes());
        let column = &g.states[col * g.h..(col + 1) * g.h];
        let mut runs = Vec::new();
        let mut y = 0;
        while y < g.h {
            let s = column[y];
            let start = y;
            while y < g.h && column[y] == s {
                y += 1;
            }
            if s != AIR {
                let k = *key_ids.entry(s).or_insert_with(|| {
                    keys.push(&reg.get(s).key);
                    (keys.len() - 1) as u16
                });
                runs.push([start as u16, (y - start) as u16, k]);
            }
        }
        body.extend_from_slice(&(runs.len() as u16).to_le_bytes());
        for r in runs {
            for v in r {
                body.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let mut out = Vec::with_capacity(body.len() + 1024);
    out.extend_from_slice(SRC_MAGIC);
    for v in [
        g.w as u32,
        g.h as u32,
        g.y0 as u32,
        keys.len() as u32,
        names.len() as u32,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for s in keys.iter().chain(names.iter()) {
        out.extend_from_slice(&(s.len() as u16).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    out.extend_from_slice(&body);
    let mut gz = Vec::new();
    crate::render::gzip_into(&out, &mut gz);
    crate::render::write_atomic(path, &gz)
}

pub fn load_source(path: &Path, reg: &Registry) -> Result<Option<Grid>> {
    let raw = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let mut b = Vec::new();
    std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&raw[..]), &mut b)?;
    let mut o = 0;
    let u16_ = |o: &mut usize| -> Result<u16> {
        let v = b
            .get(*o..*o + 2)
            .ok_or_else(|| anyhow::anyhow!("truncated LOD source"))?;
        *o += 2;
        Ok(u16::from_le_bytes([v[0], v[1]]))
    };
    if b.len() < 24 || b.get(..4) != Some(SRC_MAGIC) {
        bail!("bad LOD source {}", path.display());
    }
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let (w, h, y0, nk, nn) = (
        u32_at(4) as usize,
        u32_at(8) as usize,
        u32_at(12) as i32,
        u32_at(16) as usize,
        u32_at(20) as usize,
    );
    o += 24;
    let mut strings = Vec::with_capacity(nk + nn);
    for _ in 0..nk + nn {
        let len = u16_(&mut o)? as usize;
        let s = std::str::from_utf8(
            b.get(o..o + len)
                .ok_or_else(|| anyhow::anyhow!("truncated LOD source"))?,
        )?;
        strings.push(s.to_owned());
        o += len;
    }
    let states: Vec<u16> = strings[..nk].iter().map(|k| reg.id_for_key(k)).collect();
    let biomes: Vec<u16> = strings[nk..]
        .iter()
        .map(|n| reg.baker.biomes.id(n))
        .collect();
    // Sizes come from the file; a corrupt one is an error, never a panic.
    if w > 4 * N || h > 4096 {
        bail!("bad LOD source size in {}", path.display());
    }
    let bad = || anyhow::anyhow!("corrupt LOD source {}", path.display());
    let mut g = Grid::empty(w, h, y0);
    for col in 0..w * w {
        g.biomes[col] = *biomes.get(u16_(&mut o)? as usize).ok_or_else(bad)?;
        for _ in 0..u16_(&mut o)? {
            let (start, len, k) = (
                u16_(&mut o)? as usize,
                u16_(&mut o)? as usize,
                u16_(&mut o)? as usize,
            );
            let state = *states.get(k).ok_or_else(bad)?;
            if start + len > h {
                return Err(bad());
            }
            g.states[col * h + start..col * h + start + len].fill(state);
        }
    }
    Ok(Some(g))
}

#[allow(clippy::too_many_arguments)]
/// Builds level-k tile (x, z) from its four children's sources. Returns the
/// encoded tile (if it has geometry) and the source it leaves for level k+1.
pub fn build_tile(
    out: &Path,
    reg: &Registry,
    level: u32,
    x: i32,
    z: i32,
    min_y: i32,
    max_y: i32,
    volume: &mut Volume,
    mesher: &mut crate::mesh::Mesher,
) -> Result<(Option<Vec<u8>>, Option<Grid>)> {
    let s = 1i32 << level;
    let (min_y, max_y) = bounds(min_y, max_y);
    let h = ((max_y - min_y) / s) as usize;
    let half = (N / 2) as isize;
    let load = |cx: i32, cz: i32| -> Result<Option<Grid>> {
        let src = load_source(&source_path(out, level, cx, cz), reg)?;
        if src.as_ref().is_some_and(|g| g.w != N / 2 || g.h != h) {
            bail!("LOD source for level {level} has the wrong size");
        }
        Ok(src)
    };
    // The tile's own four quadrants.
    let mut core = Grid::empty(N, h, min_y / s);
    for (qx, qz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        if let Some(src) = load(x * 2 + qx, z * 2 + qz)? {
            core.place_at(&src, qx as isize * half, qz as isize * half);
        }
    }
    if core.is_empty() {
        return Ok((None, None));
    }
    // A one-voxel ring from the neighbors, so border faces cull against what's
    // really there. Neighbors drawn at another level are covered by skirts.
    let mut grid = Grid::empty(N + 2, h, min_y / s);
    grid.place_at(&core, 1, 1);
    for i in -1..=2 {
        for j in -1..=2 {
            if (0..=1).contains(&i) && (0..=1).contains(&j) {
                continue;
            }
            if let Some(src) = load(x * 2 + i, z * 2 + j)? {
                grid.place_at(&src, 1 + i as isize * half, 1 + j as isize * half);
            }
        }
    }
    let settings = Settings {
        min_y: min_y / s,
        remove_caves_below_y: i32::MIN,
        lod: true,
    };
    let tile = if volume.fill_grid(&grid, x * N as i32, z * N as i32, reg, s) {
        mesher.mesh(volume, reg, &settings)
    } else {
        None
    };
    let parent = (level < LEVELS).then(|| downsample(&core, reg, MIN_SOLID[level as usize + 1]));
    Ok((tile, parent.filter(|g| !g.is_empty())))
}

/// The level 1 source a hires tile leaves: its blocks, downsampled once.
pub fn hires_source(v: &Volume, reg: &Registry, min_y: i32, max_y: i32) -> Option<Grid> {
    let g = downsample(&from_volume(v, reg, min_y, max_y), reg, MIN_SOLID[1]);
    (!g.is_empty()).then_some(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::AssetFiles;
    use crate::model::Baker;

    #[test]
    fn downsample_keeps_surface_and_majority() {
        let reg = Registry::new(Baker::new(AssetFiles::new()));
        let stone = reg.id("minecraft:stone", &mut []);
        let grass = reg.id("minecraft:grass_block", &mut []);
        // 2×2 columns, 4 high: stone at y 0–2 with grass on top at y 3 in
        // three columns; the fourth column is only stone at y 0.
        let mut g = Grid::empty(2, 4, 0);
        for c in 0..3 {
            g.states[c * 4..c * 4 + 3].fill(stone);
            g.states[c * 4 + 3] = grass;
        }
        g.states[3 * 4] = stone;
        let d = downsample(&g, &reg, 4);
        assert_eq!((d.w, d.h), (1, 2));
        // Bottom voxel: 7 stone of 8. Top voxel: 3 stone + 3 grass of 8, and
        // grass is on the surface, so it wins.
        assert_eq!(d.states, vec![stone, grass]);
        // With a higher bar, the top voxel (6 of 8) still stays; 7 of 8 drops it.
        assert_eq!(downsample(&g, &reg, 7).states, vec![stone, AIR]);
    }

    #[test]
    fn bounds_align_to_the_coarsest_voxel() {
        assert_eq!(bounds(-64, 320), (-64, 320));
        assert_eq!(bounds(-64, 319), (-64, 320));
        assert_eq!(bounds(-60, 300), (-64, 304));
    }

    #[test]
    fn sources_feed_their_parent_and_bordering_tiles() {
        let mut t: Vec<_> = tiles_using_source(0, 0).collect();
        t.sort();
        assert_eq!(t, vec![(-1, -1), (-1, 0), (0, -1), (0, 0)]);
        let mut t: Vec<_> = tiles_using_source(3, -1).collect();
        t.sort();
        assert_eq!(t, vec![(1, -1), (1, 0), (2, -1), (2, 0)]);
    }

    #[test]
    fn sources_round_trip() {
        let reg = Registry::new(Baker::new(AssetFiles::new()));
        let stone = reg.id("minecraft:stone", &mut []);
        let mut props = vec![("axis".to_owned(), "y".to_owned())];
        let log = reg.id("minecraft:oak_log", &mut props);
        let mut g = Grid::empty(2, 8, -4);
        g.states[2..5].fill(stone);
        g.states[8 + 6] = log;
        g.biomes = vec![reg.baker.biomes.id("minecraft:plains"), 0, 0, 0];
        let dir = std::env::temp_dir().join(format!("tealmap-lod-{}", std::process::id()));
        let path = dir.join("src.bin");
        save_source(&path, &g, &reg).unwrap();
        // A fresh registry gets different ids; keys still map back.
        let reg2 = Registry::new(Baker::new(AssetFiles::new()));
        reg2.id("minecraft:dirt", &mut []);
        let back = load_source(&path, &reg2).unwrap().unwrap();
        let keys = |g: &Grid, r: &Registry| {
            g.states
                .iter()
                .map(|&s| r.get(s).key.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(&back, &reg2), keys(&g, &reg));
        assert_eq!((back.w, back.h, back.y0), (2, 8, -4));
        let _ = std::fs::remove_dir_all(dir);
    }
}
