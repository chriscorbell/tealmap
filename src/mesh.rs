//! Hires tile meshing and the binary tile format.
//!
//! A tile covers 64×64 columns. Blocks are copied into a dense volume with a
//! one-block halo so faces on tile and chunk borders cull correctly.
//!
//! Tile format (little-endian, gzip on disk). A tile is a list of quads, and
//! each quad is an integer block anchor plus a shape ("template": four corner
//! offsets, UVs and a texture layer) and a color from small per-tile tables.
//! Quads are grouped by column and stored as byte planes, which gzip squeezes
//! to about one byte per quad. The viewer draws them straight from these
//! records by vertex pulling.
//!
//! ```text
//! 0   magic "TMH2"
//! 4   i32 origin x, y, z            block coords of anchor (0, 0, 0)
//! 16  u16 min qy, u16 max qy        bounds, 1/64 block from origin y
//! 20  4 × u32 quad count            opaque, cutout, leaf interior, translucent
//! 36  u32 templates T, u32 colors P
//! 44  T × 21 u16 templates          4 × corner (x, y, z) in 1/64 block + 64,
//!                                   4 × (u, v) in 1/4096 texture, layer
//!     P × u32 colors                r, g, b, sky << 4 | block light
//!     per group:
//!       W × W column counts         u16 as a low-byte then a high-byte plane
//!       n × y step                  u16 planes; y runs down from 1023 per column
//!       n × template index          u16 planes
//!       n × color index             u16 planes
//! ```
//!
//! Quads are ordered by column (z, then x) and descending y within a column.
//! Leaf faces that touch other leaves form their own group, so the viewer can
//! skip them at a distance, where they can't be seen.

use crate::model::{
    BlockInfo, DIR_VEC, DOWN, EAST, F_CULL_SAME, F_LEAVES, Layer, NORTH, Quad, Registry, SOUTH,
    Tint, UP, WEST, opposite,
};
use crate::world::{Chunk, nibble};
use std::collections::HashMap;

pub const TILE: i32 = 64;
const W: usize = TILE as usize + 2;
const GROUPS: usize = 4;
/// The group for leaf faces that touch other leaves.
const LEAF_INTERIOR: usize = 2;

pub struct Settings {
    pub min_y: i32,
    pub remove_caves_below_y: i32,
    /// Meshing a LOD tile: no halo faces, and the halo is a stand-in, not real blocks.
    pub lod: bool,
}

/// Faces on a tile's outer edge within this many blocks (or voxels) of the
/// column's top are always drawn. A neighbor drawn at another detail level
/// can sit a little higher or lower, and these "skirts" close the gap.
const SKIRT: i32 = 2;

/// Dense block data for one tile plus a one-block halo.
pub struct Volume {
    pub x0: i32,
    pub z0: i32,
    /// World y of volume layer 0.
    pub y0: i32,
    pub h: usize,
    blocks: Vec<u16>,
    /// sky << 4 | block light.
    light: Vec<u8>,
    biomes: Vec<u16>,
    /// Per column (W×W): ocean floor y, or i32::MIN when caves can't be detected.
    floor: Vec<i32>,
}

impl Volume {
    pub fn new() -> Self {
        Volume {
            x0: 0,
            z0: 0,
            y0: 0,
            h: 0,
            blocks: Vec::new(),
            light: Vec::new(),
            biomes: Vec::new(),
            floor: Vec::new(),
        }
    }

    /// Fills the volume for the tile whose min corner is (x0, z0).
    /// `chunk(cx, cz)` returns decoded chunks (None if missing).
    pub fn fill<'c>(
        &mut self,
        x0: i32,
        z0: i32,
        chunk: &dyn Fn(i32, i32) -> Option<&'c Chunk>,
    ) -> bool {
        let (cx0, cz0) = (x0 >> 4, z0 >> 4);
        // An empty tile has nothing to draw.
        let core = (cz0..cz0 + 4).flat_map(|cz| (cx0..cx0 + 4).map(move |cx| (cx, cz)));
        if !core
            .filter_map(|(cx, cz)| chunk(cx, cz))
            .any(|c| c.sections.iter().any(|s| !s.is_empty()))
        {
            return false;
        }
        // Vertical range from the tile and its neighbors, so the halo faces of a
        // taller neighbor are complete.
        let (mut ylo, mut yhi) = (i32::MAX, i32::MIN);
        for cz in cz0 - 1..cz0 + 5 {
            for cx in cx0 - 1..cx0 + 5 {
                if let Some(c) = chunk(cx, cz) {
                    for s in c.sections.iter().filter(|s| !s.is_empty()) {
                        ylo = ylo.min(s.y * 16);
                        yhi = yhi.max(s.y * 16 + 15);
                    }
                }
            }
        }
        self.x0 = x0;
        self.z0 = z0;
        self.y0 = ylo - 1;
        self.h = (yhi - ylo + 3) as usize;
        let n = W * W * self.h;
        self.blocks.clear();
        self.blocks.resize(n, 0);
        self.light.clear();
        self.light.resize(n, 0xf0);
        self.biomes.clear();
        self.biomes.resize(n, 0);
        self.floor.clear();
        self.floor.resize(W * W, i32::MIN);

        for cz in cz0 - 1..cz0 + 5 {
            for cx in cx0 - 1..cx0 + 5 {
                let Some(c) = chunk(cx, cz) else { continue };
                // Overlap of this chunk with the volume, in local chunk coords.
                let lx0 = (x0 - 1 - cx * 16).max(0);
                let lx1 = (x0 + TILE + 1 - cx * 16).min(16);
                let lz0 = (z0 - 1 - cz * 16).max(0);
                let lz1 = (z0 + TILE + 1 - cz * 16).min(16);
                if lx0 >= lx1 || lz0 >= lz1 {
                    continue;
                }
                for lz in lz0..lz1 {
                    for lx in lx0..lx1 {
                        let vx = (cx * 16 + lx - (x0 - 1)) as usize;
                        let vz = (cz * 16 + lz - (z0 - 1)) as usize;
                        if c.has_light
                            && let Some(f) = &c.ocean_floor
                        {
                            self.floor[vz * W + vx] = f[(lz * 16 + lx) as usize];
                        }
                    }
                }
                // Sky light for sections without saved data comes from the bottom
                // row of the nearest section above that has it (15 if none),
                // matching how the game reads missing sky light.
                let mut carry = [15u8; 256];
                for s in c.sections.iter().rev() {
                    let sy0 = s.y * 16;
                    let in_range = sy0 + 15 >= self.y0 && sy0 < self.y0 + self.h as i32;
                    if in_range {
                        for ly in 0..16 {
                            let wy = sy0 + ly;
                            if wy < self.y0 || wy >= self.y0 + self.h as i32 {
                                continue;
                            }
                            let vy = (wy - self.y0) as usize;
                            for lz in lz0..lz1 {
                                let vz = (cz * 16 + lz - (z0 - 1)) as usize;
                                let row = (vy * W + vz) * W;
                                for lx in lx0..lx1 {
                                    let vx = (cx * 16 + lx - (x0 - 1)) as usize;
                                    let si = ((ly * 16 + lz) * 16 + lx) as usize;
                                    let vi = row + vx;
                                    self.blocks[vi] = s.block(si);
                                    let sky = match &s.sky {
                                        Some(a) => nibble(a, si),
                                        None => carry[(lz * 16 + lx) as usize],
                                    };
                                    let bl = match &s.block_light {
                                        Some(a) => nibble(a, si),
                                        None => 0,
                                    };
                                    self.light[vi] = sky << 4 | bl;
                                    self.biomes[vi] = s.biomes
                                        [(((ly >> 2) * 4 + (lz >> 2)) * 4 + (lx >> 2)) as usize];
                                }
                            }
                        }
                    }
                    if let Some(a) = &s.sky {
                        for (i, v) in carry.iter_mut().enumerate() {
                            *v = nibble(a, i);
                        }
                    }
                }
            }
        }
        true
    }

    /// Fills the volume from a LOD voxel grid of 64×64 columns whose min corner
    /// is (x0, z0) in voxel units. The halo is a solid stand-in (`skirt`) since
    /// LOD tiles don't read their neighbors; skirts close the borders instead.
    /// Fills the volume from a LOD voxel grid of 66×66 columns: the tile's
    /// 64×64 plus a one-column ring from its neighbors. (x0, z0) is the tile's
    /// min corner in voxel units.
    pub fn fill_grid(
        &mut self,
        g: &crate::lod::Grid,
        x0: i32,
        z0: i32,
        reg: &Registry,
        scale: i32,
    ) -> bool {
        debug_assert_eq!(g.w, W);
        let Some((ylo, yhi)) = g.y_range() else {
            return false;
        };
        self.x0 = x0;
        self.z0 = z0;
        self.y0 = g.y0 + ylo as i32 - 1;
        self.h = yhi - ylo + 3;
        let n = W * W * self.h;
        self.blocks.clear();
        self.blocks.resize(n, 0);
        self.light.clear();
        self.light.resize(n, 0xf0);
        self.biomes.clear();
        self.biomes.resize(n, 0);
        self.floor.clear();
        self.floor.resize(W * W, i32::MIN);
        for vz in 0..W {
            for vx in 0..W {
                let c = vz * W + vx;
                // Sky light fades through water, a level per block, so seabeds
                // darken with depth the way they do at hires.
                let mut sky = 15i32;
                for vy in (0..self.h).rev() {
                    let i = self.idx(vx, vy, vz);
                    let gy = (vy + ylo) as isize - 1;
                    let id = if gy < 0 || gy as usize >= g.h {
                        0
                    } else {
                        g.states[c * g.h + gy as usize]
                    };
                    self.blocks[i] = id;
                    self.biomes[i] = g.biomes[c];
                    if id != 0 && reg.get(id).has_water() {
                        sky = (sky - scale).max(0);
                    }
                    self.light[i] = (sky as u8) << 4;
                }
            }
        }
        true
    }

    #[inline]
    pub fn block(&self, vx: usize, vy: usize, vz: usize) -> u16 {
        self.blocks[self.idx(vx, vy, vz)]
    }

    #[inline]
    pub fn biome(&self, vx: usize, vy: usize, vz: usize) -> u16 {
        self.biomes[self.idx(vx, vy, vz)]
    }

    #[inline]
    pub fn idx(&self, vx: usize, vy: usize, vz: usize) -> usize {
        (vy * W + vz) * W + vx
    }

    /// BlueMap-style cave removal: below a y threshold and the ocean floor,
    /// blocks with no sky light around them are skipped.
    #[inline]
    pub fn cave(&self, vx: usize, vy: usize, vz: usize, s: &Settings) -> bool {
        let v = self;
        let wy = v.y0 + vy as i32;
        if wy >= s.remove_caves_below_y {
            return false;
        }
        let floor = v.floor[vz * W + vx];
        if floor == i32::MIN || wy >= floor - 1 {
            return false;
        }
        let i = v.idx(vx, vy, vz);
        if v.light[i] >> 4 != 0 {
            return false;
        }
        for d in 0..6u8 {
            // Halo blocks have no neighbor outside the volume; skip that side.
            let o = DIR_VEC[d as usize];
            let (nx, nz) = (vx as i32 + o[0], vz as i32 + o[2]);
            if nx < 0 || nz < 0 || nx >= W as i32 || nz >= W as i32 {
                continue;
            }
            if v.light[neighbor(v, vx, vy, vz, d)] >> 4 != 0 {
                return false;
            }
        }
        true
    }
}

/// A template: 4 corner offsets (x, y, z), 4 UVs (u, v), and a texture layer.
type Template = [u16; 21];

struct Rec {
    x: u8,
    z: u8,
    y: u16,
    tpl: Template,
    color: u32,
}

pub struct Mesher {
    groups: [Vec<Rec>; GROUPS],
    min_qy: u16,
    max_qy: u16,
}

const BRIGHT: [f32; 16] = {
    let mut t = [0f32; 16];
    let mut i = 0;
    while i < 16 {
        // 0.22 .. 1.0, slightly convex.
        let f = i as f32 / 15.0;
        t[i] = 0.22 + 0.78 * f * (0.6 + 0.4 * f);
        i += 1;
    }
    t
};

fn tint_color(info: &BlockInfo, biome: crate::biome::BiomeColors) -> [u8; 3] {
    match info.tint {
        Tint::Grass => biome.grass,
        Tint::Foliage => biome.foliage,
        Tint::DryFoliage => biome.dry_foliage,
        Tint::Water => biome.water,
        Tint::Fixed(c) => c,
    }
}

impl Mesher {
    pub fn new() -> Self {
        Mesher {
            groups: Default::default(),
            min_qy: u16::MAX,
            max_qy: 0,
        }
    }

    /// Meshes the volume. Returns the encoded (uncompressed) tile, or None if empty.
    pub fn mesh(&mut self, v: &Volume, reg: &Registry, s: &Settings) -> Option<Vec<u8>> {
        for g in &mut self.groups {
            g.clear();
        }
        self.min_qy = u16::MAX;
        self.max_qy = 0;

        let origin = [v.x0 - 1, s.min_y - 1, v.z0 - 1];
        let biomes = &reg.baker.biomes;
        let water_tex = reg.baker.fluid_texture(false);
        let lava_tex = reg.baker.fluid_texture(true);

        for vz in 1..=TILE as usize {
            for vx in 1..=TILE as usize {
                // Top of the current solid run, found on the way down: ground
                // under a roof or canopy starts a run of its own.
                let mut top = i32::MAX;
                let mut open_above = true;
                for vy in (1..v.h - 1).rev() {
                    let was_open = std::mem::replace(&mut open_above, true);
                    let i = v.idx(vx, vy, vz);
                    let id = v.blocks[i];
                    if id == 0 {
                        continue;
                    }
                    let info = reg.get(id);
                    if !info.visible() {
                        continue;
                    }
                    let wx = v.x0 + vx as i32 - 1;
                    let wy = v.y0 + vy as i32;
                    let wz = v.z0 + vz as i32 - 1;
                    if v.cave(vx, vy, vz, s) {
                        continue;
                    }
                    let base = [wx - origin[0], wy - origin[1], wz - origin[2]];
                    open_above = false;
                    if was_open {
                        top = wy;
                    }

                    let quads = info.variant(wx, wy, wz);
                    let mut tint: Option<[u8; 3]> = None;
                    for q in quads {
                        if let Some(d) = q.cull {
                            if d == DOWN && wy == s.min_y {
                                continue;
                            }
                            let n = neighbor(v, vx, vy, vz, d);
                            let nid = v.blocks[n];
                            let ninfo = reg.get(nid);
                            if ninfo.occludes & (1 << opposite(d)) != 0 {
                                if wy >= top - (SKIRT - 1) && outward(vx, vz, d) {
                                    // Skirt: kept against a solid neighbor, lit as if open.
                                    let color = if q.tinted {
                                        tint_color(info, biomes.colors(v.biomes[i]))
                                    } else {
                                        [255, 255, 255]
                                    };
                                    self.emit(q, base, color, 0xf0, group_of(q.layer));
                                }
                                continue;
                            }
                            if info.flags & F_CULL_SAME != 0
                                && (nid == id || ninfo.name == info.name)
                            {
                                continue;
                            }
                        }
                        let light = self.face_light(v, i, vx, vy, vz, q);
                        let color = if q.tinted {
                            *tint
                                .get_or_insert_with(|| tint_color(info, biomes.colors(v.biomes[i])))
                        } else {
                            [255, 255, 255]
                        };
                        let group = match q.cull {
                            Some(d)
                                if info.flags & F_LEAVES != 0
                                    && reg.get(v.blocks[neighbor(v, vx, vy, vz, d)]).flags
                                        & F_LEAVES
                                        != 0 =>
                            {
                                LEAF_INTERIOR
                            }
                            _ => group_of(q.layer),
                        };
                        self.emit(q, base, color, light, group);
                    }

                    if info.is_fluid() || info.has_water() {
                        let lava = info.flags & crate::model::F_LAVA != 0;
                        let tex = if lava { lava_tex } else { water_tex };
                        let color = if lava {
                            [255, 255, 255]
                        } else {
                            biomes.colors(v.biomes[i]).water
                        };
                        self.fluid(v, reg, vx, vy, vz, lava, tex, base, color);
                    }
                }
            }
        }

        if !s.lod {
            self.halo_faces(v, reg, s, origin);
        }

        if self.groups.iter().all(|g| g.is_empty()) {
            return None;
        }
        Some(self.encode(origin))
    }

    /// Faces of the halo blocks that point into the tile, such as a cliff that
    /// belongs to the neighboring tile. They close the gap where this tile meets
    /// a neighbor drawn at lowres; when the neighbor is drawn at hires too, its
    /// identical face covers the same pixels.
    fn halo_faces(&mut self, v: &Volume, reg: &Registry, s: &Settings, origin: [i32; 3]) {
        let biomes = &reg.baker.biomes;
        let last = W - 1;
        // (halo strip x or z, whether it's an x strip, face direction into the tile)
        let strips = [
            (0, true, EAST),
            (last, true, WEST),
            (0, false, SOUTH),
            (last, false, NORTH),
        ];
        for (fixed, is_x, d) in strips {
            for t in 1..=TILE as usize {
                let (vx, vz) = if is_x { (fixed, t) } else { (t, fixed) };
                for vy in 1..v.h - 1 {
                    let i = v.idx(vx, vy, vz);
                    let id = v.blocks[i];
                    if id == 0 {
                        continue;
                    }
                    let info = reg.get(id);
                    let wy = v.y0 + vy as i32;
                    if info.is_air() || v.cave(vx, vy, vz, s) {
                        continue;
                    }
                    let n = neighbor(v, vx, vy, vz, d);
                    let ninfo = reg.get(v.blocks[n]);
                    if ninfo.occludes & (1 << opposite(d)) != 0 {
                        continue;
                    }
                    if info.flags & F_CULL_SAME != 0
                        && (v.blocks[n] == id || ninfo.name == info.name)
                    {
                        continue;
                    }
                    let wx = v.x0 + vx as i32 - 1;
                    let wz = v.z0 + vz as i32 - 1;
                    let base = [wx - origin[0], wy - origin[1], wz - origin[2]];
                    for q in info
                        .variant(wx, wy, wz)
                        .iter()
                        // Translucent copies would blend twice where the neighbor is drawn too.
                        .filter(|q| q.cull == Some(d) && q.layer != Layer::Translucent)
                    {
                        let color = if q.tinted {
                            tint_color(info, biomes.colors(v.biomes[i]))
                        } else {
                            [255, 255, 255]
                        };
                        // Leaf against leaf goes with the other leaf interiors.
                        let group = if info.flags & F_LEAVES != 0 && ninfo.flags & F_LEAVES != 0 {
                            LEAF_INTERIOR
                        } else {
                            group_of(q.layer)
                        };
                        self.emit(q, base, color, v.light[n], group);
                    }
                }
            }
        }
    }

    #[inline]
    fn face_light(&self, v: &Volume, i: usize, vx: usize, vy: usize, vz: usize, q: &Quad) -> u8 {
        match q.cull {
            Some(d) => v.light[neighbor(v, vx, vy, vz, d)],
            None => {
                let own = v.light[i];
                let n = v.light[neighbor(v, vx, vy, vz, q.light_dir)];
                ((own >> 4).max(n >> 4) << 4) | (own & 15).max(n & 15)
            }
        }
    }

    fn emit(&mut self, q: &Quad, base: [i32; 3], color: [u8; 3], light: u8, group: usize) {
        let l = (light >> 4).max(light & 15) as usize;
        let f = q.shade * BRIGHT[l];
        let color = u32::from_le_bytes([
            (color[0] as f32 * f) as u8,
            (color[1] as f32 * f) as u8,
            (color[2] as f32 * f) as u8,
            light,
        ]);
        let mut tpl = [0u16; 21];
        for k in 0..4 {
            let p = q.pos[k];
            for a in 0..3 {
                tpl[k * 3 + a] = offset(p[a]);
            }
            tpl[12 + k * 2] = (q.uv[k][0] * 4096.0).round().clamp(0.0, 65535.0) as u16;
            tpl[13 + k * 2] = (q.uv[k][1] * 4096.0).round().clamp(0.0, 65535.0) as u16;
            let qy = (base[1] * 64 + tpl[k * 3 + 1] as i32 - 64).clamp(0, 65535) as u16;
            self.min_qy = self.min_qy.min(qy);
            self.max_qy = self.max_qy.max(qy);
        }
        tpl[20] = q.tex;
        self.groups[group].push(Rec {
            x: base[0] as u8,
            z: base[2] as u8,
            y: base[1] as u16,
            tpl,
            color,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn fluid(
        &mut self,
        v: &Volume,
        reg: &Registry,
        vx: usize,
        vy: usize,
        vz: usize,
        lava: bool,
        tex: u16,
        base: [i32; 3],
        color: [u8; 3],
    ) {
        let same = |n: usize| {
            let b = reg.get(v.blocks[n]);
            if lava {
                b.flags & crate::model::F_LAVA != 0
            } else {
                b.has_water()
            }
        };
        // Surface height of the fluid at a volume position: full under more fluid,
        // else from its level (sources and falling fluid sit at 8/9).
        let height_at = |x: usize, y: usize, z: usize| {
            if same(v.idx(x, y + 1, z)) {
                return 1.0;
            }
            let level = reg.get(v.blocks[v.idx(x, y, z)]).fluid_level;
            if level == 0 || level >= 8 {
                8.0 / 9.0
            } else {
                (8 - level.min(7)) as f32 / 9.0
            }
        };
        let up = neighbor(v, vx, vy, vz, UP);
        let height = height_at(vx, vy, vz);
        let layer = if lava {
            Layer::Opaque
        } else {
            Layer::Translucent
        };
        let i = v.idx(vx, vy, vz);
        for d in 0..6u8 {
            let n = neighbor(v, vx, vy, vz, d);
            // Beside lower fluid of the same kind, only the strip above it shows.
            let mut lo = 0.0;
            if same(n) {
                if d < 2 {
                    continue;
                }
                let o = DIR_VEC[d as usize];
                lo = height_at((vx as i32 + o[0]) as usize, vy, (vz as i32 + o[2]) as usize);
                if lo >= height {
                    continue;
                }
            }
            let ninfo = reg.get(v.blocks[n]);
            if ninfo.occludes & (1 << opposite(d)) != 0 {
                continue;
            }
            let h = height;
            let pos = match d {
                0 => [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                ],
                1 => [[0.0, h, 0.0], [0.0, h, 1.0], [1.0, h, 1.0], [1.0, h, 0.0]],
                2 => [[1.0, h, 0.0], [1.0, lo, 0.0], [0.0, lo, 0.0], [0.0, h, 0.0]],
                3 => [[0.0, h, 1.0], [0.0, lo, 1.0], [1.0, lo, 1.0], [1.0, h, 1.0]],
                4 => [[0.0, h, 0.0], [0.0, lo, 0.0], [0.0, lo, 1.0], [0.0, h, 1.0]],
                _ => [[1.0, h, 1.0], [1.0, lo, 1.0], [1.0, lo, 0.0], [1.0, h, 0.0]],
            };
            let (top_v, bottom_v) = if d < 2 {
                (0.0, 1.0)
            } else {
                (1.0 - h, 1.0 - lo)
            };
            let q = Quad {
                pos,
                uv: [[0.0, top_v], [0.0, bottom_v], [1.0, bottom_v], [1.0, top_v]],
                tex,
                layer,
                cull: Some(d),
                light_dir: d,
                tinted: true,
                shade: [0.5, 1.0, 0.8, 0.8, 0.6, 0.6][d as usize],
            };
            let light = if lava {
                0xff
            } else if d == UP {
                v.light[i].max(v.light[up])
            } else {
                v.light[n].max(v.light[i])
            };
            self.emit(&q, base, color, light, group_of(q.layer));
        }
    }

    fn encode(&mut self, origin: [i32; 3]) -> Vec<u8> {
        // Canonical order: by column, then top down. Ties keep emit order,
        // which is itself deterministic.
        for g in &mut self.groups {
            g.sort_by_key(|r| (r.z, r.x, std::cmp::Reverse(r.y)));
        }
        // Per-tile tables, numbered in order of first use.
        let mut templates: Vec<&Template> = Vec::new();
        let mut colors: Vec<u32> = Vec::new();
        let mut tpl_ids: HashMap<&Template, u16> = HashMap::new();
        let mut color_ids: HashMap<u32, u16> = HashMap::new();
        let mut refs: Vec<Vec<(u16, u16)>> = Vec::with_capacity(GROUPS);
        for g in &self.groups {
            refs.push(
                g.iter()
                    .map(|r| {
                        let t = *tpl_ids.entry(&r.tpl).or_insert_with(|| {
                            templates.push(&r.tpl);
                            (templates.len() - 1) as u16
                        });
                        let c = *color_ids.entry(r.color).or_insert_with(|| {
                            colors.push(r.color);
                            (colors.len() - 1) as u16
                        });
                        (t, c)
                    })
                    .collect(),
            );
        }

        let quads: usize = self.groups.iter().map(|g| g.len()).sum();
        let mut out = Vec::with_capacity(64 + templates.len() * 42 + quads * 6);
        out.extend_from_slice(b"TMH2");
        for o in origin {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out.extend_from_slice(&self.min_qy.to_le_bytes());
        out.extend_from_slice(&self.max_qy.to_le_bytes());
        for g in &self.groups {
            out.extend_from_slice(&(g.len() as u32).to_le_bytes());
        }
        out.extend_from_slice(&(templates.len() as u32).to_le_bytes());
        out.extend_from_slice(&(colors.len() as u32).to_le_bytes());
        for t in &templates {
            for v in t.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for c in &colors {
            out.extend_from_slice(&c.to_le_bytes());
        }
        let planes = |out: &mut Vec<u8>, vals: &[u16]| {
            out.extend(vals.iter().map(|v| *v as u8));
            out.extend(vals.iter().map(|v| (*v >> 8) as u8));
        };
        for (g, r) in self.groups.iter().zip(&refs) {
            let mut counts = vec![0u16; W * W];
            for q in g {
                counts[q.z as usize * W + q.x as usize] += 1;
            }
            planes(&mut out, &counts);
            let mut steps = Vec::with_capacity(g.len());
            let mut prev = (u8::MAX, u8::MAX, 1023u16);
            for q in g {
                if (q.z, q.x) != (prev.0, prev.1) {
                    prev = (q.z, q.x, 1023);
                }
                steps.push(prev.2 - q.y);
                prev.2 = q.y;
            }
            planes(&mut out, &steps);
            planes(&mut out, &r.iter().map(|x| x.0).collect::<Vec<_>>());
            planes(&mut out, &r.iter().map(|x| x.1).collect::<Vec<_>>());
        }
        out
    }
}

/// The tile group a render layer goes in (group 2 holds leaf interiors).
#[inline]
fn group_of(layer: Layer) -> usize {
    match layer {
        Layer::Opaque => 0,
        Layer::Cutout => 1,
        Layer::Translucent => 3,
    }
}

/// A corner coordinate relative to its anchor block, in 1/64 block plus 64,
/// so elements that reach a block past their own stay positive.
#[inline]
fn offset(p: f32) -> u16 {
    ((p * 64.0).round() + 64.0).clamp(0.0, 65535.0) as u16
}

/// True if face direction `d` points out of the tile from border column (vx, vz).
#[inline]
fn outward(vx: usize, vz: usize, d: u8) -> bool {
    match d {
        WEST => vx == 1,
        EAST => vx == W - 2,
        NORTH => vz == 1,
        SOUTH => vz == W - 2,
        _ => false,
    }
}

#[inline]
fn neighbor(v: &Volume, vx: usize, vy: usize, vz: usize, d: u8) -> usize {
    let o = DIR_VEC[d as usize];
    v.idx(
        (vx as i32 + o[0]) as usize,
        (vy as i32 + o[1]) as usize,
        (vz as i32 + o[2]) as usize,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes a TMH2 tile back to (group, x, y, z, template, color) per quad.
    fn decode(t: &[u8]) -> Vec<(usize, u8, u16, u8, Template, u32)> {
        let u16_at = |o: usize| u16::from_le_bytes([t[o], t[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(t[o..o + 4].try_into().unwrap());
        assert_eq!(&t[..4], b"TMH2");
        let counts: Vec<usize> = (0..4).map(|g| u32_at(20 + g * 4) as usize).collect();
        let (nt, np) = (u32_at(36) as usize, u32_at(40) as usize);
        let mut o = 44;
        let templates: Vec<Template> = (0..nt)
            .map(|i| std::array::from_fn(|k| u16_at(o + i * 42 + k * 2)))
            .collect();
        o += nt * 42;
        let colors: Vec<u32> = (0..np).map(|i| u32_at(o + i * 4)).collect();
        o += np * 4;
        let mut out = Vec::new();
        for (g, &n) in counts.iter().enumerate() {
            let plane =
                |o: usize, len: usize, i: usize| t[o + i] as u16 | (t[o + len + i] as u16) << 8;
            let (cols, steps, tpls, cols_idx) = (
                o,
                o + 2 * W * W,
                o + 2 * W * W + 2 * n,
                o + 2 * W * W + 4 * n,
            );
            let mut q = 0;
            for z in 0..W {
                for x in 0..W {
                    let mut y = 1023u16;
                    for _ in 0..plane(cols, W * W, z * W + x) {
                        y -= plane(steps, n, q);
                        let tpl = templates[plane(tpls, n, q) as usize];
                        out.push((
                            g,
                            x as u8,
                            y,
                            z as u8,
                            tpl,
                            colors[plane(cols_idx, n, q) as usize],
                        ));
                        q += 1;
                    }
                }
            }
            o += 2 * W * W + 6 * n;
        }
        assert_eq!(o, t.len());
        out
    }

    fn quad(dx: f32, layer: Layer, tex: u16) -> Quad {
        Quad {
            pos: [
                [dx, 0.5, 0.0],
                [dx, 0.0, 0.0],
                [dx + 1.0, 0.0, 1.25],
                [dx + 1.0, 0.5, 1.0],
            ],
            uv: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.5]],
            tex,
            layer,
            cull: None,
            light_dir: UP,
            tinted: false,
            shade: 1.0,
        }
    }

    #[test]
    fn skirts_cover_ground_under_a_roof() {
        use crate::assets::AssetFiles;
        use crate::model::Baker;
        let reg = Registry::new(Baker::new(AssetFiles::new()));
        let stone = reg.id("minecraft:stone", &mut []);
        // A 66×66 LOD grid (tile plus ring). The tile's east border column has
        // ground up to y=5 and a roof at y=20; the ring next to it is solid up
        // to y=8, like a neighbor drawn at another level.
        let h = 32;
        let mut g = crate::lod::Grid::empty(W, h, 0);
        let col = |x: usize, z: usize| z * W + x;
        for z in 0..W {
            let c = col(W - 2, z);
            g.states[c * h..c * h + 6].fill(stone);
            g.states[c * h + 20] = stone;
            let ring = col(W - 1, z);
            g.states[ring * h..ring * h + 9].fill(stone);
        }
        let mut v = Volume::new();
        assert!(v.fill_grid(&g, 0, 0, &reg, 2));
        let s = Settings {
            min_y: 0,
            remove_caves_below_y: i32::MIN,
            lod: true,
        };
        let tile = Mesher::new().mesh(&v, &reg, &s).unwrap();
        // East faces have all four corners at x = 1 block, stored as 64 + 64.
        let east_at = |y: u16| {
            decode(&tile)
                .iter()
                .any(|q| q.1 == 64 && q.2 == y + 1 && (0..4).all(|k| q.4[k * 3] == 128))
        };
        assert!(east_at(5), "ground under the roof has no skirt at its top");
        assert!(east_at(4), "the skirt covers two voxels");
        assert!(!east_at(3), "faces below the skirt still cull");
    }

    #[test]
    fn tmh2_round_trips() {
        let mut m = Mesher::new();
        let mut expected = Vec::new();
        // Quads in several columns, stacked in one column, across all groups,
        // and with offsets that reach past their block (negative and > 1).
        let cases = [
            ([0, 1, 0], -1.0, Layer::Opaque, 3, 0),
            ([65, 385, 65], 0.0, Layer::Cutout, 4, 1),
            ([10, 70, 20], 0.25, Layer::Opaque, 3, 0),
            ([10, 90, 20], 0.0, Layer::Translucent, 7, 3),
            ([10, 70, 20], 0.0, Layer::Cutout, 5, LEAF_INTERIOR),
            ([10, 71, 20], 0.0, Layer::Opaque, 3, 0),
            // Worlds can be up to 1000 blocks tall.
            ([3, 1001, 4], 0.0, Layer::Opaque, 3, 0),
        ];
        for (i, (base, dx, layer, tex, group)) in cases.into_iter().enumerate() {
            let q = quad(dx, layer, tex);
            m.emit(&q, base, [i as u8 * 40, 200, 30], 0xf3, group);
            let r = m.groups[group].last().unwrap();
            expected.push((group, r.x, r.y, r.z, r.tpl, r.color));
        }
        let tile = m.encode([0, 0, 0]);
        let mut got = decode(&tile);
        got.sort();
        expected.sort();
        assert_eq!(got, expected);
        // Offsets are stored with a +64 bias in 1/64 block.
        assert_eq!(
            expected
                .iter()
                .find(|e| e.4[20] == 3 && e.2 == 1)
                .unwrap()
                .4[0],
            0
        );
    }
}
