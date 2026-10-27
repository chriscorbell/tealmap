//! Hires tile meshing and the binary tile format.
//!
//! A tile covers 64×64 columns. Blocks are copied into a dense volume with a
//! one-block halo so faces on tile and chunk borders cull correctly.
//!
//! Tile format (little-endian, gzip on disk):
//!
//! ```text
//! 0   magic "TMH1"
//! 4   i32 origin x, i32 origin y, i32 origin z   (block coords of quantized 0)
//! 16  u16 min qy, u16 max qy                     (bounds, 1/64 block)
//! 20  3 × u32 vertex count                       (opaque, cutout, translucent)
//! 32  per group: pos u16×3, uvl u16×3 (u, v in 1/4096 texture, layer),
//!     color u8×4 (rgb, sky<<4|block); each array padded to 4 bytes
//!
//! Every 4 vertices form a quad (triangles 0-1-2, 0-2-3), so there are no
//! indices on disk; the viewer shares one index buffer across tiles.
//! ```

use crate::model::{
    BlockInfo, DIR_VEC, DOWN, F_CULL_SAME, Layer, Quad, Registry, Tint, UP, opposite,
};
use crate::world::{Chunk, nibble};

pub const TILE: i32 = 64;
const W: usize = TILE as usize + 2;

pub struct Settings {
    pub min_y: i32,
    pub remove_caves_below_y: i32,
}

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
        // Vertical range from the tile's own chunks.
        let (mut ylo, mut yhi) = (i32::MAX, i32::MIN);
        for cz in cz0..cz0 + 4 {
            for cx in cx0..cx0 + 4 {
                if let Some(c) = chunk(cx, cz) {
                    for s in c.sections.iter().filter(|s| !s.is_empty()) {
                        ylo = ylo.min(s.y * 16);
                        yhi = yhi.max(s.y * 16 + 15);
                    }
                }
            }
        }
        if ylo > yhi {
            return false;
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

    #[inline]
    fn idx(&self, vx: usize, vy: usize, vz: usize) -> usize {
        (vy * W + vz) * W + vx
    }
}

#[derive(Default)]
struct Group {
    pos: Vec<u16>,
    uvl: Vec<u16>,
    color: Vec<u8>,
}

impl Group {
    fn verts(&self) -> usize {
        self.pos.len() / 3
    }
}

/// Lowres sample for one column: color and top surface height (absolute y + 1).
#[derive(Clone, Copy, Default)]
pub struct Column {
    pub color: [u8; 3],
    /// 0 = no data, else top surface y - min_y + 1.
    pub height: u16,
}

pub struct Mesher {
    groups: [Group; 3],
    min_qy: u16,
    max_qy: u16,
    pub columns: Vec<Column>,
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
            columns: vec![Column::default(); (TILE * TILE) as usize],
        }
    }

    /// Meshes the volume. Returns the encoded (uncompressed) tile, or None if empty.
    pub fn mesh(&mut self, v: &Volume, reg: &Registry, s: &Settings) -> Option<Vec<u8>> {
        for g in &mut self.groups {
            g.pos.clear();
            g.uvl.clear();
            g.color.clear();
        }
        self.min_qy = u16::MAX;
        self.max_qy = 0;
        self.columns.fill(Column::default());

        let origin = [v.x0 - 1, s.min_y - 1, v.z0 - 1];
        let biomes = &reg.baker.biomes;
        let water_tex = reg.baker.fluid_texture(false);
        let lava_tex = reg.baker.fluid_texture(true);

        for vz in 1..=TILE as usize {
            for vx in 1..=TILE as usize {
                let mut col_done = false;
                let mut water_top: Option<(u16, [u8; 3])> = None;
                for vy in (1..v.h - 1).rev() {
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
                    if self.is_cave(v, vx, vy, vz, wy, s) {
                        continue;
                    }
                    let base = [wx - origin[0], wy - origin[1], wz - origin[2]];

                    // Lowres column: first visible block from the top.
                    if !col_done {
                        let bc = biomes.colors(v.biomes[i]);
                        let h = (wy + 1 - s.min_y + 1).clamp(0, u16::MAX as i32) as u16;
                        if info.has_water() && info.variants.iter().all(|x| x.quads.is_empty())
                            || info.flags & crate::model::F_WATER != 0
                        {
                            if water_top.is_none() {
                                water_top = Some((h, bc.water));
                            }
                        } else {
                            let mut c = info.top_color;
                            if info.top_tinted {
                                let t = tint_color(info, bc);
                                c = [mul(c[0], t[0]), mul(c[1], t[1]), mul(c[2], t[2])];
                            }
                            let col = &mut self.columns[(vz - 1) * TILE as usize + (vx - 1)];
                            match water_top {
                                Some((wh, wc)) => {
                                    let depth = (wh as i32 - h as i32).clamp(0, 24) as f32 / 24.0;
                                    let k = 0.55 + 0.4 * depth;
                                    let wcol = [mul(wc[0], 180), mul(wc[1], 180), mul(wc[2], 180)];
                                    col.color = lerp3(c, wcol, k);
                                    col.height = wh;
                                }
                                None => {
                                    col.color = c;
                                    col.height = h;
                                }
                            }
                            col_done = true;
                        }
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
                        self.emit(q, base, color, light);
                    }

                    if info.is_fluid() || info.has_water() {
                        let lava = info.flags & crate::model::F_LAVA != 0;
                        let tex = if lava { lava_tex } else { water_tex };
                        let color = if lava {
                            [255, 255, 255]
                        } else {
                            biomes.colors(v.biomes[i]).water
                        };
                        self.fluid(v, reg, vx, vy, vz, info, lava, tex, base, color);
                    }
                }
                if !col_done && let Some((wh, wc)) = water_top {
                    let col = &mut self.columns[(vz - 1) * TILE as usize + (vx - 1)];
                    col.color = wc;
                    col.height = wh;
                }
            }
        }

        if self.groups.iter().all(|g| g.pos.is_empty()) {
            return None;
        }
        Some(self.encode(origin))
    }

    /// BlueMap-style cave removal: below a y threshold and the ocean floor,
    /// blocks with no sky light around them are skipped.
    #[inline]
    fn is_cave(&self, v: &Volume, vx: usize, vy: usize, vz: usize, wy: i32, s: &Settings) -> bool {
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
            if v.light[neighbor(v, vx, vy, vz, d)] >> 4 != 0 {
                return false;
            }
        }
        true
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

    fn emit(&mut self, q: &Quad, base: [i32; 3], color: [u8; 3], light: u8) {
        let l = (light >> 4).max(light & 15) as usize;
        let f = q.shade * BRIGHT[l];
        let c = [
            (color[0] as f32 * f) as u8,
            (color[1] as f32 * f) as u8,
            (color[2] as f32 * f) as u8,
            light,
        ];
        let g = &mut self.groups[q.layer as usize];
        for k in 0..4 {
            let p = q.pos[k];
            let qx = quant(base[0] as f32 + p[0]);
            let qy = quant(base[1] as f32 + p[1]);
            let qz = quant(base[2] as f32 + p[2]);
            self.min_qy = self.min_qy.min(qy);
            self.max_qy = self.max_qy.max(qy);
            g.pos.extend_from_slice(&[qx, qy, qz]);
            let u = (q.uv[k][0] * 4096.0).round().clamp(0.0, 65535.0) as u16;
            let vv = (q.uv[k][1] * 4096.0).round().clamp(0.0, 65535.0) as u16;
            g.uvl.extend_from_slice(&[u, vv, q.tex]);
            g.color.extend_from_slice(&c);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fluid(
        &mut self,
        v: &Volume,
        reg: &Registry,
        vx: usize,
        vy: usize,
        vz: usize,
        info: &BlockInfo,
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
        let up = neighbor(v, vx, vy, vz, UP);
        let height = if same(up) {
            1.0
        } else if info.fluid_level == 0 || info.fluid_level >= 8 {
            8.0 / 9.0
        } else {
            (8 - info.fluid_level.min(7)) as f32 / 9.0
        };
        let layer = if lava {
            Layer::Opaque
        } else {
            Layer::Translucent
        };
        let i = v.idx(vx, vy, vz);
        for d in 0..6u8 {
            let n = neighbor(v, vx, vy, vz, d);
            if same(n) {
                continue;
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
                2 => [
                    [1.0, h, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, h, 0.0],
                ],
                3 => [
                    [0.0, h, 1.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, h, 1.0],
                ],
                4 => [
                    [0.0, h, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, h, 1.0],
                ],
                _ => [
                    [1.0, h, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, h, 0.0],
                ],
            };
            let top_v = if d < 2 { 0.0 } else { 1.0 - h };
            let q = Quad {
                pos,
                uv: [[0.0, top_v], [0.0, 1.0], [1.0, 1.0], [1.0, top_v]],
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
            self.emit(&q, base, color, light);
        }
    }

    fn encode(&self, origin: [i32; 3]) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            32 + self
                .groups
                .iter()
                .map(|g| g.verts() * 16 + 8)
                .sum::<usize>(),
        );
        out.extend_from_slice(b"TMH1");
        for o in origin {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out.extend_from_slice(&self.min_qy.to_le_bytes());
        out.extend_from_slice(&self.max_qy.to_le_bytes());
        for g in &self.groups {
            out.extend_from_slice(&(g.verts() as u32).to_le_bytes());
        }
        let pad = |out: &mut Vec<u8>| {
            while !out.len().is_multiple_of(4) {
                out.push(0);
            }
        };
        for g in &self.groups {
            for v in &g.pos {
                out.extend_from_slice(&v.to_le_bytes());
            }
            pad(&mut out);
            for v in &g.uvl {
                out.extend_from_slice(&v.to_le_bytes());
            }
            pad(&mut out);
            out.extend_from_slice(&g.color);
        }
        out
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

#[inline]
fn quant(v: f32) -> u16 {
    (v * 64.0).round().clamp(0.0, 65535.0) as u16
}

#[inline]
fn mul(a: u8, b: u8) -> u8 {
    ((a as u16 * b as u16 + 127) / 255) as u8
}

fn lerp3(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2])]
}
