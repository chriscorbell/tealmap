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

use crate::biome::BiomeColors;
use crate::model::{
    AIR, BlockInfo, DIR_VEC, DOWN, EAST, F_CULL_SAME, F_LEAVES, Layer, NORTH, Quad, QuadShape,
    Registry, SOUTH, Tint, UP, WEST, opposite,
};
use crate::world::Chunk;
use rustc_hash::FxHashMap;

pub const TILE: i32 = 64;
const W: usize = TILE as usize + 2;
/// Volume index strides: a row of x is contiguous, then z, then y.
const STRIDE_Z: usize = W;
const STRIDE_Y: usize = W * W;
/// Biome cells (4×4×4 blocks) per volume side: 16 per tile plus the halo's.
const BIOME_W: usize = TILE as usize / 4 + 2;
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
    /// A hires volume has a biome per 4×4×4 cell, a LOD volume per column.
    biomes: Vec<u16>,
    per_column_biomes: bool,
    /// Per column (W×W): ocean floor y, or i32::MIN when caves can't be detected.
    floor: Vec<i32>,
    /// Below `caves_below`: a voxel's light ORed with its six neighbors'.
    lit: Vec<u8>,
    caves_below: i32,
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
            per_column_biomes: false,
            floor: Vec::new(),
            lit: Vec::new(),
            caves_below: i32::MIN,
        }
    }

    /// Fills the volume for the tile whose min corner is (x0, z0).
    /// `chunk(cx, cz)` returns decoded chunks (None if missing).
    pub fn fill<'c>(
        &mut self,
        x0: i32,
        z0: i32,
        settings: &Settings,
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
        self.biomes
            .resize(BIOME_W * BIOME_W * (self.h.div_ceil(4) + 1), 0);
        self.per_column_biomes = false;
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
                if c.has_light
                    && let Some(f) = &c.ocean_floor
                {
                    for lz in lz0..lz1 {
                        let vz = (cz * 16 + lz - (z0 - 1)) as usize;
                        let from = (lz * 16 + lx0) as usize;
                        let to = vz * W + (cx * 16 + lx0 - (x0 - 1)) as usize;
                        let len = (lx1 - lx0) as usize;
                        self.floor[to..to + len].copy_from_slice(&f[from..from + len]);
                    }
                }
                let len = (lx1 - lx0) as usize;
                let vx0 = (cx * 16 + lx0 - (x0 - 1)) as usize;
                for s in &c.sections {
                    let sy0 = s.y * 16;
                    if sy0 + 15 < self.y0 || sy0 >= self.y0 + self.h as i32 {
                        continue;
                    }
                    self.fill_biomes(s, cx, cz);
                    // Air in full sky light is what the volume starts as.
                    let blank = s.is_empty() && s.light.is_none() && s.uniform_light == 0xf0;
                    if blank {
                        continue;
                    }
                    for ly in 0..16 {
                        let wy = sy0 + ly;
                        if wy < self.y0 || wy >= self.y0 + self.h as i32 {
                            continue;
                        }
                        let vy = (wy - self.y0) as usize;
                        for lz in lz0..lz1 {
                            let vz = (cz * 16 + lz - (z0 - 1)) as usize;
                            let to = (vy * W + vz) * W + vx0;
                            let from = ((ly * 16 + lz) * 16 + lx0) as usize;
                            match &s.blocks {
                                Some(b) => {
                                    self.blocks[to..to + len].copy_from_slice(&b[from..from + len])
                                }
                                None => self.blocks[to..to + len].fill(s.single),
                            }
                            match &s.light {
                                Some(l) => {
                                    self.light[to..to + len].copy_from_slice(&l[from..from + len])
                                }
                                None => self.light[to..to + len].fill(s.uniform_light),
                            }
                        }
                    }
                }
            }
        }
        self.build_lit(settings.remove_caves_below_y);
        true
    }

    /// Copies a section's biome cells that overlap the volume.
    fn fill_biomes(&mut self, s: &crate::world::Section, cx: i32, cz: i32) {
        let origin = |v: i32| v.div_euclid(4);
        let (ox, oy, oz) = (origin(self.x0 - 1), origin(self.y0), origin(self.z0 - 1));
        let layers = self.biomes.len() / (BIOME_W * BIOME_W);
        for cy in 0..4 {
            let vy = origin(s.y * 16 + cy * 4) - oy;
            if vy < 0 || vy as usize >= layers {
                continue;
            }
            for lz in 0..4 {
                let vz = cz * 4 + lz - oz;
                if vz < 0 || vz as usize >= BIOME_W {
                    continue;
                }
                for lx in 0..4 {
                    let vx = cx * 4 + lx - ox;
                    if vx < 0 || vx as usize >= BIOME_W {
                        continue;
                    }
                    let to = (vy as usize * BIOME_W + vz as usize) * BIOME_W + vx as usize;
                    self.biomes[to] = s.biomes[((cy * 4 + lz) * 4 + lx) as usize];
                }
            }
        }
    }

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
        self.biomes.extend_from_slice(&g.biomes);
        self.per_column_biomes = true;
        self.floor.clear();
        self.floor.resize(W * W, i32::MIN);
        self.lit.clear();
        self.caves_below = i32::MIN;
        let infos = reg.snapshot();
        for vz in 0..W {
            for vx in 0..W {
                let c = vz * W + vx;
                let column = &g.states[c * g.h..(c + 1) * g.h];
                // Sky light fades through water, a level per block, so seabeds
                // darken with depth the way they do at hires.
                let mut sky = 15i32;
                for vy in (0..self.h).rev() {
                    let gy = (vy + ylo) as isize - 1;
                    let id = if gy < 0 || gy as usize >= g.h {
                        0
                    } else {
                        column[gy as usize]
                    };
                    if id == 0 && sky == 15 {
                        continue;
                    }
                    if id != 0 && infos[id as usize].has_water() {
                        sky = (sky - scale).max(0);
                    }
                    let i = self.idx(vx, vy, vz);
                    self.blocks[i] = id;
                    self.light[i] = (sky as u8) << 4;
                }
            }
        }
        true
    }

    /// For each voxel below `below`, ORs its light with its neighbors', a row
    /// at a time so it vectorizes. Halo columns have no neighbor outside the
    /// volume on that side.
    fn build_lit(&mut self, below: i32) {
        self.caves_below = below;
        let layers = below.saturating_sub(self.y0).clamp(0, self.h as i32 - 1) as usize;
        self.lit.clear();
        self.lit.resize(layers * W * W, 0);
        let l = &self.light;
        for vy in 1..layers {
            for vz in 0..W {
                let row = (vy * W + vz) * W;
                let out = &mut self.lit[row..row + W];
                let this = &l[row..row + W];
                let or = |out: &mut [u8], from: &[u8]| {
                    for (o, &v) in out.iter_mut().zip(from) {
                        *o |= v;
                    }
                };
                out.copy_from_slice(this);
                or(out, &l[row - STRIDE_Y..row - STRIDE_Y + W]);
                or(out, &l[row + STRIDE_Y..row + STRIDE_Y + W]);
                if vz > 0 {
                    or(out, &l[row - STRIDE_Z..row - STRIDE_Z + W]);
                }
                if vz < W - 1 {
                    or(out, &l[row + STRIDE_Z..row + STRIDE_Z + W]);
                }
                or(&mut out[1..], &this[..W - 1]);
                or(&mut out[..W - 1], &this[1..]);
            }
        }
    }

    #[inline]
    pub fn block(&self, vx: usize, vy: usize, vz: usize) -> u16 {
        self.blocks[self.idx(vx, vy, vz)]
    }

    #[inline]
    pub fn biome(&self, vx: usize, vy: usize, vz: usize) -> u16 {
        if self.per_column_biomes {
            self.biomes[vz * W + vx]
        } else {
            // Volume coordinates are 1 past a cell boundary at x0, z0 and y0's
            // section, so a cell starts at every 4th voxel from 1.
            self.biomes[(((vy + 3) >> 2) * BIOME_W + ((vz + 3) >> 2)) * BIOME_W + ((vx + 3) >> 2)]
        }
    }

    #[inline]
    pub fn idx(&self, vx: usize, vy: usize, vz: usize) -> usize {
        (vy * W + vz) * W + vx
    }

    /// BlueMap-style cave removal: below a y threshold and the ocean floor,
    /// blocks with no sky light around them are skipped.
    #[inline]
    pub fn cave(&self, vx: usize, vy: usize, vz: usize) -> bool {
        let wy = self.y0 + vy as i32;
        if wy >= self.caves_below {
            return false;
        }
        let floor = self.floor[vz * W + vx];
        if floor == i32::MIN || wy >= floor - 1 {
            return false;
        }
        self.lit[self.idx(vx, vy, vz)] >> 4 == 0
    }
}

/// One quad in a tile: a sort key (column, then top down), its template and color.
struct Rec {
    key: u32,
    template: u32,
    color: u32,
}

impl Rec {
    fn new(base: [i32; 3], template: u32, color: u32) -> Self {
        let y = 1023u32.wrapping_sub(base[1] as u32) & 1023;
        Rec {
            key: (base[2] as u32) << 17 | (base[0] as u32) << 10 | y,
            template,
            color,
        }
    }
    fn z(&self) -> usize {
        (self.key >> 17) as usize
    }
    fn x(&self) -> usize {
        (self.key >> 10 & 127) as usize
    }
    fn y(&self) -> u16 {
        1023 - (self.key & 1023) as u16
    }
}

/// Per-state flags read for every voxel.
const IS_OCCLUDER: u8 = 1;
const ALWAYS_CULLED: u8 = 2;
const DRAWS_NOTHING: u8 = 4;

pub struct Mesher {
    groups: [Vec<Rec>; GROUPS],
    min_qy: u16,
    max_qy: u16,
    /// Flags per state id; the array size spares every lookup a bounds check.
    flags: Box<[u8; 65536]>,
    /// One byte per voxel: its flags, then whether meshing can skip it.
    voxel_flags: Vec<u8>,
    skip: Vec<u8>,
    /// Border columns only: the top of the current solid run, and whether the
    /// voxel above is open.
    run_top: Vec<i32>,
    open_above: Vec<bool>,
    /// Fluid faces, baked on first use. See [`fluid_slot`].
    fluid_quads: Vec<Option<Quad>>,
    /// Per template id: the tile it was last used in, and its table index there.
    template_slots: Vec<(u32, u16)>,
    tile_stamp: u32,
    color_ids: FxHashMap<u32, u16>,
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
            flags: vec![0; 65536].into_boxed_slice().try_into().unwrap(),
            voxel_flags: Vec::new(),
            skip: Vec::new(),
            run_top: vec![i32::MAX; W * W],
            open_above: vec![true; W * W],
            fluid_quads: vec![None; 2 * 6 * 10 * 10],
            template_slots: Vec::new(),
            tile_stamp: 0,
            color_ids: FxHashMap::default(),
        }
    }

    /// Marks voxels the mesher can skip: those that draw nothing, and those whose
    /// faces are all culled by six occluding neighbors. A row at a time, so the
    /// neighbor test vectorizes.
    fn find_skippable(&mut self, v: &Volume, infos: &[&BlockInfo]) {
        for (id, info) in infos.iter().enumerate() {
            let mut f = 0;
            if info.occludes == 0x3f {
                f |= IS_OCCLUDER;
            }
            if info.always_culled {
                f |= ALWAYS_CULLED;
            }
            if !info.visible {
                f |= DRAWS_NOTHING;
            }
            self.flags[id] = f;
        }
        let n = v.blocks.len();
        self.voxel_flags.resize(n, 0);
        self.skip.clear();
        self.skip.resize(n, 0);
        let table = &*self.flags;
        for (o, &id) in self.voxel_flags.iter_mut().zip(&v.blocks) {
            *o = table[id as usize];
        }
        let f = &self.voxel_flags;
        for vy in 1..v.h.saturating_sub(1) {
            for vz in 1..W - 1 {
                let row = (vy * W + vz) * W;
                let center = &f[row..row + W];
                let (ym, yp) = (&f[row - STRIDE_Y..][..W], &f[row + STRIDE_Y..][..W]);
                let (zm, zp) = (&f[row - STRIDE_Z..][..W], &f[row + STRIDE_Z..][..W]);
                let out = &mut self.skip[row + 1..row + W - 1];
                for x in 0..W - 2 {
                    let c = center[x + 1];
                    let enclosed = c >> 1
                        & center[x]
                        & center[x + 2]
                        & ym[x + 1]
                        & yp[x + 1]
                        & zm[x + 1]
                        & zp[x + 1];
                    out[x] = (c >> 2 | enclosed) & 1;
                }
            }
        }
    }

    /// Meshes the volume. Returns the encoded (uncompressed) tile, or None if empty.
    pub fn mesh(&mut self, v: &Volume, reg: &Registry, s: &Settings) -> Option<Vec<u8>> {
        for g in &mut self.groups {
            g.clear();
        }
        self.min_qy = u16::MAX;
        self.max_qy = 0;

        let infos = reg.snapshot();
        let biomes = reg.baker.biomes.snapshot();
        self.find_skippable(v, &infos);
        self.run_top.fill(i32::MAX);
        self.open_above.fill(true);

        let origin = [v.x0 - 1, s.min_y - 1, v.z0 - 1];
        let water_tex = reg.baker.fluid_texture(false);
        let lava_tex = reg.baker.fluid_texture(true);
        let last = TILE as usize;

        // Layer by layer, top down: the volume is laid out by layer, so this
        // stays on one cache line at a time. Quads sort by column in encode().
        for vy in (1..v.h.saturating_sub(1)).rev() {
            let wy = v.y0 + vy as i32;
            for vz in 1..=last {
                let row = (vy * W + vz) * W;
                let border_row = vz == 1 || vz == last;
                for vx in 1..=last {
                    let i = row + vx;
                    // Only border columns have outward faces, so only they track
                    // the solid run that skirts hang from.
                    let border = border_row || vx == 1 || vx == last;
                    if !border && self.skip[i] != 0 {
                        continue;
                    }
                    let col = vz * W + vx;
                    let was_open = border && std::mem::replace(&mut self.open_above[col], true);
                    let id = v.blocks[i];
                    if id == AIR {
                        continue;
                    }
                    let info = infos[id as usize];
                    if !info.visible || v.cave(vx, vy, vz) {
                        continue;
                    }
                    let wx = v.x0 + vx as i32 - 1;
                    let wz = v.z0 + vz as i32 - 1;
                    let base = [wx - origin[0], wy - origin[1], wz - origin[2]];
                    let mut top = i32::MAX;
                    if border {
                        self.open_above[col] = false;
                        if was_open {
                            self.run_top[col] = wy;
                        }
                        top = self.run_top[col];
                    }

                    let quads = info.variant(wx, wy, wz);
                    let mut tint: Option<[u8; 3]> = None;
                    for q in quads {
                        if let Some(d) = q.cull {
                            if d == DOWN && wy == s.min_y {
                                continue;
                            }
                            let n = neighbor(i, d);
                            let nid = v.blocks[n];
                            let ninfo = infos[nid as usize];
                            if ninfo.occludes & (1 << opposite(d)) != 0 {
                                if border && wy >= top - (SKIRT - 1) && outward(vx, vz, d) {
                                    // Skirt: kept against a solid neighbor, lit as if open.
                                    let color = if q.tinted {
                                        tint_color(info, biomes[v.biome(vx, vy, vz) as usize])
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
                        let light = face_light(v, i, q);
                        let color = if q.tinted {
                            *tint.get_or_insert_with(|| {
                                tint_color(info, biomes[v.biome(vx, vy, vz) as usize])
                            })
                        } else {
                            [255, 255, 255]
                        };
                        let group = match q.cull {
                            Some(d)
                                if info.flags & F_LEAVES != 0
                                    && infos[v.blocks[neighbor(i, d)] as usize].flags
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
                            biomes[v.biome(vx, vy, vz) as usize].water
                        };
                        self.fluid(v, &infos, reg, i, lava, tex, base, color);
                    }
                }
            }
        }

        if !s.lod {
            self.halo_faces(v, &infos, &biomes, origin);
        }

        if self.groups.iter().all(|g| g.is_empty()) {
            return None;
        }
        Some(self.encode(reg, origin))
    }

    /// Faces of the halo blocks that point into the tile, such as a cliff that
    /// belongs to the neighboring tile. They close the gap where this tile meets
    /// a neighbor drawn at lowres; when the neighbor is drawn at hires too, its
    /// identical face covers the same pixels.
    fn halo_faces(
        &mut self,
        v: &Volume,
        infos: &[&BlockInfo],
        biomes: &[BiomeColors],
        origin: [i32; 3],
    ) {
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
                for vy in 1..v.h.saturating_sub(1) {
                    let i = v.idx(vx, vy, vz);
                    let id = v.blocks[i];
                    if id == 0 {
                        continue;
                    }
                    let info = infos[id as usize];
                    let wy = v.y0 + vy as i32;
                    if info.is_air() || v.cave(vx, vy, vz) {
                        continue;
                    }
                    let n = neighbor(i, d);
                    let ninfo = infos[v.blocks[n] as usize];
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
                            tint_color(info, biomes[v.biome(vx, vy, vz) as usize])
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

    fn emit(&mut self, q: &Quad, base: [i32; 3], color: [u8; 3], light: u8, group: usize) {
        let l = (light >> 4).max(light & 15) as usize;
        let f = q.shade * BRIGHT[l];
        let color = u32::from_le_bytes([
            (color[0] as f32 * f) as u8,
            (color[1] as f32 * f) as u8,
            (color[2] as f32 * f) as u8,
            light,
        ]);
        let qy = |o: u16| (base[1] * 64 + o as i32 - 64).clamp(0, 65535) as u16;
        self.min_qy = self.min_qy.min(qy(q.y_range.0));
        self.max_qy = self.max_qy.max(qy(q.y_range.1));
        self.groups[group].push(Rec::new(base, q.template, color));
    }

    #[allow(clippy::too_many_arguments)]
    fn fluid(
        &mut self,
        v: &Volume,
        infos: &[&BlockInfo],
        reg: &Registry,
        i: usize,
        lava: bool,
        tex: u16,
        base: [i32; 3],
        color: [u8; 3],
    ) {
        let same = |n: usize| {
            let b = infos[v.blocks[n] as usize];
            if lava {
                b.flags & crate::model::F_LAVA != 0
            } else {
                b.has_water()
            }
        };
        // Surface height of the fluid at a volume index, in ninths: full under
        // more fluid, else from its level (sources and falling fluid sit at 8/9).
        let height_at = |n: usize| -> usize {
            if same(n + STRIDE_Y) {
                return 9;
            }
            let level = infos[v.blocks[n] as usize].fluid_level as usize;
            if level == 0 || level >= 8 {
                8
            } else {
                8 - level.min(7)
            }
        };
        let up = neighbor(i, UP);
        let height = height_at(i);
        for d in 0..6u8 {
            let n = neighbor(i, d);
            // Beside lower fluid of the same kind, only the strip above it shows.
            let mut lo = 0;
            if same(n) {
                if d < 2 {
                    continue;
                }
                lo = height_at(n);
                if lo >= height {
                    continue;
                }
            }
            let ninfo = infos[v.blocks[n] as usize];
            if ninfo.occludes & (1 << opposite(d)) != 0 {
                continue;
            }
            let light = if lava {
                0xff
            } else if d == UP {
                v.light[i].max(v.light[up])
            } else {
                v.light[n].max(v.light[i])
            };
            let slot = fluid_slot(lava, d, height, lo);
            if self.fluid_quads[slot].is_none() {
                self.fluid_quads[slot] = Some(fluid_quad(reg, lava, d, height, lo, tex));
            }
            let q = self.fluid_quads[slot].take().unwrap();
            self.emit(&q, base, color, light, group_of(q.layer));
            self.fluid_quads[slot] = Some(q);
        }
    }

    fn encode(&mut self, reg: &Registry, origin: [i32; 3]) -> Vec<u8> {
        // Canonical order: by column, then top down. Ties keep emit order,
        // which is itself deterministic.
        for g in &mut self.groups {
            g.sort_by_key(|r| r.key);
        }
        // Per-tile tables, numbered in order of first use.
        self.tile_stamp += 1;
        let stamp = self.tile_stamp;
        self.color_ids.clear();
        let mut templates: Vec<u32> = Vec::new();
        let mut colors: Vec<u32> = Vec::new();
        let mut refs: Vec<Vec<(u16, u16)>> = Vec::with_capacity(GROUPS);
        for g in &self.groups {
            let mut group_refs = Vec::with_capacity(g.len());
            for r in g {
                let id = r.template as usize;
                if id >= self.template_slots.len() {
                    self.template_slots.resize(id + 1, (0, 0));
                }
                let slot = &mut self.template_slots[id];
                if slot.0 != stamp {
                    *slot = (stamp, templates.len() as u16);
                    templates.push(r.template);
                }
                let c = *self.color_ids.entry(r.color).or_insert_with(|| {
                    colors.push(r.color);
                    (colors.len() - 1) as u16
                });
                group_refs.push((slot.1, c));
            }
            refs.push(group_refs);
        }

        let quads: usize = self.groups.iter().map(|g| g.len()).sum();
        let mut out =
            Vec::with_capacity(64 + templates.len() * 42 + quads * 6 + GROUPS * 2 * W * W);
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
        for &t in &templates {
            for v in reg.baker.template(t).iter() {
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
                counts[q.z() * W + q.x()] += 1;
            }
            planes(&mut out, &counts);
            let mut steps = Vec::with_capacity(g.len());
            let mut prev = (usize::MAX, usize::MAX, 1023u16);
            for q in g {
                if (q.z(), q.x()) != (prev.0, prev.1) {
                    prev = (q.z(), q.x(), 1023);
                }
                steps.push(prev.2 - q.y());
                prev.2 = q.y();
            }
            planes(&mut out, &steps);
            planes(&mut out, &r.iter().map(|x| x.0).collect::<Vec<_>>());
            planes(&mut out, &r.iter().map(|x| x.1).collect::<Vec<_>>());
        }
        out
    }
}

/// A fluid face's slot in the bake cache: its kind, direction, and the top and
/// bottom of the face in ninths of a block.
#[inline]
fn fluid_slot(lava: bool, d: u8, height: usize, lo: usize) -> usize {
    ((lava as usize * 6 + d as usize) * 10 + height) * 10 + lo
}

fn fluid_quad(reg: &Registry, lava: bool, d: u8, height: usize, lo: usize, tex: u16) -> Quad {
    let h = height as f32 / 9.0;
    let lo = lo as f32 / 9.0;
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
    reg.baker.quad(QuadShape {
        pos,
        uv: [[0.0, top_v], [0.0, bottom_v], [1.0, bottom_v], [1.0, top_v]],
        tex,
        layer: if lava {
            Layer::Opaque
        } else {
            Layer::Translucent
        },
        cull: Some(d),
        light_dir: d,
        tinted: true,
        shade: [0.5, 1.0, 0.8, 0.8, 0.6, 0.6][d as usize],
    })
}

#[inline]
fn face_light(v: &Volume, i: usize, q: &Quad) -> u8 {
    match q.cull {
        Some(d) => v.light[neighbor(i, d)],
        None => {
            let own = v.light[i];
            let n = v.light[neighbor(i, q.light_dir)];
            ((own >> 4).max(n >> 4) << 4) | (own & 15).max(n & 15)
        }
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

/// The volume index of the voxel next to `i` in direction `d`.
#[inline]
fn neighbor(i: usize, d: u8) -> usize {
    let o = DIR_VEC[d as usize];
    i.wrapping_add_signed(
        o[0] as isize + o[2] as isize * STRIDE_Z as isize + o[1] as isize * STRIDE_Y as isize,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes a TMH2 tile back to (group, x, y, z, template, color) per quad.
    fn decode(t: &[u8]) -> Vec<(usize, u8, u16, u8, crate::model::Template, u32)> {
        let u16_at = |o: usize| u16::from_le_bytes([t[o], t[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(t[o..o + 4].try_into().unwrap());
        assert_eq!(&t[..4], b"TMH2");
        let counts: Vec<usize> = (0..4).map(|g| u32_at(20 + g * 4) as usize).collect();
        let (nt, np) = (u32_at(36) as usize, u32_at(40) as usize);
        let mut o = 44;
        let templates: Vec<crate::model::Template> = (0..nt)
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

    fn quad(reg: &Registry, dx: f32, layer: Layer, tex: u16) -> Quad {
        reg.baker.quad(QuadShape {
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
        })
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
        use crate::assets::AssetFiles;
        use crate::model::Baker;
        let reg = Registry::new(Baker::new(AssetFiles::new()));
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
            let q = quad(&reg, dx, layer, tex);
            m.emit(&q, base, [i as u8 * 40, 200, 30], 0xf3, group);
            let r = m.groups[group].last().unwrap();
            let t = *reg.baker.template(r.template);
            expected.push((group, r.x() as u8, r.y(), r.z() as u8, t, r.color));
        }
        let tile = m.encode(&reg, [0, 0, 0]);
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
