//! Hires tile meshing and the binary tile format.
//!
//! A tile covers 64×64 columns. Blocks are copied into a dense volume with a
//! one-block halo so faces on tile and chunk borders cull correctly.
//!
//! Tile format (little-endian, gzip on disk). A tile is a list of quads, and
//! each quad is an integer block anchor plus a shape ("template": four corner
//! offsets, UVs and a texture layer) and a color from small per-tile tables.
//! Neighboring faces that look the same merge into one quad that stretches its
//! template over several blocks. Quads are stored by render group and face
//! direction, so the viewer can skip whole directions that face away from the
//! camera, and coded plane by plane as byte planes of differences, which gzip
//! squeezes well. The viewer draws them straight from these records by vertex
//! pulling.
//!
//! ```text
//! 0   magic "TMH3"
//! 4   i32 origin x, y, z            block coords of anchor (0, 0, 0)
//! 16  u16 min qy, u16 max qy        bounds, 1/64 block from origin y
//! 20  4 × 7 × u32 quad count        per group (opaque, cutout, leaf interior,
//!                                   translucent) and direction (down, up, north,
//!                                   south, west, east, other)
//! 132 u32 templates T, u32 colors P
//! 140 22 × T × u16 templates        one plane per field, low bytes then high:
//!                                   4 × corner (x, y, z) in 1/64 block + 64,
//!                                   4 × (u, v) in 1/4096 texture, layer, stretch
//!     P × 3 colors                  r, g, b planes
//!     per stream, n quads sorted by plane (the block coordinate along the face
//!     normal), then v, then u, the axes the quad stretches along:
//!       n × plane step              u16 planes; plane minus the previous plane
//!       n × v step                  u16 planes; v minus the previous v in this plane
//!       n × u step                  u8; u minus the previous u in this row
//!       n × w - 1, n × h - 1        u8, blocks along u and v
//!       n × template index          u16 planes
//!       n × color index             u16 planes
//! ```
//!
//! The u and v of a stream are x and y (north, south), z and y (west, east), or
//! x and z (everything else), with y, x, z or y left as the plane. Stretch: bits
//! 0-3 mark the corners at the far end of u, bits 4-7 those at the far end of v,
//! and bits 8-15 hold the UV steps per block along u and then v, as (u, v) pairs
//! of 2-bit codes (0 back a texture, 1 none, 2 forward).
//!
//! Leaf faces that touch other leaves form their own group, so the viewer can
//! skip them at a distance, where they can't be seen.

use crate::biome::BiomeColors;
use crate::model::{
    AIR, BlockInfo, DIR_VEC, DOWN, EAST, F_CULL_SAME, F_LEAVES, Layer, NORTH, Quad, QuadShape,
    Registry, SOUTH, Tint, UP, WEST, merge_axes, opposite,
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
/// Face directions a stream can hold: six block faces and everything else.
const DIRS: usize = 7;
/// The group for leaf faces that touch other leaves.
const LEAF_INTERIOR: usize = 2;
/// The most blocks a merged quad spans along one axis.
const MAX_RUN: u16 = 64;
/// Tile tables never grow past these; see [`Mesher::encode`].
const MAX_TEMPLATES: usize = 1 << 13;
const MAX_COLORS: usize = 1 << 13;

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
    /// A voxel's light ORed with its six neighbors', for finding caves.
    lit: Vec<u8>,
    caves_below: i32,
    /// Removes unlit faces under the surface; off when cave removal is off.
    remove_dark: bool,
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
            remove_dark: false,
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
        self.caves_below = settings.remove_caves_below_y;
        self.remove_dark = settings.remove_caves_below_y > settings.min_y;
        self.build_lit();
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
        self.remove_dark = false;
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

    /// For each voxel, ORs its light with its neighbors', a row at a time so it
    /// vectorizes. Halo columns have no neighbor outside the volume on that side.
    fn build_lit(&mut self) {
        let layers = self.h - 1;
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
        wy < self.caves_below
            && self.under_surface(vx, vz, wy)
            && self.lit[self.idx(vx, vy, vz)] >> 4 == 0
    }

    /// Whether a block is below the surface and nothing around it has any light:
    /// not sky, not a torch. Nobody can see it, at any height.
    #[inline]
    pub fn in_the_dark(&self, vx: usize, vy: usize, vz: usize) -> bool {
        self.remove_dark
            && self.under_surface(vx, vz, self.y0 + vy as i32)
            && self.lit[self.idx(vx, vy, vz)] == 0
    }

    /// Whether a face this dark is one nobody can see. A seabed's stays: water
    /// doesn't block the view.
    #[inline]
    fn dark_face(&self, light: u8, vx: usize, vy: usize, vz: usize) -> bool {
        light == 0 && self.remove_dark && self.under_surface(vx, vz, self.y0 + vy as i32)
    }

    #[inline]
    fn under_surface(&self, vx: usize, vz: usize, wy: i32) -> bool {
        let floor = self.floor[vz * W + vx];
        floor != i32::MIN && wy < floor - 1
    }
}

/// One quad in a tile: a sort key (column, then top down), its template and color.
#[derive(Clone, Copy)]
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

/// A quad of a stream, in the field order streams sort by.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Cell {
    plane: u16,
    v: u16,
    u: u16,
    template: u32,
    color: u32,
    w: u16,
    h: u16,
}

/// A run of equal faces along u, growing along v.
#[derive(Clone, Copy)]
struct Run {
    template: u32,
    color: u32,
    u: u16,
    w: u16,
    v0: u16,
    h: u16,
}

/// Per-state flags read for every voxel.
const IS_OCCLUDER: u8 = 1;
const ALWAYS_CULLED: u8 = 2;
const DRAWS_NOTHING: u8 = 4;

pub struct Mesher {
    groups: [Vec<Rec>; GROUPS],
    /// The final quads, merged where possible.
    streams: [[Vec<Cell>; DIRS]; GROUPS],
    /// Faces waiting to merge, per direction: sorted, they sit row by row.
    faces: [Vec<Cell>; DIRS],
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
    template_ids: FxHashMap<u32, u16>,
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
            streams: Default::default(),
            faces: Default::default(),
            min_qy: u16::MAX,
            max_qy: 0,
            flags: vec![0; 65536].into_boxed_slice().try_into().unwrap(),
            voxel_flags: Vec::new(),
            skip: Vec::new(),
            run_top: vec![i32::MAX; W * W],
            open_above: vec![true; W * W],
            fluid_quads: vec![None; 2 * 6 * 10 * 10],
            template_ids: FxHashMap::default(),
            color_ids: FxHashMap::default(),
        }
    }

    /// Marks voxels the mesher can skip: those that draw nothing, and those whose
    /// faces are all culled by six occluding neighbors. A row at a time, so the
    /// neighbor test vectorizes.
    fn find_skippable(&mut self, v: &Volume, infos: &[&BlockInfo], lod: bool) {
        for (id, info) in infos.iter().enumerate() {
            let mut f = 0;
            if info.occludes == 0x3f || (lod && info.flags & F_LEAVES != 0) {
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
        self.find_skippable(v, &infos, s.lod);
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

                    // At LOD scale a voxel is a few pixels: which way its texture
                    // is turned can't be seen, and one variant merges better.
                    let quads = if s.lod {
                        info.first_variant()
                    } else {
                        info.variant(wx, wy, wz)
                    };
                    let solid_leaves = s.lod && info.flags & F_LEAVES != 0;
                    let mut tint: Option<[u8; 3]> = None;
                    for q in quads {
                        if let Some(d) = q.cull {
                            if d == DOWN && wy == s.min_y {
                                continue;
                            }
                            let n = neighbor(i, d);
                            let nid = v.blocks[n];
                            let ninfo = infos[nid as usize];
                            // LOD leaves are solid cubes: they draw with no alpha
                            // and hide what's behind them.
                            if ninfo.occludes & (1 << opposite(d)) != 0
                                || (s.lod && ninfo.flags & F_LEAVES != 0)
                            {
                                if border && wy >= top - (SKIRT - 1) && outward(vx, vz, d) {
                                    // Skirt: kept against a solid neighbor, lit as if open.
                                    let color = if q.tinted {
                                        tint_color(info, biomes[v.biome(vx, vy, vz) as usize])
                                    } else {
                                        [255, 255, 255]
                                    };
                                    let group = if solid_leaves { 0 } else { group_of(q.layer) };
                                    self.emit(q, base, color, 0xf0, group);
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
                        if v.dark_face(light, vx, vy, vz) {
                            continue;
                        }
                        let color = if q.tinted {
                            *tint.get_or_insert_with(|| {
                                tint_color(info, biomes[v.biome(vx, vy, vz) as usize])
                            })
                        } else {
                            [255, 255, 255]
                        };
                        let group = match q.cull {
                            _ if solid_leaves => 0,
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
                        let dark = |light| v.dark_face(light, vx, vy, vz);
                        self.fluid(v, &infos, reg, i, lava, tex, base, color, &dark);
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
                    if v.dark_face(v.light[n], vx, vy, vz) {
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
            0,
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
        dark: &dyn Fn(u8) -> bool,
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
            if dark(light) {
                continue;
            }
            let slot = fluid_slot(lava, d, height, lo);
            if self.fluid_quads[slot].is_none() {
                self.fluid_quads[slot] = Some(fluid_quad(reg, lava, d, height, lo, tex));
            }
            let q = self.fluid_quads[slot].take().unwrap();
            self.emit(&q, base, color, light, group_of(q.layer));
            self.fluid_quads[slot] = Some(q);
        }
    }

    /// Merges equal neighboring faces in group `g` into `streams[g]`: first
    /// along u within each row, then equal runs across consecutive rows.
    fn merge_group(&mut self, reg: &Registry, g: usize) {
        for d in 0..DIRS {
            self.streams[g][d].clear();
            self.faces[d].clear();
        }
        for r in &self.groups[g] {
            let info = reg.baker.template_info(r.template);
            let d = info.dir as usize;
            let p = [r.x() as u16, r.y(), r.z() as u16];
            let (u, v) = merge_axes(info.dir);
            let cell = Cell {
                plane: p[3 - u - v],
                v: p[v],
                u: p[u],
                template: r.template,
                color: r.color,
                w: 1,
                h: 1,
            };
            if info.merge_u || info.merge_v {
                self.faces[d].push(cell);
            } else {
                self.streams[g][d].push(cell);
            }
        }
        let shape = |c: &Run| (c.template, c.color, c.u);
        let (mut prev, mut cur): (Vec<Run>, Vec<Run>) = (Vec::new(), Vec::new());
        for d in 0..DIRS {
            let mut faces = std::mem::take(&mut self.faces[d]);
            // Row by row, with equal shapes next to each other.
            faces.sort_unstable_by_key(|c| (c.plane, c.v, c.template, c.color, c.u));
            let out = &mut self.streams[g][d];
            let finish = |plane: u16, r: &Run, out: &mut Vec<Cell>| {
                out.push(Cell {
                    plane,
                    v: r.v0,
                    u: r.u,
                    template: r.template,
                    color: r.color,
                    w: r.w,
                    h: r.h,
                });
            };
            let mut i = 0;
            while i < faces.len() {
                let plane = faces[i].plane;
                prev.clear();
                let mut prev_v = u16::MAX - 1;
                while i < faces.len() && faces[i].plane == plane {
                    let v = faces[i].v;
                    cur.clear();
                    while i < faces.len() && faces[i].plane == plane && faces[i].v == v {
                        let c = faces[i];
                        let info = reg.baker.template_info(c.template);
                        i += 1;
                        let mut w = 1;
                        if info.merge_u {
                            while i < faces.len()
                                && (faces[i].plane, faces[i].v) == (plane, v)
                                && (faces[i].template, faces[i].color) == (c.template, c.color)
                                && faces[i].u == c.u + w
                                && w < MAX_RUN
                            {
                                w += 1;
                                i += 1;
                            }
                        }
                        cur.push(Run {
                            template: c.template,
                            color: c.color,
                            u: c.u,
                            w,
                            v0: v,
                            h: 1,
                        });
                    }
                    // Both rows are sorted by shape, so they can be walked together.
                    // Runs that match one in the row before grow it; the others
                    // are finished.
                    let touching = v == prev_v + 1;
                    let mut p = 0;
                    for run in &mut cur {
                        while touching && p < prev.len() && shape(&prev[p]) < shape(run) {
                            finish(plane, &prev[p], out);
                            p += 1;
                        }
                        if touching
                            && p < prev.len()
                            && shape(&prev[p]) == shape(run)
                            && prev[p].w == run.w
                            && prev[p].h < MAX_RUN
                            && reg.baker.template_info(run.template).merge_v
                        {
                            run.v0 = prev[p].v0;
                            run.h = prev[p].h + 1;
                            p += 1;
                        }
                    }
                    for r in &prev[p..] {
                        finish(plane, r, out);
                    }
                    std::mem::swap(&mut prev, &mut cur);
                    prev_v = v;
                }
                for r in &prev {
                    finish(plane, r, out);
                }
            }
            self.faces[d] = faces;
        }
    }

    fn encode(&mut self, reg: &Registry, origin: [i32; 3]) -> Vec<u8> {
        for g in 0..GROUPS {
            self.merge_group(reg, g);
        }
        // Per-tile tables, numbered in order of first use.
        self.template_ids.clear();
        self.color_ids.clear();
        let mut templates: Vec<u32> = Vec::new();
        let mut colors: Vec<u32> = Vec::new();
        let mut quads = 0;
        for stream in self.streams.iter_mut().flatten() {
            // Sorting by every field makes the order canonical.
            stream.sort_unstable();
            quads += stream.len();
            for r in stream.iter() {
                self.template_ids.entry(r.template).or_insert_with(|| {
                    templates.push(r.template);
                    (templates.len() - 1) as u16
                });
                self.color_ids.entry(r.color).or_insert_with(|| {
                    colors.push(r.color);
                    (colors.len() - 1) as u16
                });
            }
        }
        // The quad records have 13 bits for each table index. A tile that needs
        // more is beyond anything real terrain produces; it draws wrong colors
        // or shapes rather than failing the whole render.
        if templates.len() > MAX_TEMPLATES || colors.len() > MAX_COLORS {
            tracing::warn!(
                "tile at {origin:?} has {} templates and {} colors; some quads will look wrong",
                templates.len(),
                colors.len()
            );
        }

        let mut out =
            Vec::with_capacity(140 + templates.len() * 44 + colors.len() * 3 + quads * 10);
        out.extend_from_slice(b"TMH3");
        for o in origin {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out.extend_from_slice(&self.min_qy.to_le_bytes());
        out.extend_from_slice(&self.max_qy.to_le_bytes());
        for stream in self.streams.iter().flatten() {
            out.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        }
        out.extend_from_slice(&(templates.len() as u32).to_le_bytes());
        out.extend_from_slice(&(colors.len() as u32).to_le_bytes());
        let field = |id: u32, k: usize| {
            if k < 21 {
                reg.baker.template(id)[k]
            } else {
                reg.baker.template_info(id).stretch
            }
        };
        for k in 0..22 {
            out.extend(templates.iter().map(|&t| field(t, k) as u8));
            out.extend(templates.iter().map(|&t| (field(t, k) >> 8) as u8));
        }
        for channel in 0..3 {
            out.extend(colors.iter().map(|c| c.to_le_bytes()[channel]));
        }
        let planes = |out: &mut Vec<u8>, vals: &mut dyn Iterator<Item = u16>, n: usize| {
            let start = out.len();
            out.resize(start + 2 * n, 0);
            let (lo, hi) = out[start..].split_at_mut(n);
            for (i, v) in vals.enumerate() {
                lo[i] = v as u8;
                hi[i] = (v >> 8) as u8;
            }
        };
        for stream in self.streams.iter().flatten() {
            let n = stream.len();
            let mut last = 0;
            planes(
                &mut out,
                &mut stream.iter().map(|q| {
                    let d = q.plane - last;
                    last = q.plane;
                    d
                }),
                n,
            );
            let mut last = (u16::MAX, 0);
            planes(
                &mut out,
                &mut stream.iter().map(|q| {
                    let d = if q.plane == last.0 { q.v - last.1 } else { q.v };
                    last = (q.plane, q.v);
                    d
                }),
                n,
            );
            let mut last = (u16::MAX, u16::MAX, 0);
            out.extend(stream.iter().map(|q| {
                let d = if (q.plane, q.v) == (last.0, last.1) {
                    q.u - last.2
                } else {
                    q.u
                };
                last = (q.plane, q.v, q.u);
                d as u8
            }));
            out.extend(stream.iter().map(|q| (q.w - 1) as u8));
            out.extend(stream.iter().map(|q| (q.h - 1) as u8));
            planes(
                &mut out,
                &mut stream.iter().map(|q| self.template_ids[&q.template]),
                n,
            );
            planes(
                &mut out,
                &mut stream.iter().map(|q| self.color_ids[&q.color]),
                n,
            );
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
    use crate::assets::AssetFiles;
    use crate::model::{Baker, Template};

    /// A quad decoded back from a tile.
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Decoded {
        group: usize,
        dir: usize,
        x: u16,
        y: u16,
        z: u16,
        w: u16,
        h: u16,
        template: Template,
        stretch: u16,
        color: u32,
    }

    fn decode(t: &[u8]) -> Vec<Decoded> {
        let u32_at = |o: usize| u32::from_le_bytes(t[o..o + 4].try_into().unwrap());
        assert_eq!(&t[..4], b"TMH3");
        let (nt, np) = (u32_at(132) as usize, u32_at(136) as usize);
        let mut o = 140;
        let field = |o: usize, i: usize| t[o + i] as u16 | (t[o + nt + i] as u16) << 8;
        let templates: Vec<(Template, u16)> = (0..nt)
            .map(|i| {
                (
                    std::array::from_fn(|k| field(o + k * 2 * nt, i)),
                    field(o + 21 * 2 * nt, i),
                )
            })
            .collect();
        o += nt * 44;
        let colors: Vec<u32> = (0..np)
            .map(|i| u32::from_le_bytes([t[o + i], t[o + np + i], t[o + 2 * np + i], 0]))
            .collect();
        o += np * 3;
        let mut out = Vec::new();
        for s in 0..GROUPS * DIRS {
            let n = u32_at(20 + s * 4) as usize;
            let (planes, vs, us, ws, hs, tpls, cols) = (
                o,
                o + 2 * n,
                o + 4 * n,
                o + 5 * n,
                o + 6 * n,
                o + 7 * n,
                o + 9 * n,
            );
            let plane = |at: usize, i: usize| t[at + i] as u16 | (t[at + n + i] as u16) << 8;
            let (dir, mut pl, mut v, mut u) = (s % DIRS, 0u16, 0u16, 0u16);
            for i in 0..n {
                let (dp, dv) = (plane(planes, i), plane(vs, i));
                pl += dp;
                let new_plane = dp != 0 || i == 0;
                v = if new_plane { dv } else { v + dv };
                let new_row = new_plane || dv != 0;
                let du = t[us + i] as u16;
                u = if new_row { du } else { u + du };
                let (a_u, a_v) = merge_axes(dir as u8);
                let mut p = [0u16; 3];
                p[a_u] = u;
                p[a_v] = v;
                p[3 - a_u - a_v] = pl;
                let (template, stretch) = templates[plane(tpls, i) as usize];
                out.push(Decoded {
                    group: s / DIRS,
                    dir,
                    x: p[0],
                    y: p[1],
                    z: p[2],
                    w: t[ws + i] as u16 + 1,
                    h: t[hs + i] as u16 + 1,
                    template,
                    stretch,
                    color: colors[plane(cols, i) as usize],
                });
            }
            o += 11 * n;
        }
        assert_eq!(o, t.len());
        out
    }

    fn registry() -> Registry {
        Registry::new(Baker::new(AssetFiles::new()))
    }

    /// A one-block face pointing up, or a skewed quad that can't merge.
    fn quad(reg: &Registry, skew: f32, layer: Layer, tex: u16) -> Quad {
        reg.baker.quad(QuadShape {
            pos: [
                [skew, 1.0, 0.0],
                [skew, 1.0, 1.0],
                [1.0 + skew * 2.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            uv: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
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
        let reg = registry();
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
        // East faces sit in column 64 and merge down the wall into strips.
        let quads = decode(&tile);
        let east_covers = |y: u16| {
            quads
                .iter()
                .any(|q| q.dir == EAST as usize && q.x == 64 && q.y <= y + 1 && y + 1 < q.y + q.h)
        };
        assert!(
            east_covers(5),
            "ground under the roof has no skirt at its top"
        );
        assert!(east_covers(4), "the skirt covers two voxels");
        assert!(!east_covers(3), "faces below the skirt still cull");
    }

    #[test]
    fn tiles_round_trip() {
        let reg = registry();
        let mut m = Mesher::new();
        let mut expected = Vec::new();
        // Quads in several columns, stacked in one column, across all groups,
        // and with shapes that can't merge.
        let cases = [
            ([0, 1, 0], 0.25, Layer::Opaque, 3, 0),
            ([65, 385, 65], 0.5, Layer::Cutout, 4, 1),
            ([10, 70, 20], 0.25, Layer::Opaque, 3, 0),
            ([10, 90, 20], 0.5, Layer::Translucent, 7, 3),
            ([10, 70, 20], 0.5, Layer::Cutout, 5, LEAF_INTERIOR),
            ([10, 71, 20], 0.25, Layer::Opaque, 3, 0),
            // Worlds can be up to 1000 blocks tall.
            ([3, 1001, 4], 0.25, Layer::Opaque, 3, 0),
        ];
        for (i, (base, skew, layer, tex, group)) in cases.into_iter().enumerate() {
            let q = quad(&reg, skew, layer, tex);
            m.emit(&q, base, [i as u8 * 40, 200, 30], 0xf3, group);
            let r = m.groups[group].last().unwrap();
            let t = reg.baker.template(r.template);
            expected.push(Decoded {
                group,
                dir: UP as usize,
                x: base[0] as u16,
                y: base[1] as u16,
                z: base[2] as u16,
                w: 1,
                h: 1,
                template: *t,
                stretch: 0,
                color: r.color,
            });
        }
        let tile = m.encode(&reg, [0, 0, 0]);
        let mut got = decode(&tile);
        got.sort();
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn equal_faces_merge_into_one_quad() {
        let reg = registry();
        let mut m = Mesher::new();
        let q = quad(&reg, 0.0, Layer::Opaque, 3);
        // A 3×2 patch, plus a separate face of another color next to it.
        for x in 0..3 {
            for z in 0..2 {
                m.emit(&q, [10 + x, 5, 20 + z], [255, 255, 255], 0xf0, 0);
            }
        }
        m.emit(&q, [13, 5, 20], [1, 2, 3], 0xf0, 0);
        let quads = decode(&m.encode(&reg, [0, 0, 0]));
        assert_eq!(quads.len(), 2);
        let big = quads.iter().find(|q| q.x == 10).unwrap();
        assert_eq!((big.y, big.z, big.w, big.h), (5, 20, 3, 2));
        assert_eq!(quads.iter().find(|q| q.x == 13).unwrap().w, 1);
        // Corners 2 and 3 are at far x, 1 and 2 at far z, and the texture
        // advances along x and along z, each in one texture axis.
        assert_eq!(big.stretch & 0xf, 0b1100);
        assert_eq!(big.stretch >> 4 & 0xf, 0b0110);
        assert_eq!(big.stretch >> 8, 0b10_01_01_10);
    }

    #[test]
    fn rotated_textures_stretch_backwards() {
        let reg = registry();
        let mut m = Mesher::new();
        // A face whose texture runs backwards along x.
        let q = reg.baker.quad(QuadShape {
            pos: [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            uv: [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            tex: 3,
            layer: Layer::Opaque,
            cull: None,
            light_dir: UP,
            tinted: false,
            shade: 1.0,
        });
        m.emit(&q, [1, 1, 1], [9, 9, 9], 0xf0, 0);
        m.emit(&q, [2, 1, 1], [9, 9, 9], 0xf0, 0);
        let quads = decode(&m.encode(&reg, [0, 0, 0]));
        assert_eq!((quads.len(), quads[0].w), (1, 2));
        assert_eq!(quads[0].stretch >> 8 & 3, 0, "u runs back along x");
    }

    #[test]
    fn every_tile_stretches_to_the_right_size() {
        // Far corners stretch by (w - 1) blocks along u and (h - 1) along v; a
        // merged quad covers the same area as the faces that went into it.
        let reg = registry();
        let q = quad(&reg, 0.0, Layer::Opaque, 3);
        let mut m = Mesher::new();
        for x in 0..MAX_RUN as i32 + 3 {
            m.emit(&q, [x % 66, 2, x / 66 * 2], [1, 1, 1], 0xf0, 0);
        }
        let quads = decode(&m.encode(&reg, [0, 0, 0]));
        let area: u32 = quads.iter().map(|q| q.w as u32 * q.h as u32).sum();
        assert_eq!(area, MAX_RUN as u32 + 3);
        assert!(quads.iter().all(|q| q.w <= MAX_RUN && q.h <= MAX_RUN));
    }
}
