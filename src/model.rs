//! Block state → baked quads.
//!
//! Resolves blockstate JSON (variants and multipart), model parent chains and
//! texture references, then bakes elements into quads in block-local space.

use crate::assets::{AssetFiles, split_id};
use crate::biome::Biomes;
use crate::textures::{Alpha, MISSING, Textures};
use dashmap::DashMap;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub const DOWN: u8 = 0;
pub const UP: u8 = 1;
pub const NORTH: u8 = 2;
pub const SOUTH: u8 = 3;
pub const WEST: u8 = 4;
pub const EAST: u8 = 5;
pub const DIR_VEC: [[i32; 3]; 6] = [
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1],
    [-1, 0, 0],
    [1, 0, 0],
];

#[inline]
pub fn opposite(d: u8) -> u8 {
    d ^ 1
}

fn parse_dir(s: &str) -> Option<u8> {
    Some(match s {
        "down" | "bottom" => DOWN,
        "up" | "top" => UP,
        "north" => NORTH,
        "south" => SOUTH,
        "west" => WEST,
        "east" => EAST,
        _ => return None,
    })
}

fn nearest_dir(n: [f32; 3]) -> u8 {
    let (ax, ay, az) = (n[0].abs(), n[1].abs(), n[2].abs());
    if ay >= ax && ay >= az {
        if n[1] < 0.0 { DOWN } else { UP }
    } else if ax >= az {
        if n[0] < 0.0 { WEST } else { EAST }
    } else if n[2] < 0.0 {
        NORTH
    } else {
        SOUTH
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layer {
    Opaque = 0,
    Cutout = 1,
    Translucent = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tint {
    Grass,
    Foliage,
    DryFoliage,
    Water,
    Fixed([u8; 3]),
}

#[derive(Clone, Debug)]
pub struct Quad {
    /// Corner positions in block units (usually 0..1, may extend past the block).
    pub pos: [[f32; 3]; 4],
    /// Texture coordinates in texture units (0..1 spans one texture).
    pub uv: [[f32; 2]; 4],
    pub tex: u16,
    pub layer: Layer,
    pub cull: Option<u8>,
    /// Direction to sample light from, the face normal snapped to an axis.
    pub light_dir: u8,
    pub tinted: bool,
    pub shade: f32,
}

pub struct Variant {
    pub weight: u32,
    pub quads: Vec<Quad>,
}

pub const F_AIR: u8 = 1;
pub const F_WATER: u8 = 2;
pub const F_LAVA: u8 = 4;
pub const F_WATERLOGGED: u8 = 8;
/// Culls faces against the same block (glass, ice).
pub const F_CULL_SAME: u8 = 16;
pub const F_UNKNOWN: u8 = 32;

pub struct BlockInfo {
    pub name: String,
    pub variants: Vec<Variant>,
    pub total_weight: u32,
    /// Bit `d` set: the block fully covers its `d` face with opaque texture.
    pub occludes: u8,
    pub tint: Tint,
    pub flags: u8,
    pub fluid_level: u8,
    /// Untinted average color seen from above, for the lowres layer.
    pub top_color: [u8; 3],
    pub top_tinted: bool,
}

impl BlockInfo {
    #[inline]
    pub fn is_air(&self) -> bool {
        self.flags & F_AIR != 0
    }
    #[inline]
    pub fn has_water(&self) -> bool {
        self.flags & (F_WATER | F_WATERLOGGED) != 0
    }
    #[inline]
    pub fn is_fluid(&self) -> bool {
        self.flags & (F_WATER | F_LAVA) != 0
    }
    pub fn visible(&self) -> bool {
        !self.is_air()
            && (self.variants.iter().any(|v| !v.quads.is_empty())
                || self.flags & (F_WATER | F_LAVA | F_WATERLOGGED) != 0)
    }
    /// Picks a weighted variant deterministically from a block position.
    #[inline]
    pub fn variant(&self, x: i32, y: i32, z: i32) -> &[Quad] {
        match self.variants.len() {
            0 => &[],
            1 => &self.variants[0].quads,
            _ => {
                let h = (x as u32).wrapping_mul(3129871)
                    ^ (z as u32).wrapping_mul(116129781)
                    ^ (y as u32);
                let h = h.wrapping_mul(h.wrapping_mul(42317861).wrapping_add(11));
                let mut r = (h >> 16) % self.total_weight.max(1);
                for v in &self.variants {
                    if r < v.weight {
                        return &v.quads;
                    }
                    r -= v.weight;
                }
                &self.variants[0].quads
            }
        }
    }
}

#[derive(Deserialize)]
struct RawModel {
    parent: Option<String>,
    #[serde(default)]
    textures: HashMap<String, Value>,
    elements: Option<Vec<RawElement>>,
}

#[derive(Deserialize)]
struct RawElement {
    from: [f32; 3],
    to: [f32; 3],
    rotation: Option<RawRotation>,
    #[serde(default = "yes")]
    shade: bool,
    #[serde(default)]
    faces: HashMap<String, RawFace>,
}

fn yes() -> bool {
    true
}

#[derive(Deserialize)]
struct RawRotation {
    #[serde(default = "center")]
    origin: [f32; 3],
    axis: Option<String>,
    angle: Option<f32>,
    #[serde(default)]
    rescale: bool,
    x: Option<f32>,
    y: Option<f32>,
    z: Option<f32>,
}

fn center() -> [f32; 3] {
    [8.0, 8.0, 8.0]
}

#[derive(Deserialize)]
struct RawFace {
    uv: Option<[f32; 4]>,
    texture: String,
    cullface: Option<String>,
    #[serde(default)]
    rotation: i32,
    tintindex: Option<i32>,
}

struct TexRef {
    sprite: String,
    force_translucent: bool,
}

/// Shared asset state used to bake block states.
pub struct Baker {
    files: AssetFiles,
    pub textures: Textures,
    pub biomes: Biomes,
    blockstates: DashMap<String, Option<Arc<Value>>>,
    models: DashMap<String, Option<Arc<RawModel>>>,
}

// Resolved models are only used while baking, so plain borrowed Arcs suffice.
struct Resolved {
    textures: HashMap<String, Value>,
    elements: Option<Arc<RawModel>>,
}

impl Baker {
    pub fn new(files: AssetFiles) -> Self {
        let textures = Textures::load(&files);
        let biomes = Biomes::load(&files);
        Baker {
            files,
            textures,
            biomes,
            blockstates: DashMap::new(),
            models: DashMap::new(),
        }
    }

    fn blockstate(&self, name: &str) -> Option<Arc<Value>> {
        if let Some(v) = self.blockstates.get(name) {
            return v.clone();
        }
        let (ns, p) = split_id(name);
        let v = self
            .files
            .get(&format!("assets/{ns}/blockstates/{p}.json"))
            .and_then(|b| serde_json::from_slice(b).ok())
            .map(Arc::new);
        self.blockstates.insert(name.to_owned(), v.clone());
        v
    }

    fn raw_model(&self, id: &str) -> Option<Arc<RawModel>> {
        let (ns, p) = split_id(id);
        let key = format!("{ns}:{p}");
        if let Some(v) = self.models.get(&key) {
            return v.clone();
        }
        let v = self
            .files
            .get(&format!("assets/{ns}/models/{p}.json"))
            .and_then(|b| serde_json::from_slice::<RawModel>(b).ok())
            .map(Arc::new);
        self.models.insert(key, v.clone());
        v
    }

    fn resolve(&self, id: &str) -> Option<Resolved> {
        let mut chain = Vec::new();
        let mut cur = Some(id.to_owned());
        while let Some(m) = cur {
            if chain.len() > 32 || m.starts_with("builtin/") {
                break;
            }
            let Some(raw) = self.raw_model(&m) else { break };
            cur = raw.parent.clone();
            chain.push(raw);
        }
        if chain.is_empty() {
            return None;
        }
        let mut textures = HashMap::new();
        for m in chain.iter().rev() {
            for (k, v) in &m.textures {
                textures.insert(k.clone(), v.clone());
            }
        }
        let elements = chain.iter().find(|m| m.elements.is_some()).cloned();
        Some(Resolved { textures, elements })
    }

    fn texture(&self, textures: &HashMap<String, Value>, r: &str) -> TexRef {
        let mut force = false;
        let mut cur = r.to_owned();
        for _ in 0..16 {
            let Some(key) = cur.strip_prefix('#') else {
                break;
            };
            match textures.get(key) {
                Some(Value::String(s)) => cur = s.clone(),
                Some(Value::Object(o)) => {
                    force |= o
                        .get("force_translucent")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    cur = o
                        .get("sprite")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned();
                }
                _ => {
                    cur.clear();
                    break;
                }
            }
        }
        TexRef {
            sprite: cur,
            force_translucent: force,
        }
    }

    fn bake_model(&self, id: &str, rx: i32, ry: i32, uvlock: bool, out: &mut Vec<Quad>) {
        let Some(m) = self.resolve(id) else { return };
        let Some(holder) = &m.elements else { return };
        for el in holder.elements.as_ref().unwrap() {
            for (face_name, face) in &el.faces {
                let Some(dir) = parse_dir(face_name) else {
                    continue;
                };
                if let Some(q) = self.bake_face(&m.textures, el, dir, face, rx, ry, uvlock) {
                    out.push(q);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn bake_face(
        &self,
        textures: &HashMap<String, Value>,
        el: &RawElement,
        dir: u8,
        face: &RawFace,
        rx: i32,
        ry: i32,
        uvlock: bool,
    ) -> Option<Quad> {
        let [x0, y0, z0] = el.from;
        let [x1, y1, z1] = el.to;
        let mut pos: [[f32; 3]; 4] = match dir {
            DOWN => [[x0, y0, z1], [x0, y0, z0], [x1, y0, z0], [x1, y0, z1]],
            UP => [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]],
            NORTH => [[x1, y1, z0], [x1, y0, z0], [x0, y0, z0], [x0, y1, z0]],
            SOUTH => [[x0, y1, z1], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1]],
            WEST => [[x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1]],
            _ => [[x1, y1, z1], [x1, y0, z1], [x1, y0, z0], [x1, y1, z0]],
        };
        let uvr = face.uv.unwrap_or(match dir {
            DOWN => [x0, 16.0 - z1, x1, 16.0 - z0],
            UP => [x0, z0, x1, z1],
            NORTH => [16.0 - x1, 16.0 - y1, 16.0 - x0, 16.0 - y0],
            SOUTH => [x0, 16.0 - y1, x1, 16.0 - y0],
            WEST => [z0, 16.0 - y1, z1, 16.0 - y0],
            _ => [16.0 - z1, 16.0 - y1, 16.0 - z0, 16.0 - y0],
        });
        let corners = [
            [uvr[0], uvr[1]],
            [uvr[0], uvr[3]],
            [uvr[2], uvr[3]],
            [uvr[2], uvr[1]],
        ];
        let shift = (face.rotation.rem_euclid(360) / 90) as usize;
        let mut uv = [[0f32; 2]; 4];
        for (i, c) in uv.iter_mut().enumerate() {
            *c = corners[(i + shift) % 4];
        }

        // Element rotation.
        let mut rotated = false;
        if let Some(r) = &el.rotation {
            let o = r.origin;
            if let (Some(axis), Some(angle)) = (&r.axis, r.angle) {
                if angle != 0.0 {
                    rotated = true;
                    let a = angle.to_radians();
                    let (s, c) = a.sin_cos();
                    let scale = if r.rescale {
                        1.0 / c.abs().max(1e-4)
                    } else {
                        1.0
                    };
                    for p in pos.iter_mut() {
                        let (dx, dy, dz) = (p[0] - o[0], p[1] - o[1], p[2] - o[2]);
                        let (nx, ny, nz) = match axis.as_str() {
                            "x" => (dx, (dy * c - dz * s) * scale, (dy * s + dz * c) * scale),
                            "y" => ((dx * c + dz * s) * scale, dy, (-dx * s + dz * c) * scale),
                            _ => ((dx * c - dy * s) * scale, (dx * s + dy * c) * scale, dz),
                        };
                        *p = [nx + o[0], ny + o[1], nz + o[2]];
                    }
                }
            } else if r.x.is_some() || r.y.is_some() || r.z.is_some() {
                rotated = true;
                for p in pos.iter_mut() {
                    let mut d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
                    d = rot_x(d, r.x.unwrap_or(0.0));
                    d = rot_y(d, r.y.unwrap_or(0.0));
                    d = rot_z(d, r.z.unwrap_or(0.0));
                    *p = [d[0] + o[0], d[1] + o[1], d[2] + o[2]];
                }
            }
        }

        // Variant rotation around the block center: x first, then y.
        if rx != 0 || ry != 0 {
            for p in pos.iter_mut() {
                let mut d = [p[0] - 8.0, p[1] - 8.0, p[2] - 8.0];
                d = rot_x(d, -(rx as f32));
                d = rot_y(d, -(ry as f32));
                *p = [
                    round_eps(d[0] + 8.0),
                    round_eps(d[1] + 8.0),
                    round_eps(d[2] + 8.0),
                ];
            }
        }

        let n = normal(&pos);
        let final_dir = nearest_dir(n);
        if uvlock && (rx != 0 || ry != 0) && !rotated {
            for (i, p) in pos.iter().enumerate() {
                uv[i] = match final_dir {
                    DOWN => [p[0], 16.0 - p[2]],
                    UP => [p[0], p[2]],
                    NORTH => [16.0 - p[0], 16.0 - p[1]],
                    SOUTH => [p[0], 16.0 - p[1]],
                    WEST => [p[2], 16.0 - p[1]],
                    _ => [16.0 - p[2], 16.0 - p[1]],
                };
            }
        }

        let cull = face.cullface.as_deref().and_then(parse_dir).map(|c| {
            let v = DIR_VEC[c as usize];
            let mut d = [v[0] as f32, v[1] as f32, v[2] as f32];
            d = rot_x(d, -(rx as f32));
            d = rot_y(d, -(ry as f32));
            nearest_dir(d)
        });

        let t = self.texture(textures, &face.texture);
        let (tex, layer) = match self.textures.get(&t.sprite) {
            Some(id) => {
                let layer = if t.force_translucent {
                    Layer::Translucent
                } else {
                    match self.textures.info[id as usize].alpha {
                        Alpha::Opaque => Layer::Opaque,
                        Alpha::Cutout => Layer::Cutout,
                        Alpha::Translucent => Layer::Translucent,
                    }
                };
                (id, layer)
            }
            None => (MISSING, Layer::Opaque),
        };

        let shade = if el.shade {
            let (x2, y2, z2) = (n[0] * n[0], n[1] * n[1], n[2] * n[2]);
            (x2 * 0.6 + y2 * if n[1] > 0.0 { 1.0 } else { 0.5 } + z2 * 0.8).min(1.0)
        } else {
            1.0
        };

        Some(Quad {
            pos: pos.map(|p| [p[0] / 16.0, p[1] / 16.0, p[2] / 16.0]),
            uv: uv.map(|c| [c[0] / 16.0, c[1] / 16.0]),
            tex,
            layer,
            cull,
            light_dir: final_dir,
            tinted: face.tintindex.is_some(),
            shade,
        })
    }

    /// Bakes one block state. `props` are the state's properties, sorted by key.
    pub fn bake_state(&self, name: &str, props: &[(String, String)]) -> BlockInfo {
        let short = split_id(name).1;
        let prop = |k: &str| prop(props, k);
        let mut flags = 0;
        if matches!(
            short,
            "air" | "cave_air" | "void_air" | "light" | "barrier" | "structure_void"
        ) {
            flags |= F_AIR;
        }
        if short == "water" {
            flags |= F_WATER;
        }
        if short == "lava" {
            flags |= F_LAVA;
        }
        if prop("waterlogged") == Some("true")
            || matches!(
                short,
                "kelp" | "kelp_plant" | "seagrass" | "tall_seagrass" | "bubble_column"
            )
        {
            flags |= F_WATERLOGGED;
        }
        let fluid_level = prop("level").and_then(|v| v.parse().ok()).unwrap_or(0);

        let mut variants = Vec::new();
        let mut known = true;
        if flags & (F_AIR | F_WATER | F_LAVA) == 0 {
            match self.blockstate(name) {
                Some(bs) => self.bake_blockstate(&bs, props, &mut variants),
                None => known = false,
            }
        }
        if !known {
            flags |= F_UNKNOWN;
            variants.push(Variant {
                weight: 1,
                quads: self.fallback_cube(),
            });
        }
        let total_weight = variants.iter().map(|v| v.weight).sum();

        // Occlusion and full-cube shape from the first variant.
        let (mut occludes, mut full) = (0u8, 0u8);
        if let Some(v) = variants.first() {
            for q in &v.quads {
                if let Some(d) = q.cull
                    && covers_face(q, d)
                {
                    full |= 1 << d;
                    if q.layer == Layer::Opaque {
                        occludes |= 1 << d;
                    }
                }
            }
        }
        if full == 0x3f && occludes != 0x3f && !short.contains("leaves") {
            flags |= F_CULL_SAME;
        }

        let (top_color, top_tinted) = self.top_color(&variants, flags);
        BlockInfo {
            name: name.to_owned(),
            variants,
            total_weight,
            occludes,
            tint: tint_for(short, props),
            flags,
            fluid_level,
            top_color,
            top_tinted,
        }
    }

    fn bake_blockstate(&self, bs: &Value, props: &[(String, String)], out: &mut Vec<Variant>) {
        if let Some(vars) = bs.get("variants").and_then(|v| v.as_object()) {
            let chosen = vars.iter().find(|(key, _)| {
                key.split(',')
                    .filter(|s| !s.is_empty())
                    .all(|kv| match kv.split_once('=') {
                        Some((k, v)) => prop(props, k) == Some(v),
                        None => false,
                    })
            });
            if let Some((_, v)) = chosen {
                for (w, m) in model_list(v) {
                    let mut quads = Vec::new();
                    self.bake_ref(m, &mut quads);
                    out.push(Variant { weight: w, quads });
                }
            }
        } else if let Some(parts) = bs.get("multipart").and_then(|v| v.as_array()) {
            let mut quads = Vec::new();
            for part in parts {
                if part.get("when").is_none_or(|w| when_matches(w, props))
                    && let Some((_, m)) = part
                        .get("apply")
                        .map(model_list)
                        .and_then(|l| l.into_iter().next())
                {
                    self.bake_ref(m, &mut quads);
                }
            }
            out.push(Variant { weight: 1, quads });
        }
    }

    fn bake_ref(&self, m: &Value, out: &mut Vec<Quad>) {
        let Some(id) = m.get("model").and_then(|v| v.as_str()) else {
            return;
        };
        let rx = m.get("x").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let ry = m.get("y").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let uvlock = m.get("uvlock").and_then(|v| v.as_bool()).unwrap_or(false);
        self.bake_model(id, rx, ry, uvlock, out);
    }

    fn fallback_cube(&self) -> Vec<Quad> {
        let el = RawElement {
            from: [0.0; 3],
            to: [16.0; 3],
            rotation: None,
            shade: true,
            faces: HashMap::new(),
        };
        let face = |d: &str| RawFace {
            uv: None,
            texture: "#missing".into(),
            cullface: Some(d.into()),
            rotation: 0,
            tintindex: None,
        };
        ["down", "up", "north", "south", "west", "east"]
            .iter()
            .filter_map(|d| {
                self.bake_face(
                    &HashMap::new(),
                    &el,
                    parse_dir(d).unwrap(),
                    &face(d),
                    0,
                    0,
                    false,
                )
            })
            .collect()
    }

    fn top_color(&self, variants: &[Variant], flags: u8) -> ([u8; 3], bool) {
        let info = &self.textures.info;
        if flags & F_WATER != 0 {
            return ([255, 255, 255], true);
        }
        if flags & F_LAVA != 0 {
            let lava = self.textures.get("block/lava_still").unwrap_or(MISSING);
            return (info[lava as usize].avg, false);
        }
        let Some(v) = variants.first() else {
            return ([0; 3], false);
        };
        let ups: Vec<&Quad> = v.quads.iter().filter(|q| q.light_dir == UP).collect();
        let qs: Vec<&Quad> = if ups.is_empty() {
            v.quads.iter().collect()
        } else {
            ups
        };
        if qs.is_empty() {
            return ([0; 3], false);
        }
        let mut sum = [0u32; 3];
        for q in &qs {
            let c = info[q.tex as usize].avg;
            for i in 0..3 {
                sum[i] += c[i] as u32;
            }
        }
        let n = qs.len() as u32;
        (
            [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8],
            qs.iter().any(|q| q.tinted),
        )
    }

    pub fn fluid_texture(&self, lava: bool) -> u16 {
        let name = if lava {
            "block/lava_still"
        } else {
            "block/water_still"
        };
        self.textures.get(name).unwrap_or(MISSING)
    }
}

fn model_list(v: &Value) -> Vec<(u32, &Value)> {
    match v {
        Value::Array(items) => items
            .iter()
            .map(|m| {
                (
                    m.get("weight").and_then(|w| w.as_u64()).unwrap_or(1) as u32,
                    m,
                )
            })
            .collect(),
        _ => vec![(1, v)],
    }
}

fn prop<'a>(props: &'a [(String, String)], k: &str) -> Option<&'a str> {
    props
        .iter()
        .find(|(pk, _)| pk == k)
        .map(|(_, v)| v.as_str())
}

fn when_matches(w: &Value, props: &[(String, String)]) -> bool {
    if let Some(any) = w.get("OR").and_then(|v| v.as_array()) {
        return any.iter().any(|c| when_matches(c, props));
    }
    if let Some(all) = w.get("AND").and_then(|v| v.as_array()) {
        return all.iter().all(|c| when_matches(c, props));
    }
    let Some(obj) = w.as_object() else {
        return false;
    };
    obj.iter().all(|(k, v)| {
        let want = match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let (negate, want) = match want.strip_prefix('!') {
            Some(rest) => (true, rest.to_owned()),
            None => (false, want),
        };
        let have = prop(props, k).unwrap_or("");
        want.split('|').any(|alt| alt == have) != negate
    })
}

fn tint_for(short: &str, props: &[(String, String)]) -> Tint {
    let prop = |k: &str| prop(props, k);
    match short {
        "birch_leaves" => Tint::Fixed([0x80, 0xa7, 0x55]),
        "spruce_leaves" => Tint::Fixed([0x61, 0x99, 0x61]),
        "lily_pad" => Tint::Fixed([0x20, 0x80, 0x30]),
        "attached_melon_stem" | "attached_pumpkin_stem" => Tint::Fixed([0xe0, 0xc7, 0x1c]),
        "melon_stem" | "pumpkin_stem" => {
            let age: u32 = prop("age").and_then(|a| a.parse().ok()).unwrap_or(0);
            Tint::Fixed([(age * 32) as u8, (255 - age * 8) as u8, (age * 4) as u8])
        }
        "redstone_wire" => {
            let p: f32 = prop("power").and_then(|a| a.parse().ok()).unwrap_or(0.0) / 15.0;
            let r = p * 0.6 + if p > 0.0 { 0.4 } else { 0.3 };
            let g = (p * p * 0.7 - 0.5).clamp(0.0, 1.0);
            let b = (p * p * 0.6 - 0.7).clamp(0.0, 1.0);
            Tint::Fixed([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
        }
        "water" | "bubble_column" | "water_cauldron" => Tint::Water,
        "leaf_litter" => Tint::DryFoliage,
        "oak_leaves" | "jungle_leaves" | "acacia_leaves" | "dark_oak_leaves"
        | "mangrove_leaves" | "vine" => Tint::Foliage,
        s if s.contains("leaves") => Tint::Foliage,
        _ => Tint::Grass,
    }
}

/// True if a quad lies on the block boundary `d` and covers that whole face.
fn covers_face(q: &Quad, d: u8) -> bool {
    let (axis, at) = match d {
        DOWN => (1, 0.0),
        UP => (1, 1.0),
        NORTH => (2, 0.0),
        SOUTH => (2, 1.0),
        WEST => (0, 0.0),
        _ => (0, 1.0),
    };
    let eps = 1e-4;
    if q.pos.iter().any(|p| (p[axis] - at).abs() > eps) {
        return false;
    }
    let (a, b) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let min_a = q.pos.iter().map(|p| p[a]).fold(f32::MAX, f32::min);
    let max_a = q.pos.iter().map(|p| p[a]).fold(f32::MIN, f32::max);
    let min_b = q.pos.iter().map(|p| p[b]).fold(f32::MAX, f32::min);
    let max_b = q.pos.iter().map(|p| p[b]).fold(f32::MIN, f32::max);
    min_a <= eps && max_a >= 1.0 - eps && min_b <= eps && max_b >= 1.0 - eps
}

fn round_eps(v: f32) -> f32 {
    let r = v.round();
    if (v - r).abs() < 1e-3 { r } else { v }
}

fn rot_x(p: [f32; 3], deg: f32) -> [f32; 3] {
    if deg == 0.0 {
        return p;
    }
    let (s, c) = deg.to_radians().sin_cos();
    [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c]
}

fn rot_y(p: [f32; 3], deg: f32) -> [f32; 3] {
    if deg == 0.0 {
        return p;
    }
    let (s, c) = deg.to_radians().sin_cos();
    [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]
}

fn rot_z(p: [f32; 3], deg: f32) -> [f32; 3] {
    if deg == 0.0 {
        return p;
    }
    let (s, c) = deg.to_radians().sin_cos();
    [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]]
}

fn normal(pos: &[[f32; 3]; 4]) -> [f32; 3] {
    let a = [
        pos[1][0] - pos[0][0],
        pos[1][1] - pos[0][1],
        pos[1][2] - pos[0][2],
    ];
    let b = [
        pos[2][0] - pos[0][0],
        pos[2][1] - pos[0][1],
        pos[2][2] - pos[0][2],
    ];
    let n = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len < 1e-9 {
        // Degenerate (zero-width) faces: fall back to the second edge pair.
        let c = [
            pos[3][0] - pos[0][0],
            pos[3][1] - pos[0][1],
            pos[3][2] - pos[0][2],
        ];
        let n = [
            c[1] * a[2] - c[2] * a[1],
            c[2] * a[0] - c[0] * a[2],
            c[0] * a[1] - c[1] * a[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
        return [n[0] / len, n[1] / len, n[2] / len];
    }
    [n[0] / len, n[1] / len, n[2] / len]
}

/// Global registry of block states seen in the world, with dense u16 ids.
pub struct Registry {
    pub baker: Baker,
    ids: DashMap<String, u16>,
    blocks: boxcar::Vec<BlockInfo>,
}

pub const AIR: u16 = 0;

impl Registry {
    pub fn new(baker: Baker) -> Self {
        let r = Registry {
            baker,
            ids: DashMap::new(),
            blocks: boxcar::Vec::new(),
        };
        r.id("minecraft:air", &mut Vec::new());
        r
    }

    /// Returns the id for a state. `props` is sorted in place.
    pub fn id(&self, name: &str, props: &mut [(String, String)]) -> u16 {
        props.sort_unstable();
        let mut key = String::with_capacity(name.len() + 32);
        key.push_str(name);
        for (k, v) in props.iter() {
            key.push('|');
            key.push_str(k);
            key.push('=');
            key.push_str(v);
        }
        if let Some(id) = self.ids.get(&key) {
            return *id;
        }
        *self.ids.entry(key).or_insert_with(|| {
            let info = self.baker.bake_state(name, props);
            let id = self.blocks.push(info);
            if id > u16::MAX as usize {
                tracing::warn!("more than 65535 block states; extra states render as air");
                return AIR;
            }
            id as u16
        })
    }

    #[inline]
    pub fn get(&self, id: u16) -> &BlockInfo {
        &self.blocks[id as usize]
    }

    /// Block names that had no blockstate definition, with how many states each.
    pub fn unknown(&self) -> Vec<(String, usize)> {
        let mut m: HashMap<&str, usize> = HashMap::new();
        for (_, b) in self.blocks.iter() {
            if b.flags & F_UNKNOWN != 0 {
                *m.entry(b.name.as_str()).or_default() += 1;
            }
        }
        let mut v: Vec<(String, usize)> = m.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
        v.sort();
        v
    }

    pub fn len(&self) -> usize {
        self.blocks.count()
    }
}
