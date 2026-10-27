//! Biome colors from worldgen biome JSON (vanilla and datapack/mod biomes).

use crate::assets::{AssetFiles, split_id};
use crate::textures::decode_colormap;
use dashmap::DashMap;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiomeColors {
    pub grass: [u8; 3],
    pub foliage: [u8; 3],
    pub dry_foliage: [u8; 3],
    pub water: [u8; 3],
}

pub struct Biomes {
    grass_map: Option<Vec<[u8; 3]>>,
    foliage_map: Option<Vec<[u8; 3]>>,
    dry_foliage_map: Option<Vec<[u8; 3]>>,
    ids: DashMap<String, u16>,
    /// Name and colors per id, pushed together so concurrent inserts stay aligned.
    entries: boxcar::Vec<(String, BiomeColors)>,
    json: std::collections::HashMap<String, Value>,
}

const DEFAULT_WATER: [u8; 3] = [0x3f, 0x76, 0xe4];

fn rgb(v: u32) -> [u8; 3] {
    [(v >> 16) as u8, (v >> 8) as u8, v as u8]
}

/// Biome colors are ints in 1.21 and "#rrggbb" strings in 26.x.
fn color(v: Option<&Value>) -> Option<[u8; 3]> {
    match v? {
        Value::Number(n) => Some(rgb(n.as_u64()? as u32)),
        Value::String(s) => Some(rgb(u32::from_str_radix(s.trim_start_matches('#'), 16).ok()?)),
        _ => None,
    }
}

fn sample(map: &Option<Vec<[u8; 3]>>, temp: f64, downfall: f64, fallback: [u8; 3]) -> [u8; 3] {
    let Some(map) = map else { return fallback };
    let t = temp.clamp(0.0, 1.0);
    let d = downfall.clamp(0.0, 1.0) * t;
    let x = ((1.0 - t) * 255.0) as usize;
    let y = ((1.0 - d) * 255.0) as usize;
    map[y * 256 + x]
}

impl Biomes {
    pub fn load(assets: &AssetFiles) -> Self {
        let mut json = std::collections::HashMap::new();
        for path in assets.paths() {
            // data/<ns>/worldgen/biome/<path>.json
            let Some(rest) = path.strip_prefix("data/") else {
                continue;
            };
            let Some((ns, p)) = rest.split_once("/worldgen/biome/") else {
                continue;
            };
            if ns.contains('/') {
                continue;
            }
            let id = format!("{ns}:{}", p.trim_end_matches(".json"));
            if let Ok(v) = serde_json::from_slice::<Value>(assets.get(path).unwrap()) {
                json.insert(id, v);
            }
        }
        let map = |n: &str| {
            assets
                .get(&format!("assets/minecraft/textures/colormap/{n}.png"))
                .and_then(decode_colormap)
        };
        let biomes = Biomes {
            grass_map: map("grass"),
            foliage_map: map("foliage"),
            dry_foliage_map: map("dry_foliage"),
            ids: DashMap::new(),
            entries: boxcar::Vec::new(),
            json,
        };
        // Id 0 is the fallback for sections and columns with no biome, so it
        // has to be the same biome in every run.
        biomes.id("minecraft:plains");
        biomes
    }

    /// Returns a dense id for a biome name, computing its colors on first use.
    pub fn id(&self, name: &str) -> u16 {
        if let Some(id) = self.ids.get(name) {
            return *id;
        }
        *self.ids.entry(name.to_owned()).or_insert_with(|| {
            let c = self.compute(name);
            self.entries.push((name.to_owned(), c)) as u16
        })
    }

    pub fn name(&self, id: u16) -> &str {
        &self.entries[id as usize].0
    }

    pub fn colors(&self, id: u16) -> BiomeColors {
        self.entries[id as usize].1
    }

    fn compute(&self, name: &str) -> BiomeColors {
        let (ns, p) = split_id(name);
        let v = self.json.get(&format!("{ns}:{p}"));
        if v.is_none() {
            tracing::debug!("unknown biome {name}, using plains colors");
        }
        let temp = v.and_then(|v| v["temperature"].as_f64()).unwrap_or(0.8);
        let downfall = v.and_then(|v| v["downfall"].as_f64()).unwrap_or(0.4);
        let fx = v.map(|v| &v["effects"]);
        let attr = v.map(|v| &v["attributes"]);
        let grass = color(fx.and_then(|f| f.get("grass_color")))
            .unwrap_or_else(|| sample(&self.grass_map, temp, downfall, [0x91, 0xbd, 0x59]));
        let grass = match fx.and_then(|f| f["grass_color_modifier"].as_str()) {
            Some("swamp") => [0x6a, 0x70, 0x39],
            Some("dark_forest") => {
                let c = u32::from_be_bytes([0, grass[0], grass[1], grass[2]]);
                rgb(((c & 0xfefefe) + 0x28340a) >> 1)
            }
            _ => grass,
        };
        BiomeColors {
            grass,
            foliage: color(fx.and_then(|f| f.get("foliage_color")))
                .unwrap_or_else(|| sample(&self.foliage_map, temp, downfall, [0x77, 0xab, 0x2f])),
            dry_foliage: color(fx.and_then(|f| f.get("dry_foliage_color"))).unwrap_or_else(|| {
                sample(&self.dry_foliage_map, temp, downfall, [0xa0, 0x7a, 0x4b])
            }),
            water: color(fx.and_then(|f| f.get("water_color")))
                .or_else(|| color(attr.and_then(|a| a.get("minecraft:visual/water_color"))))
                .unwrap_or(DEFAULT_WATER),
        }
    }
}
