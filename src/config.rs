use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Output tiles, state and the asset cache live here.
    pub data_dir: PathBuf,
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Render threads. Defaults to half the CPUs.
    pub threads: Option<usize>,
    /// How often `serve` checks region files for changes.
    #[serde(default = "default_poll")]
    pub poll_seconds: u64,
    pub maps: Vec<MapConfig>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct MapConfig {
    pub id: String,
    pub name: Option<String>,
    /// Dimension directory containing `region/`, e.g.
    /// `world/dimensions/minecraft/overworld` (26.x) or `world` (1.21).
    pub world: PathBuf,
    /// Minecraft version, used to fetch the matching client jar.
    pub version: String,
    /// Use this client jar instead of downloading one.
    pub client_jar: Option<PathBuf>,
    /// A directory of mod jars to read biome definitions from.
    pub mods: Option<PathBuf>,
    #[serde(default = "default_min_y")]
    pub min_y: i32,
    #[serde(default = "default_max_y")]
    pub max_y: i32,
    /// Skip unlit blocks below this y and the ocean floor (BlueMap's default is 55).
    #[serde(default = "default_caves")]
    pub remove_caves_below_y: i32,
}

fn default_listen() -> String {
    "0.0.0.0:8200".into()
}
fn default_poll() -> u64 {
    10
}
fn default_min_y() -> i32 {
    -64
}
fn default_max_y() -> i32 {
    320
}
fn default_caves() -> i32 {
    55
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let cfg: Config =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        for m in &cfg.maps {
            if m.id.is_empty()
                || !m
                    .id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                bail!(
                    "map id {:?} must be non-empty and use only letters, digits, - and _",
                    m.id
                );
            }
            if m.max_y - m.min_y > 1000 {
                bail!(
                    "map {}: world height over 1000 blocks isn't supported",
                    m.id
                );
            }
        }
        Ok(cfg)
    }

    pub fn threads(&self) -> usize {
        self.threads
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get() / 2)
                    .unwrap_or(1)
            })
            .max(1)
    }
}

impl MapConfig {
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }
}
