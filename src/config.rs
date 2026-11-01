use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashSet;
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
    /// Where the map starts and its Home button goes, as [x, z]. Defaults to the
    /// middle of what's rendered.
    pub start: Option<[f64; 2]>,
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
        cfg.validate()
            .with_context(|| format!("checking {}", path.display()))?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.maps.is_empty() {
            bail!("it has no [[maps]]");
        }
        let mut ids = HashSet::new();
        for m in &self.maps {
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
            if !ids.insert(&m.id) {
                bail!("map id {:?} is used twice", m.id);
            }
            if m.min_y >= m.max_y {
                bail!(
                    "map {}: min_y ({}) must be below max_y ({})",
                    m.id,
                    m.min_y,
                    m.max_y
                );
            }
            if m.max_y - m.min_y > 1000 {
                bail!(
                    "map {}: world height over 1000 blocks isn't supported",
                    m.id
                );
            }
        }
        Ok(())
    }

    /// Whether tiles can be written, to say so before any time goes on rendering.
    /// Not always fatal: a read-only root can hold writable map directories.
    pub fn check_data_dir(&self) -> Result<()> {
        let hint = "In the container image tealmap runs as uid 1000, so the directory must belong to that user";
        let dir = &self.data_dir;
        std::fs::create_dir_all(dir)
            .with_context(|| format!("can't create {}. {hint}", dir.display()))?;
        let probe = dir.join(format!(".tealmap-write-check-{}", std::process::id()));
        // Only ever a new file: a name that's somehow taken isn't this check's to touch.
        match std::fs::File::create_new(&probe) {
            Ok(_) => {
                let _ = std::fs::remove_file(&probe);
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(e) => Err(e).with_context(|| format!("can't write to {}. {hint}", dir.display())),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn error(body: &str) -> String {
        let cfg: Config = toml::from_str(&format!("data_dir = \"/tmp/x\"\n{body}")).unwrap();
        format!("{:#}", cfg.validate().unwrap_err())
    }

    #[test]
    fn rejects_configs_that_cant_work() {
        assert!(error("maps = []").contains("no [[maps]]"));
        let map = |id: &str, extra: &str| {
            format!("[[maps]]\nid = \"{id}\"\nworld = \"/w\"\nversion = \"26.2\"\n{extra}\n")
        };
        let twice = format!("{}{}", map("a", ""), map("a", ""));
        assert!(error(&twice).contains("used twice"));
        assert!(error(&map("a", "min_y = 10\nmax_y = 10")).contains("must be below"));
        assert!(error(&map("a", "min_y = 0\nmax_y = 2000")).contains("over 1000"));
        assert!(error(&map("a/b", "")).contains("only letters"));

        let good = format!("data_dir = \"/tmp/x\"\n{}{}", map("a", ""), map("b", ""));
        let cfg: Config = toml::from_str(&good).unwrap();
        cfg.validate().unwrap();
    }
}
