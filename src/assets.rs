//! Loads vanilla assets from a Minecraft client jar (and biome data from
//! optional extra jars), and downloads the client jar from Mojang when needed.

use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The subset of jar entries tealmap needs, held in memory.
pub struct AssetFiles {
    files: HashMap<String, Vec<u8>>,
}

fn wanted(name: &str) -> bool {
    if name.ends_with('/') {
        return false;
    }
    if let Some(rest) = name.strip_prefix("assets/") {
        let Some((_, path)) = rest.split_once('/') else {
            return false;
        };
        return path.starts_with("blockstates/")
            || path.starts_with("models/block/")
            || path.starts_with("textures/block/")
            || path.starts_with("textures/colormap/");
    }
    if let Some(rest) = name.strip_prefix("data/") {
        return rest.contains("/worldgen/biome/") && name.ends_with(".json");
    }
    false
}

impl AssetFiles {
    pub fn new() -> Self {
        AssetFiles {
            files: HashMap::new(),
        }
    }

    /// Adds the wanted entries of a jar. `biomes_only` limits it to biome data,
    /// which is how mod jars are used: their blocks are out of scope for now.
    pub fn add_jar(&mut self, path: &Path, biomes_only: bool) -> Result<()> {
        let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut zip = zip::ZipArchive::new(std::io::BufReader::new(f))
            .with_context(|| format!("reading {}", path.display()))?;
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i)?;
            let Ok(name) = entry.name().map(|n| n.into_owned()) else {
                continue;
            };
            if !wanted(&name) || (biomes_only && !name.starts_with("data/")) {
                continue;
            }
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf)?;
            self.files.entry(name).or_insert(buf);
        }
        Ok(())
    }

    pub fn get(&self, path: &str) -> Option<&[u8]> {
        self.files.get(path).map(|v| v.as_slice())
    }

    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(|s| s.as_str())
    }
}

/// Splits `ns:path` (namespace defaults to minecraft).
pub fn split_id(id: &str) -> (&str, &str) {
    match id.split_once(':') {
        Some((ns, p)) => (ns, p),
        None => ("minecraft", id),
    }
}

/// Returns the path of the client jar for `version`, downloading it into
/// `cache_dir` if it isn't there yet. The jar is verified by SHA-1.
pub fn client_jar(version: &str, cache_dir: &Path) -> Result<PathBuf> {
    let path = cache_dir.join(format!("client-{version}.jar"));
    let sha_path = cache_dir.join(format!("client-{version}.jar.sha1"));
    if path.exists() && sha_path.exists() {
        return Ok(path);
    }
    std::fs::create_dir_all(cache_dir)?;
    tracing::info!("downloading Minecraft {version} client jar from Mojang");
    let manifest: serde_json::Value =
        get_json("https://piston-meta.mojang.com/mc/game/version_manifest_v2.json")?;
    let url = manifest["versions"]
        .as_array()
        .and_then(|vs| vs.iter().find(|v| v["id"] == version))
        .and_then(|v| v["url"].as_str())
        .ok_or_else(|| anyhow!("Minecraft version {version} not found in Mojang's manifest"))?
        .to_owned();
    let meta: serde_json::Value = get_json(&url)?;
    let client = &meta["downloads"]["client"];
    let jar_url = client["url"]
        .as_str()
        .ok_or_else(|| anyhow!("no client download for {version}"))?;
    let sha1 = client["sha1"].as_str().unwrap_or_default().to_owned();

    let mut body = Vec::new();
    ureq::get(jar_url)
        .call()?
        .into_body()
        .with_config()
        .limit(512 * 1024 * 1024)
        .reader()
        .read_to_end(&mut body)?;
    let got = sha1_smol::Sha1::from(&body).digest().to_string();
    if got != sha1 {
        bail!("client jar checksum mismatch for {version}: expected {sha1}, got {got}");
    }
    let tmp = path.with_extension("jar.tmp");
    std::fs::write(&tmp, &body)?;
    std::fs::rename(&tmp, &path)?;
    std::fs::write(&sha_path, &sha1)?;
    Ok(path)
}

fn get_json(url: &str) -> Result<serde_json::Value> {
    let text = ureq::get(url).call()?.into_body().read_to_string()?;
    Ok(serde_json::from_str(&text)?)
}
