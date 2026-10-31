//! Map rendering: change detection, region jobs, the LOD pyramid and the map manifest.

use crate::assets::{self, AssetFiles};
use crate::config::MapConfig;
use crate::lod;
use crate::mesh::{Mesher, Settings, TILE, Volume};
use crate::model::{Baker, Registry};
use crate::region::{Region, list_regions};
use crate::world::{Chunk, DecodeCache, decode_chunk};
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, UNIX_EPOCH};

/// Bump when output for the same input changes; forces a full re-render.
pub const RENDER_VERSION: u32 = 6;
const STATE_MAGIC: &[u8; 4] = b"TMS2";
/// State from before it remembered the region file: just hashes.
const OLD_STATE_MAGIC: &[u8; 4] = b"TMS1";
/// Hires tiles per render thread in each batch of a pass, about 20 seconds
/// of work.
const BATCH_TILES_PER_THREAD: usize = 256;
const NO_STAMP: (u64, u64) = (u64::MAX, u64::MAX);

pub struct MapCtx {
    pub cfg: MapConfig,
    pub reg: Registry,
    pub out: PathBuf,
    pub settings: Settings,
    /// Region file (mtime, size) seen at the last successful scan.
    seen: Mutex<HashMap<(i32, i32), (u64, u64)>>,
    /// What's on disk, once read, so manifests don't list directories.
    index: Mutex<Option<MapIndex>>,
    batch_tiles: usize,
}

/// Which tiles exist.
#[derive(Default)]
struct MapIndex {
    /// Hires tile mask per region, with bit `(tz % 8) * 8 + tx % 8`.
    hires: BTreeMap<(i32, i32), u64>,
    /// LOD tiles with a mesh, per level (index 0 is level 1).
    lod: Vec<BTreeSet<(i32, i32)>>,
    /// The manifest last written, to skip writing an identical one.
    written: Option<Vec<u8>>,
    render_version: Option<Option<u32>>,
}

impl MapIndex {
    fn set_hires(&mut self, tx: i32, tz: i32, exists: bool) {
        let region = (tx.div_euclid(8), tz.div_euclid(8));
        let bit = 1u64 << (tz.rem_euclid(8) * 8 + tx.rem_euclid(8));
        let mask = self.hires.entry(region).or_default();
        if exists {
            *mask |= bit;
        } else {
            *mask &= !bit;
            if *mask == 0 {
                self.hires.remove(&region);
            }
        }
    }

    /// LOD nodes that sit over a hires tile but have no mesh, sorted.
    fn lod_missing(&self) -> Vec<[i32; 3]> {
        let mut out = Vec::new();
        let mut nodes: BTreeSet<(i32, i32)> = BTreeSet::new();
        for (&(rx, rz), &mask) in &self.hires {
            for bit in (0..64).filter(|b| mask >> b & 1 == 1) {
                nodes.insert((rx * 8 + (bit & 7), rz * 8 + (bit >> 3)));
            }
        }
        for level in 1..=lod::LEVELS {
            nodes = nodes.iter().map(|&(x, z)| (x >> 1, z >> 1)).collect();
            let meshes = &self.lod[level as usize - 1];
            out.extend(
                nodes
                    .iter()
                    .filter(|n| !meshes.contains(n))
                    .map(|&(x, z)| [level as i32, x, z]),
            );
        }
        out
    }
}

#[derive(Default, Debug, Serialize, Clone)]
pub struct Update {
    pub hires: Vec<[i32; 2]>,
    /// Changed LOD tiles as [level, x, z].
    pub lod: Vec<[i32; 3]>,
    #[serde(skip)]
    pub stats: Stats,
}

impl Update {
    fn add(&mut self, results: &[RegionResult], lod: &[[i32; 3]]) {
        for r in results {
            self.hires.extend(r.tiles.iter().map(|&(x, z)| [x, z]));
            self.stats.tiles += r.tiles.len();
            self.stats.empty_tiles += r.empty;
            self.stats.failed_chunks += r.failed.len();
            self.stats.bytes += r.bytes;
            self.stats.vertices += r.vertices;
        }
        self.lod.extend_from_slice(lod);
    }
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub regions: usize,
    pub chunks: usize,
    pub tiles: usize,
    pub empty_tiles: usize,
    pub failed_chunks: usize,
    pub bytes: u64,
    pub vertices: u64,
    pub seconds: f64,
}

impl MapCtx {
    pub fn new(cfg: MapConfig, data_dir: &Path) -> Result<Self> {
        let jar = match &cfg.client_jar {
            Some(p) => p.clone(),
            None => assets::client_jar(&cfg.version, &data_dir.join("cache"))?,
        };
        let mut files = AssetFiles::new();
        files.add_jar(&jar, false)?;
        if let Some(mods) = &cfg.mods {
            let mut jars: Vec<PathBuf> = std::fs::read_dir(mods)
                .with_context(|| format!("listing {}", mods.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "jar"))
                .collect();
            jars.sort();
            for j in jars {
                if let Err(e) = files.add_jar(&j, true) {
                    tracing::warn!("skipping {}: {e:#}", j.display());
                }
            }
        }
        Self::with_baker(cfg, data_dir, Baker::new(files))
    }

    pub fn with_baker(cfg: MapConfig, data_dir: &Path, baker: Baker) -> Result<Self> {
        let out = data_dir.join("maps").join(&cfg.id);
        std::fs::create_dir_all(&out)?;
        let settings = Settings {
            min_y: cfg.min_y,
            remove_caves_below_y: cfg.remove_caves_below_y,
            lod: false,
        };
        Ok(MapCtx {
            cfg,
            reg: Registry::new(baker),
            out,
            settings,
            seen: Mutex::new(HashMap::new()),
            index: Mutex::new(None),
            batch_tiles: BATCH_TILES_PER_THREAD,
        })
    }

    fn region_dir(&self) -> PathBuf {
        self.cfg.world.join("region")
    }

    /// Renders everything that changed since the last pass (or everything, if
    /// `full`), and returns the whole pass.
    ///
    /// A big pass renders in batches, nearest the middle of the map first.
    /// Each batch writes the manifest and goes to `publish`, so viewers fill
    /// in as it renders, and saves the state of the regions it finished, so a
    /// restart picks up where the pass stopped.
    pub fn render(
        &self,
        pool: &rayon::ThreadPool,
        full: bool,
        mut publish: impl FnMut(&Update),
    ) -> Result<Update> {
        let t0 = Instant::now();
        // Batches keep this version in the manifest until the pass ends, so a
        // forced full pass that stops early runs as a full pass again.
        let prev = self.stored_version()?;
        let full = full || prev != Some(RENDER_VERSION);
        if full {
            self.seen.lock().unwrap().clear();
            // Output from older versions that this one no longer writes.
            let _ = std::fs::remove_dir_all(self.out.join("lowres"));
        }
        if prev != Some(RENDER_VERSION) {
            // Another version's LOD sources can't feed this one's tiles.
            for level in 1..=lod::LEVELS {
                let _ = std::fs::remove_dir_all(self.out.join("lod").join(format!("src{level}")));
            }
        }
        let fresh = !self.out.join("map.json").exists();
        self.write_textures(full || fresh)?;
        if fresh {
            // Publish the map before the first batch finishes so viewers can open it.
            self.write_manifest(prev)?;
        }
        let regions = list_regions(&self.region_dir())?;
        let present: BTreeSet<(i32, i32)> = regions.iter().map(|&(x, z, _)| (x, z)).collect();

        // 1. Find changed chunks by hashing compressed payloads of changed region files.
        let scans: Vec<RegionScan> = pool.install(|| {
            regions
                .par_iter()
                .filter_map(|(rx, rz, path)| self.scan_region(*rx, *rz, path, full).transpose())
                .collect::<Result<Vec<_>>>()
        })?;
        // Regions whose files disappeared: every chunk they had is now gone.
        let mut scans = scans;
        for (rx, rz) in self.stored_regions()? {
            if !present.contains(&(rx, rz)) {
                let (old, _) = self.load_state(rx, rz);
                let dirty = (0..1024).filter(|&i| old[i] != 0).collect();
                scans.push(RegionScan {
                    rx,
                    rz,
                    hashes: vec![0; 1024],
                    dirty,
                    stamp: None,
                });
            }
        }

        // 2. Changed chunks → tiles (including neighbors whose halo touches them).
        let mut tiles: BTreeMap<(i32, i32), BTreeSet<(i32, i32)>> = BTreeMap::new();
        let mut chunks = 0;
        for s in &scans {
            for &i in &s.dirty {
                chunks += 1;
                let cx = s.rx * 32 + (i % 32) as i32;
                let cz = s.rz * 32 + (i / 32) as i32;
                for (tx, tz) in tiles_touching(cx, cz) {
                    tiles
                        .entry((tx.div_euclid(8), tz.div_euclid(8)))
                        .or_default()
                        .insert((tx, tz));
                }
            }
        }

        // 3. Render region jobs in batches, nearest the middle first, so the map
        // grows outward from where viewers start.
        let jobs = nearest_first(tiles);
        let batches = batches(&jobs, self.batch_tiles * pool.current_num_threads());
        let total: usize = jobs.iter().map(|j| j.1.len()).sum();
        let queued: BTreeSet<(i32, i32)> = jobs.iter().map(|j| j.0).collect();
        let mut finished = BTreeSet::new();
        let mut failed = BTreeSet::new();
        let mut saved = vec![false; scans.len()];
        let mut pass = Update::default();
        let mut rendered = 0;
        for batch in &batches {
            let results: Vec<RegionResult> = pool.install(|| {
                batch
                    .par_iter()
                    .map(|((rx, rz), set)| self.render_region(*rx, *rz, set))
                    .collect::<Result<Vec<_>>>()
            })?;
            let lod = self.build_lods(pool, &results)?;
            let mut up = Update::default();
            up.add(&results, &lod);
            pass.add(&results, &lod);
            self.update_index(&results, &lod);
            failed.extend(results.iter().flat_map(|r| r.failed.iter().copied()));
            finished.extend(batch.iter().map(|j| j.0));
            rendered += batch.iter().map(|j| j.1.len()).sum::<usize>();
            // List the tiles before saving state: a region with saved state
            // won't render again, so nothing else would list them.
            self.write_manifest(prev)?;

            // 4. Persist state for regions that every job reading their chunks
            // (their own and the eight around it) has finished.
            let done = |j| !queued.contains(&j) || finished.contains(&j);
            for (s, saved) in scans.iter().zip(&mut saved) {
                if !*saved && (-1..=1).all(|dx| (-1..=1).all(|dz| done((s.rx + dx, s.rz + dz)))) {
                    self.save_scan(s, &failed)?;
                    *saved = true;
                }
            }
            if rendered < total {
                tracing::info!("{}: {rendered}/{total} tiles", self.cfg.id);
            }
            publish(&up);
        }
        // With no jobs, no batch saved state.
        for (s, _) in scans.iter().zip(&saved).filter(|(_, saved)| !**saved) {
            self.save_scan(s, &failed)?;
        }
        if prev != Some(RENDER_VERSION) {
            self.write_manifest(Some(RENDER_VERSION))?;
        }

        pass.stats.regions = regions.len();
        pass.stats.chunks = chunks;
        pass.stats.seconds = t0.elapsed().as_secs_f64();
        Ok(pass)
    }

    /// Saves a region's chunk hashes. Chunks that failed to read get a
    /// sentinel hash so the next pass retries them, and the region isn't
    /// marked as seen.
    fn save_scan(&self, s: &RegionScan, failed: &BTreeSet<(i32, i32)>) -> Result<()> {
        let mut hashes = s.hashes.clone();
        let mut ok = true;
        for (i, h) in hashes.iter_mut().enumerate() {
            if failed.contains(&(s.rx * 32 + (i % 32) as i32, s.rz * 32 + (i / 32) as i32)) {
                *h = u64::MAX;
                ok = false;
            }
        }
        // A failed chunk leaves no stamp, so the region is scanned again.
        let stamp = s.stamp.filter(|_| ok);
        self.save_state(s.rx, s.rz, &hashes, stamp)?;
        if let Some(st) = stamp {
            self.seen.lock().unwrap().insert((s.rx, s.rz), st);
        }
        Ok(())
    }

    /// Rebuilds the LOD pyramid over these jobs: every tile whose sources they
    /// changed (its own, or a neighbor's along its border), one level at a
    /// time, since each level reads the sources of the one below. Returns the
    /// LOD tiles whose mesh changed.
    fn build_lods(
        &self,
        pool: &rayon::ThreadPool,
        results: &[RegionResult],
    ) -> Result<Vec<[i32; 3]>> {
        let mut dirty: BTreeSet<(i32, i32)> = results
            .iter()
            .flat_map(|r| {
                r.sources
                    .iter()
                    .flat_map(|&(x, z)| lod::tiles_using_source(x, z))
            })
            .collect();
        let mut lod_changed = Vec::new();
        for level in 1..=lod::LEVELS {
            let built: Vec<LodBuild> = pool.install(|| {
                dirty
                    .par_iter()
                    .map_init(
                        || (Volume::new(), Mesher::new()),
                        |(volume, mesher), &(x, z)| self.build_lod(level, x, z, volume, mesher),
                    )
                    .collect::<Result<Vec<_>>>()
            })?;
            lod_changed.extend(
                built
                    .iter()
                    .filter(|b| b.mesh_changed)
                    .map(|b| [level as i32, b.x, b.z]),
            );
            // A tile whose source came out the same changes nothing above it.
            dirty = built
                .iter()
                .filter(|b| b.source_changed)
                .flat_map(|b| lod::tiles_using_source(b.x, b.z))
                .collect();
        }
        Ok(lod_changed)
    }

    fn scan_region(&self, rx: i32, rz: i32, path: &Path, full: bool) -> Result<Option<RegionScan>> {
        let meta = std::fs::metadata(path)?;
        let mtime = meta
            .modified()?
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let stamp = (mtime, meta.len());
        if !full && self.seen.lock().unwrap().get(&(rx, rz)) == Some(&stamp) {
            return Ok(None);
        }
        // Full renders still compare against saved state, so chunks that have
        // since disappeared get their stale tiles cleared too.
        let (old, saved_stamp) = self.load_state(rx, rz);
        // Saved state remembers the file, so a restart doesn't reread the world.
        if !full && saved_stamp == Some(stamp) {
            self.seen.lock().unwrap().insert((rx, rz), stamp);
            return Ok(None);
        }
        let region = Region::open(path, rx, rz)?;
        let mut raw = Vec::new();
        let mut hashes = vec![0u64; 1024];
        let mut dirty = Vec::new();
        for i in 0..1024 {
            let h = match region.raw_chunk(i % 32, i / 32, &mut raw) {
                Ok(Some(kind)) => xxhash_rust::xxh3::xxh3_64(&raw) ^ kind as u64 | 1,
                Ok(None) => 0,
                // Torn or truncated read: treat as changed and let the render retry.
                Err(_) => u64::MAX,
            };
            hashes[i] = h;
            if if full {
                h != 0 || old[i] != 0
            } else {
                h != old[i]
            } {
                dirty.push(i);
            }
        }
        Ok(Some(RegionScan {
            rx,
            rz,
            hashes,
            dirty,
            stamp: Some(stamp),
        }))
    }

    fn render_region(
        &self,
        rx: i32,
        rz: i32,
        tiles: &BTreeSet<(i32, i32)>,
    ) -> Result<RegionResult> {
        let mut res = RegionResult::default();
        let mut src = ChunkSource::new(self);
        let mut volume = Volume::new();
        let mut mesher = Mesher::new();
        let mut gz = Vec::new();
        let dir = self.out.join("hires").join(format!("{rx}_{rz}"));
        std::fs::create_dir_all(&dir)?;

        let mut current_row = i32::MIN;
        for &(tz, tx) in &tiles.iter().map(|&(x, z)| (z, x)).collect::<BTreeSet<_>>() {
            if tz != current_row {
                src.evict_before(tz * 4 - 1);
                current_row = tz;
            }
            let (x0, z0) = (tx * TILE, tz * TILE);
            let mut tile_failed = false;
            for cz in (z0 >> 4) - 1..(z0 >> 4) + 5 {
                for cx in (x0 >> 4) - 1..(x0 >> 4) + 5 {
                    if let Err(e) = src.load(cx, cz) {
                        tracing::warn!("{}: {e:#}", self.cfg.id);
                        res.failed.insert((cx, cz));
                        tile_failed |= (cx - (x0 >> 4)) as u32 <= 3 && (cz - (z0 >> 4)) as u32 <= 3;
                    }
                }
            }
            if tile_failed {
                // Keep the last good tile; the chunk is retried next pass.
                continue;
            }
            let path = dir.join(format!("{tx}_{tz}.tmh"));
            let filled = volume.fill(x0, z0, &self.settings, &self.reg, &|cx, cz| src.get(cx, cz));
            let encoded = if filled {
                mesher.mesh(&volume, &self.reg, &self.settings)
            } else {
                None
            };
            // The level 1 LOD source this tile leaves for its parent.
            let source = filled
                .then(|| lod::hires_source(&volume, &self.reg, self.cfg.min_y, self.cfg.max_y))
                .flatten();
            let source_path = lod::source_path(&self.out, 1, tx, tz);
            if lod::write_source(&source_path, source.as_ref(), &self.reg)? {
                res.sources.push((tx, tz));
            }
            match encoded {
                Some(data) => {
                    res.vertices += vertex_count(&data);
                    gzip_into(&data, &mut gz);
                    res.bytes += gz.len() as u64;
                    // An unchanged tile keeps its file, so caches stay good.
                    if write_if_changed(&path, &gz)? {
                        res.tiles.push((tx, tz));
                    }
                }
                None => {
                    res.empty += 1;
                    if path.exists() {
                        std::fs::remove_file(&path)?;
                        res.tiles.push((tx, tz));
                    }
                }
            }
        }
        Ok(res)
    }

    /// Builds LOD tile (x, z) at `level`, writing its mesh and its source for
    /// the level above. Returns whether the mesh changed.
    fn build_lod(
        &self,
        level: u32,
        x: i32,
        z: i32,
        volume: &mut Volume,
        mesher: &mut Mesher,
    ) -> Result<LodBuild> {
        let (tile, parent) = lod::build_tile(
            &self.out,
            &self.reg,
            level,
            x,
            z,
            self.cfg.min_y,
            self.cfg.max_y,
            volume,
            mesher,
        )?;
        let path = lod::tile_path(&self.out, level, x, z);
        let mesh_changed = match tile {
            Some(data) => {
                let mut gz = Vec::new();
                gzip_into(&data, &mut gz);
                write_if_changed(&path, &gz)?
            }
            None if path.exists() => {
                std::fs::remove_file(&path)?;
                true
            }
            None => false,
        };
        let source_changed = level < lod::LEVELS
            && lod::write_source(
                &lod::source_path(&self.out, level + 1, x, z),
                parent.as_ref(),
                &self.reg,
            )?;
        Ok(LodBuild {
            x,
            z,
            mesh_changed,
            source_changed,
        })
    }

    fn state_path(&self, rx: i32, rz: i32) -> PathBuf {
        self.out.join("state").join(format!("{rx}_{rz}.bin"))
    }

    /// A region's saved chunk hashes, and the (mtime, size) of the file they
    /// describe if it was read cleanly. State is magic, stamp, then hashes; a
    /// stamp of all ones means there isn't one.
    fn load_state(&self, rx: i32, rz: i32) -> (Vec<u64>, Option<(u64, u64)>) {
        let mut out = vec![0u64; 1024];
        let mut stamp = None;
        if let Ok(b) = std::fs::read(self.state_path(rx, rz)) {
            let hashes = if b.len() == 4 + 16 + 1024 * 8 && &b[..4] == STATE_MAGIC {
                let at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
                stamp = Some((at(4), at(12))).filter(|&s| s != NO_STAMP);
                &b[20..]
            } else if b.len() == 4 + 1024 * 8 && &b[..4] == OLD_STATE_MAGIC {
                &b[4..]
            } else {
                &[]
            };
            for (i, c) in hashes.chunks_exact(8).enumerate() {
                out[i] = u64::from_le_bytes(c.try_into().unwrap());
            }
        }
        (out, stamp)
    }

    fn save_state(
        &self,
        rx: i32,
        rz: i32,
        hashes: &[u64],
        stamp: Option<(u64, u64)>,
    ) -> Result<()> {
        let path = self.state_path(rx, rz);
        if hashes.iter().all(|&h| h == 0) {
            let _ = std::fs::remove_file(&path);
            return Ok(());
        }
        std::fs::create_dir_all(path.parent().unwrap())?;
        let mut b = Vec::with_capacity(4 + 16 + 1024 * 8);
        b.extend_from_slice(STATE_MAGIC);
        let (mtime, size) = stamp.unwrap_or(NO_STAMP);
        b.extend_from_slice(&mtime.to_le_bytes());
        b.extend_from_slice(&size.to_le_bytes());
        for h in hashes {
            b.extend_from_slice(&h.to_le_bytes());
        }
        write_atomic(&path, &b)
    }

    fn stored_regions(&self) -> Result<Vec<(i32, i32)>> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(self.out.join("state")) {
            for e in rd.flatten() {
                let name = e.file_name();
                let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".bin")) else {
                    continue;
                };
                if let Some((x, z)) = stem.split_once('_')
                    && let (Ok(x), Ok(z)) = (x.parse(), z.parse())
                {
                    out.push((x, z));
                }
            }
        }
        Ok(out)
    }

    /// The render version of the map's tiles and state. A map with neither
    /// is new, so it counts as current.
    fn stored_version(&self) -> Result<Option<u32>> {
        let mut guard = self.index.lock().unwrap();
        let index = self.index_mut(&mut guard);
        if let Some(v) = index.render_version {
            return Ok(v);
        }
        let listed = read_manifest(&self.out.join("map.json"))
            .and_then(|v| v["renderVersion"].as_u64())
            .map(|v| v as u32);
        let v = match listed {
            Some(v) => Some(v),
            None => self.stored_regions()?.is_empty().then_some(RENDER_VERSION),
        };
        index.render_version = Some(v);
        Ok(v)
    }

    /// The index, read from disk the first time.
    fn index_mut<'g>(&self, guard: &'g mut Option<MapIndex>) -> &'g mut MapIndex {
        guard.get_or_insert_with(|| {
            let mut index = MapIndex {
                hires: hires_masks(&self.out.join("hires")).into_iter().collect(),
                lod: vec![BTreeSet::new(); lod::LEVELS as usize],
                ..Default::default()
            };
            for [l, x, z] in lod_tiles(&self.out.join("lod")) {
                index.lod[l as usize - 1].insert((x, z));
            }
            index
        })
    }

    /// Records what a batch changed on disk.
    fn update_index(&self, results: &[RegionResult], lod_changed: &[[i32; 3]]) {
        let mut guard = self.index.lock().unwrap();
        let index = self.index_mut(&mut guard);
        for r in results {
            for &(x, z) in &r.tiles {
                let path = self
                    .out
                    .join("hires")
                    .join(format!("{}_{}", x.div_euclid(8), z.div_euclid(8)))
                    .join(format!("{x}_{z}.tmh"));
                index.set_hires(x, z, path.exists());
            }
        }
        for &[l, x, z] in lod_changed {
            let set = &mut index.lod[l as usize - 1];
            if lod::tile_path(&self.out, l as u32, x, z).exists() {
                set.insert((x, z));
            } else {
                set.remove(&(x, z));
            }
        }
    }

    fn write_textures(&self, force: bool) -> Result<()> {
        let path = self.out.join("textures.bin");
        if !force && path.exists() {
            return Ok(());
        }
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(9));
        gz.write_all(&self.reg.baker.textures.data)?;
        write_atomic(&path, &gz.finish()?)
    }

    fn write_manifest(&self, render_version: Option<u32>) -> Result<()> {
        let mut guard = self.index.lock().unwrap();
        let index = self.index_mut(&mut guard);
        let hires: Vec<[i64; 4]> = index
            .hires
            .iter()
            .map(|(&(rx, rz), &mask)| {
                [
                    rx as i64,
                    rz as i64,
                    (mask & 0xffff_ffff) as i64,
                    (mask >> 32) as i64,
                ]
            })
            .collect();
        let manifest = serde_json::json!({
            "id": self.cfg.id,
            "name": self.cfg.display_name(),
            "version": self.cfg.version,
            "renderVersion": render_version,
            "minY": self.cfg.min_y,
            "maxY": self.cfg.max_y,
            "tileSize": TILE,
            "textures": self.reg.baker.textures.count(),
            "hires": hires,
            "lodMissing": index.lod_missing(),
            "lodLevels": lod::LEVELS,
        });
        let json = serde_json::to_vec(&manifest)?;
        index.render_version = Some(render_version);
        if index.written.as_deref() == Some(&json) {
            return Ok(());
        }
        let mut gz = Vec::new();
        gzip_into(&json, &mut gz);
        write_atomic(&self.out.join("map.json"), &gz)?;
        index.written = Some(json);
        Ok(())
    }
}

/// A map's manifest as written, gzipped or from an older version in plain JSON.
fn read_manifest(path: &Path) -> Option<serde_json::Value> {
    let raw = std::fs::read(path).ok()?;
    if raw.starts_with(&[0x1f, 0x8b]) {
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(&raw[..])
            .read_to_end(&mut json)
            .ok()?;
        serde_json::from_slice(&json).ok()
    } else {
        serde_json::from_slice(&raw).ok()
    }
}

/// Which hires tiles are on disk, as a mask per region with bit
/// `(tz % 8) * 8 + tx % 8`.
fn hires_masks(dir: &Path) -> Vec<((i32, i32), u64)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for region in rd.flatten() {
        let name = region.file_name();
        let Some((rx, rz)) = name.to_str().and_then(parse_pair) else {
            continue;
        };
        let mut mask = 0u64;
        for tile in std::fs::read_dir(region.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            let name = tile.file_name();
            let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".tmh")) else {
                continue;
            };
            if let Some((tx, tz)) = parse_pair(stem) {
                mask |= 1 << (tz.rem_euclid(8) * 8 + tx.rem_euclid(8));
            }
        }
        if mask != 0 {
            out.push(((rx, rz), mask));
        }
    }
    out
}

/// LOD tiles on disk as [level, x, z], sorted.
fn lod_tiles(dir: &Path) -> Vec<[i32; 3]> {
    let mut out = Vec::new();
    for level in 1..=lod::LEVELS {
        for e in std::fs::read_dir(dir.join(level.to_string()))
            .into_iter()
            .flatten()
            .flatten()
        {
            let name = e.file_name();
            if let Some((x, z)) = name
                .to_str()
                .and_then(|n| n.strip_suffix(".tmh"))
                .and_then(parse_pair)
            {
                out.push([level as i32, x, z]);
            }
        }
    }
    out.sort_unstable();
    out
}

/// A region job: the region and the tiles in it to render.
type Job = ((i32, i32), BTreeSet<(i32, i32)>);

/// Region jobs, nearest the middle of them all first.
fn nearest_first(jobs: BTreeMap<(i32, i32), BTreeSet<(i32, i32)>>) -> Vec<Job> {
    let n = jobs.len().max(1) as f64;
    let (sx, sz) = jobs.keys().fold((0.0, 0.0), |(x, z), &(rx, rz)| {
        (x + rx as f64, z + rz as f64)
    });
    let d = |&(rx, rz): &(i32, i32)| (rx as f64 - sx / n).powi(2) + (rz as f64 - sz / n).powi(2);
    let mut jobs: Vec<Job> = jobs.into_iter().collect();
    jobs.sort_by(|a, b| d(&a.0).total_cmp(&d(&b.0)));
    jobs
}

/// Splits jobs into runs of at least `tiles` tiles. The last run may be shorter.
fn batches(jobs: &[Job], tiles: usize) -> Vec<&[Job]> {
    let mut out = Vec::new();
    let mut rest = jobs;
    while !rest.is_empty() {
        let mut n = 0;
        let mut sum = 0;
        while n < rest.len() && (n == 0 || sum < tiles) {
            sum += rest[n].1.len();
            n += 1;
        }
        let (batch, tail) = rest.split_at(n);
        out.push(batch);
        rest = tail;
    }
    out
}

fn parse_pair(s: &str) -> Option<(i32, i32)> {
    let (a, b) = s.split_once('_')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

struct RegionScan {
    rx: i32,
    rz: i32,
    hashes: Vec<u64>,
    dirty: Vec<usize>,
    stamp: Option<(u64, u64)>,
}

#[derive(Default)]
struct RegionResult {
    tiles: Vec<(i32, i32)>,
    empty: usize,
    failed: BTreeSet<(i32, i32)>,
    /// Hires tiles whose LOD source was rewritten or removed.
    sources: Vec<(i32, i32)>,
    bytes: u64,
    vertices: u64,
}

/// What building one LOD tile changed.
struct LodBuild {
    x: i32,
    z: i32,
    mesh_changed: bool,
    source_changed: bool,
}

/// Tiles whose volume (tile plus one-block halo) includes chunk (cx, cz).
fn tiles_touching(cx: i32, cz: i32) -> impl Iterator<Item = (i32, i32)> {
    let span = |c: i32| {
        let t = c.div_euclid(4);
        let lo = if c.rem_euclid(4) == 0 { t - 1 } else { t };
        let hi = if c.rem_euclid(4) == 3 { t + 1 } else { t };
        lo..=hi
    };
    let zs = span(cz);
    span(cx).flat_map(move |tx| zs.clone().map(move |tz| (tx, tz)))
}

fn vertex_count(tile: &[u8]) -> u64 {
    tile[20..132]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as u64 * 4)
        .sum()
}

thread_local! {
    /// A compressor is big, so each thread keeps one. libdeflate's levels aren't
    /// zlib's: this one is smaller than zlib's level 6 at a fraction of its time.
    static COMPRESSOR: std::cell::RefCell<libdeflater::Compressor> = std::cell::RefCell::new(
        libdeflater::Compressor::new(libdeflater::CompressionLvl::new(6).unwrap()),
    );
}

/// Replaces `out` with `data` gzipped.
pub fn gzip_into(data: &[u8], out: &mut Vec<u8>) {
    COMPRESSOR.with_borrow_mut(|c| {
        out.resize(c.gzip_compress_bound(data.len()), 0);
        let n = c.gzip_compress(data, out).expect("the bound fits");
        out.truncate(n);
    });
}

/// Writes a file unless it already holds exactly `data`. Returns whether it did.
pub fn write_if_changed(path: &Path, data: &[u8]) -> Result<bool> {
    if std::fs::read(path).is_ok_and(|old| old == data) {
        return Ok(false);
    }
    write_atomic(path, data)?;
    Ok(true)
}

pub fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!(
        "{}.tmp",
        path.extension().and_then(|e| e.to_str()).unwrap_or("")
    ));
    std::fs::write(&tmp, data).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Decoded chunks for one region job, loaded on demand and evicted by row.
struct ChunkSource<'a> {
    ctx: &'a MapCtx,
    regions: HashMap<(i32, i32), Option<Region>>,
    chunks: HashMap<(i32, i32), Option<Chunk>>,
    raw: Vec<u8>,
    buf: Vec<u8>,
    cache: DecodeCache,
}

impl<'a> ChunkSource<'a> {
    fn new(ctx: &'a MapCtx) -> Self {
        ChunkSource {
            ctx,
            regions: HashMap::new(),
            chunks: HashMap::new(),
            raw: Vec::new(),
            buf: Vec::new(),
            cache: DecodeCache::default(),
        }
    }

    fn evict_before(&mut self, cz: i32) {
        self.chunks.retain(|&(_, z), _| z >= cz);
    }

    fn load(&mut self, cx: i32, cz: i32) -> Result<()> {
        if self.chunks.contains_key(&(cx, cz)) {
            return Ok(());
        }
        let (rx, rz) = (cx.div_euclid(32), cz.div_euclid(32));
        let region = self.regions.entry((rx, rz)).or_insert_with(|| {
            let p = self.ctx.region_dir().join(format!("r.{rx}.{rz}.mca"));
            if p.exists() {
                Region::open(&p, rx, rz).ok()
            } else {
                None
            }
        });
        let chunk = match region {
            None => None,
            Some(r) => {
                let (lx, lz) = (cx.rem_euclid(32) as usize, cz.rem_euclid(32) as usize);
                if r.read_chunk(lx, lz, &mut self.raw, &mut self.buf)? {
                    decode_chunk(
                        &self.buf,
                        &self.ctx.reg,
                        self.ctx.cfg.min_y,
                        &mut self.cache,
                    )
                    .with_context(|| format!("chunk {cx},{cz}"))?
                } else {
                    None
                }
            }
        };
        self.chunks.insert((cx, cz), chunk);
        Ok(())
    }

    fn get(&self, cx: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&(cx, cz)).and_then(|c| c.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::MapConfig;
    use crate::model::Registry;
    use crate::world::decode_chunk;

    fn fixture(version: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(version)
    }

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tealmap-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A registry with no assets: every block bakes to a fallback cube.
    fn bare_registry() -> Registry {
        Registry::new(Baker::new(AssetFiles::new()))
    }

    fn map_config(world: &Path) -> MapConfig {
        MapConfig {
            id: "test".into(),
            name: None,
            world: world.to_owned(),
            version: "test".into(),
            client_jar: None,
            mods: None,
            min_y: -64,
            max_y: 320,
            remove_caves_below_y: 55,
        }
    }

    /// Every file under the map directory except state, keyed by relative path.
    fn snapshot(dir: &Path) -> BTreeMap<String, Vec<u8>> {
        fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(root, &p, out);
                } else {
                    let rel = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
                    out.insert(rel, std::fs::read(&p).unwrap());
                }
            }
        }
        let mut out = BTreeMap::new();
        for sub in ["hires", "lod"] {
            if dir.join(sub).exists() {
                walk(dir, &dir.join(sub), &mut out);
            }
        }
        out
    }

    fn hires_paths(snapshot: &BTreeMap<String, Vec<u8>>) -> BTreeSet<String> {
        snapshot
            .keys()
            .filter(|k| k.starts_with("hires/"))
            .cloned()
            .collect()
    }

    /// The hires tiles the map's manifest lists, as snapshot paths.
    fn listed_hires(map: &Path) -> BTreeSet<String> {
        let manifest = read_manifest(&map.join("map.json")).unwrap();
        let mut listed = BTreeSet::new();
        for e in manifest["hires"].as_array().unwrap() {
            let v: Vec<i64> = e
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_i64().unwrap())
                .collect();
            let mask = v[2] as u64 | (v[3] as u64) << 32;
            for bit in (0..64).filter(|b| mask >> b & 1 == 1) {
                let (tx, tz) = (v[0] * 8 + bit % 8, v[1] * 8 + bit / 8);
                listed.insert(format!("hires/{}_{}/{tx}_{tz}.tmh", v[0], v[1]));
            }
        }
        listed
    }

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for e in std::fs::read_dir(from).unwrap().flatten() {
            std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
        }
    }

    /// Points chunk `a`'s header entry at chunk `b`'s data, like a save that
    /// changed chunk `a`.
    fn swap_chunk(region: &Path, a: usize, b: usize) {
        let mut d = std::fs::read(region).unwrap();
        let (src, ts) = (d[b * 4..b * 4 + 4].to_vec(), 4096 + a * 4);
        d[a * 4..a * 4 + 4].copy_from_slice(&src);
        d[ts..ts + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        std::fs::write(region, d).unwrap();
    }

    #[test]
    fn fixture_chunks_decode() {
        for (version, min_full) in [("26.2", 64), ("1.21.1", 16)] {
            let reg = bare_registry();
            let bedrock = reg.id("minecraft:bedrock", &mut []);
            let (mut full, mut single, mut min_section) = (0, 0, i32::MAX);
            let mut cache = DecodeCache::default();
            let dir = fixture(version).join("region");
            for (rx, rz, path) in list_regions(&dir).unwrap() {
                let region = Region::open(&path, rx, rz).unwrap();
                let (mut raw, mut buf) = (Vec::new(), Vec::new());
                for i in 0..1024 {
                    if !region
                        .read_chunk(i % 32, i / 32, &mut raw, &mut buf)
                        .unwrap()
                    {
                        continue;
                    }
                    let Some(c) = decode_chunk(&buf, &reg, -64, &mut cache).unwrap() else {
                        continue;
                    };
                    full += 1;
                    min_section = min_section.min(c.sections[0].y);
                    single += c.sections.iter().filter(|s| s.blocks.is_none()).count();
                    // The bottom layer of the overworld is bedrock in every column.
                    let s = c.sections.iter().find(|s| s.y == -4).unwrap();
                    assert!(
                        (0..256).all(|i| match &s.blocks {
                            Some(b) => b[i] == bedrock,
                            None => s.single == bedrock,
                        }),
                        "{version}: y=-64 isn't all bedrock"
                    );
                }
            }
            assert!(full >= min_full, "{version}: only {full} full chunks");
            assert_eq!(min_section, -4, "{version}");
            assert!(single > 0, "{version}: no single-state sections");
        }
    }

    #[test]
    fn renders_are_deterministic_and_incremental_matches_full() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let base = temp_dir("render");
        let world = base.join("world");
        copy_dir(&fixture("26.2").join("region"), &world.join("region"));

        let render = |data: &str, full: bool| {
            let ctx = MapCtx::with_baker(
                map_config(&world),
                &base.join(data),
                Baker::new(AssetFiles::new()),
            )
            .unwrap();
            ctx.render(&pool, full, |_| {}).unwrap();
            snapshot(&base.join(data).join("maps/test"))
        };

        let first = render("a", true);
        assert!(
            first.keys().any(|k| k.starts_with("hires/-1_-1/")),
            "no tiles at negative coords"
        );
        assert!(first.keys().any(|k| k.starts_with("hires/0_0/")));
        assert_eq!(first, render("b", true), "two full renders differ");

        // The manifest's hires masks list exactly the hires tiles on disk.
        assert_eq!(listed_hires(&base.join("a/maps/test")), hires_paths(&first));

        // Chunk (1,1) of region 0,0 is inside tile (0,0); chunk (0,0) sits on a
        // tile corner and also touches tiles in three other regions.
        let region = world.join("region/r.0.0.mca");
        swap_chunk(&region, 32 + 1, 2 * 32 + 2);
        swap_chunk(&region, 0, 3 * 32 + 3);
        let ctx = MapCtx::with_baker(
            map_config(&world),
            &base.join("a"),
            Baker::new(AssetFiles::new()),
        )
        .unwrap();
        let up = ctx.render(&pool, false, |_| {}).unwrap();
        assert_eq!(up.stats.chunks, 2);
        assert!(up.hires.len() <= 4, "re-rendered {} tiles", up.hires.len());
        let incremental = snapshot(&base.join("a/maps/test"));
        assert_ne!(first, incremental);
        assert_eq!(
            incremental,
            render("c", true),
            "incremental render differs from a full one"
        );

        // A full render after a region disappears must not keep its old tiles.
        // The copy sits far from the others so no neighbor's halo reaches it.
        let far = world.join("region/r.5.5.mca");
        std::fs::copy(world.join("region/r.-1.-1.mca"), &far).unwrap();
        assert!(
            render("a", true)
                .keys()
                .any(|k| k.starts_with("hires/5_5/"))
        );
        std::fs::remove_file(&far).unwrap();
        assert_eq!(
            render("a", true),
            render("d", true),
            "full render kept stale tiles"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn batched_passes_fill_in_and_resume() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let base = temp_dir("batches");
        let world = base.join("world");
        copy_dir(&fixture("26.2").join("region"), &world.join("region"));
        // A region far from the rest, so the ones near the middle finish (and
        // save their state) before the pass reaches it.
        std::fs::copy(
            world.join("region/r.0.0.mca"),
            world.join("region/r.5.5.mca"),
        )
        .unwrap();
        let ctx = |data: &str, batch_tiles: usize| {
            let mut ctx = MapCtx::with_baker(
                map_config(&world),
                &base.join(data),
                Baker::new(AssetFiles::new()),
            )
            .unwrap();
            ctx.batch_tiles = batch_tiles;
            ctx
        };
        let whole_pass = ctx("a", BATCH_TILES_PER_THREAD)
            .render(&pool, false, |_| {})
            .unwrap();
        let whole = snapshot(&base.join("a/maps/test"));

        // One region job per batch. Each batch publishes its tiles, with a
        // manifest that lists exactly the hires tiles on disk so far.
        let map = base.join("b/maps/test");
        let mut batches = 0;
        let mut published = BTreeSet::new();
        let pass = ctx("b", 1)
            .render(&pool, false, |up| {
                batches += 1;
                published.extend(up.hires.iter().copied());
                assert_eq!(listed_hires(&map), hires_paths(&snapshot(&map)));
            })
            .unwrap();
        assert!(batches > 2, "only {batches} batches");
        assert_eq!(published.len(), pass.hires.len());
        assert_eq!(
            snapshot(&map),
            whole,
            "batched render differs from a whole one"
        );

        // A pass that stops before its last batch resumes where it stopped.
        let mut left = batches - 1;
        let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ctx("c", 1).render(&pool, false, |_| {
                left -= 1;
                assert!(left > 0, "stopping the pass");
            })
        }));
        assert!(stopped.is_err());
        let resumed = ctx("c", BATCH_TILES_PER_THREAD)
            .render(&pool, false, |_| {})
            .unwrap();
        assert!(
            resumed.stats.chunks > 0 && resumed.stats.chunks < whole_pass.stats.chunks,
            "resumed pass rendered {} of {} chunks",
            resumed.stats.chunks,
            whole_pass.stats.chunks
        );
        assert_eq!(
            snapshot(&base.join("c/maps/test")),
            whole,
            "resumed render differs from a whole one"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn halo_tiles() {
        // Interior chunk of a tile touches only that tile.
        assert_eq!(tiles_touching(1, 1).collect::<Vec<_>>(), vec![(0, 0)]);
        // Corner chunk touches four tiles.
        let mut t: Vec<_> = tiles_touching(0, 0).collect();
        t.sort();
        assert_eq!(t, vec![(-1, -1), (-1, 0), (0, -1), (0, 0)]);
        // Negative coordinates.
        let mut t: Vec<_> = tiles_touching(-1, -6).collect();
        t.sort();
        assert_eq!(t, vec![(-1, -2), (0, -2)]);
    }
}
