//! Lowres layers: one pixel per block column (level 1, one tile per region)
//! and one pixel per 8×8 columns (level 2, one tile per 8×8 regions).
//!
//! Each tile is a 512×1024 RGB PNG. The top half is color. The bottom half is
//! height: R is the high byte and G the low byte of `top y - min_y + 1`, with
//! 0 meaning no data.

use crate::mesh::{Column, TILE};
use crate::render::write_atomic;
use anyhow::Result;
use std::path::{Path, PathBuf};

pub const SIZE: usize = 512;

pub struct Level1 {
    color: Vec<[u8; 3]>,
    height: Vec<u16>,
    dirty: bool,
}

fn path(out: &Path, level: u8, x: i32, z: i32) -> PathBuf {
    out.join("lowres")
        .join(level.to_string())
        .join(format!("{x}_{z}.png"))
}

impl Level1 {
    pub fn empty() -> Self {
        Level1 {
            color: vec![[0; 3]; SIZE * SIZE],
            height: vec![0; SIZE * SIZE],
            dirty: false,
        }
    }

    pub fn load(out: &Path, rx: i32, rz: i32) -> Self {
        let mut l = Self::empty();
        if let Some((c, h)) = read(&path(out, 1, rx, rz)) {
            l.color = c;
            l.height = h;
        }
        l
    }

    /// Writes a tile's 64×64 columns at local offset (lx, lz).
    pub fn put(&mut self, lx: usize, lz: usize, cols: &[Column]) {
        let t = TILE as usize;
        for z in 0..t {
            for x in 0..t {
                let c = cols[z * t + x];
                let i = (lz + z) * SIZE + lx + x;
                self.color[i] = c.color;
                self.height[i] = c.height;
            }
        }
        self.dirty = true;
    }

    pub fn clear(&mut self, lx: usize, lz: usize) {
        let t = TILE as usize;
        for z in 0..t {
            let i = (lz + z) * SIZE + lx;
            self.color[i..i + t].fill([0; 3]);
            self.height[i..i + t].fill(0);
        }
        self.dirty = true;
    }

    /// Saves if anything changed. Returns whether the file changed.
    pub fn save(&self, out: &Path, rx: i32, rz: i32) -> Result<bool> {
        if !self.dirty {
            return Ok(false);
        }
        write_or_remove(&path(out, 1, rx, rz), &self.color, &self.height)?;
        Ok(true)
    }
}

/// Rebuilds the level 2 tile (qx, qz) from its 8×8 level 1 tiles.
pub fn build_level2(out: &Path, qx: i32, qz: i32) -> Result<()> {
    let mut color = vec![[0u8; 3]; SIZE * SIZE];
    let mut height = vec![0u16; SIZE * SIZE];
    let per = SIZE / 8; // L2 pixels per region
    for dz in 0..8 {
        for dx in 0..8 {
            let Some((c, h)) = read(&path(out, 1, qx * 8 + dx, qz * 8 + dz)) else {
                continue;
            };
            for pz in 0..per {
                for px in 0..per {
                    let (mut sum, mut hs, mut n) = ([0u32; 3], 0u32, 0u32);
                    for z in 0..8 {
                        for x in 0..8 {
                            let i = (pz * 8 + z) * SIZE + px * 8 + x;
                            if h[i] != 0 {
                                for k in 0..3 {
                                    sum[k] += c[i][k] as u32;
                                }
                                hs += h[i] as u32;
                                n += 1;
                            }
                        }
                    }
                    if n > 0 {
                        let o = (dz as usize * per + pz) * SIZE + dx as usize * per + px;
                        color[o] = [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8];
                        height[o] = (hs / n) as u16;
                    }
                }
            }
        }
    }
    write_or_remove(&path(out, 2, qx, qz), &color, &height)
}

fn write_or_remove(p: &Path, color: &[[u8; 3]], height: &[u16]) -> Result<()> {
    if height.iter().all(|&h| h == 0) {
        if p.exists() {
            std::fs::remove_file(p)?;
        }
        return Ok(());
    }
    let mut rgb = Vec::with_capacity(SIZE * SIZE * 2 * 3);
    for c in color {
        rgb.extend_from_slice(c);
    }
    for &h in height {
        rgb.extend_from_slice(&[(h >> 8) as u8, h as u8, 0]);
    }
    let mut png_bytes = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut png_bytes, SIZE as u32, (SIZE * 2) as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(png::Compression::Balanced);
        let mut w = enc.write_header()?;
        w.write_image_data(&rgb)?;
    }
    write_atomic(p, &png_bytes)
}

fn read(p: &Path) -> Option<(Vec<[u8; 3]>, Vec<u16>)> {
    let f = std::fs::File::open(p).ok()?;
    let dec = png::Decoder::new(std::io::BufReader::new(f));
    let mut r = dec.read_info().ok()?;
    let mut buf = vec![0; r.output_buffer_size()?];
    let info = r.next_frame(&mut buf).ok()?;
    if info.width as usize != SIZE
        || info.height as usize != SIZE * 2
        || info.color_type != png::ColorType::Rgb
    {
        return None;
    }
    let half = SIZE * SIZE * 3;
    let color = buf[..half]
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let height = buf[half..half * 2]
        .chunks_exact(3)
        .map(|c| (c[0] as u16) << 8 | c[1] as u16)
        .collect();
    Some((color, height))
}

/// Lists tiles in a lowres level directory as [x, z] pairs.
pub fn list(dir: &Path) -> Vec<[i32; 2]> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name();
            let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".png")) else {
                continue;
            };
            if let Some((x, z)) = stem.split_once('_')
                && let (Ok(x), Ok(z)) = (x.parse(), z.parse())
            {
                out.push([x, z]);
            }
        }
    }
    out.sort();
    out
}
