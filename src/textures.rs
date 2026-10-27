//! Block textures packed as fixed 16×16 RGBA layers for a texture array.
//!
//! Layers are assigned in sorted path order so output is deterministic.

use crate::assets::{AssetFiles, split_id};
use std::collections::HashMap;

pub const SIZE: usize = 16;
pub const LAYER_BYTES: usize = SIZE * SIZE * 4;
/// Magenta/black checker, for missing textures.
pub const MISSING: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alpha {
    Opaque,
    /// Only fully transparent or fully opaque pixels.
    Cutout,
    /// Has partially transparent pixels.
    Translucent,
}

pub struct TexInfo {
    pub alpha: Alpha,
    /// Alpha-weighted average color.
    pub avg: [u8; 3],
}

pub struct Textures {
    index: HashMap<String, u16>,
    pub info: Vec<TexInfo>,
    /// All layers back to back, `LAYER_BYTES` each.
    pub data: Vec<u8>,
}

impl Textures {
    pub fn load(assets: &AssetFiles) -> Self {
        let mut paths: Vec<&str> = assets
            .paths()
            .filter(|p| p.contains("/textures/block/") && p.ends_with(".png"))
            .collect();
        paths.sort_unstable();

        let mut t = Textures {
            index: HashMap::new(),
            info: Vec::new(),
            data: Vec::new(),
        };
        // Layer 0 is plain white, layer 1 the missing-texture checker.
        t.push(&[255u8; LAYER_BYTES]);
        let mut missing = [0u8; LAYER_BYTES];
        for y in 0..SIZE {
            for x in 0..SIZE {
                let i = (y * SIZE + x) * 4;
                let on = (x < 8) != (y < 8);
                missing[i..i + 4].copy_from_slice(if on {
                    &[248, 0, 248, 255]
                } else {
                    &[0, 0, 0, 255]
                });
            }
        }
        t.push(&missing);

        for path in paths {
            let Some(rgba) = decode_first_frame(assets.get(path).unwrap()) else {
                continue;
            };
            // assets/<ns>/textures/block/x.png -> <ns>:block/x
            let rest = &path["assets/".len()..];
            let (ns, p) = rest.split_once('/').unwrap();
            let p = p.trim_start_matches("textures/").trim_end_matches(".png");
            let id = t.push(&rgba);
            t.index.insert(format!("{ns}:{p}"), id);
        }
        t
    }

    fn push(&mut self, rgba: &[u8]) -> u16 {
        let id = self.info.len() as u16;
        self.info.push(analyze(rgba));
        self.data.extend_from_slice(rgba);
        id
    }

    /// Looks up a texture by model reference, e.g. `block/stone` or `minecraft:block/stone`.
    pub fn get(&self, id: &str) -> Option<u16> {
        let (ns, p) = split_id(id);
        if ns == "minecraft" {
            self.index.get(&format!("minecraft:{p}")).copied()
        } else {
            self.index.get(&format!("{ns}:{p}")).copied()
        }
    }

    pub fn count(&self) -> usize {
        self.info.len()
    }
}

fn analyze(rgba: &[u8]) -> TexInfo {
    let mut alpha = Alpha::Opaque;
    let (mut r, mut g, mut b, mut w) = (0u64, 0u64, 0u64, 0u64);
    for px in rgba.chunks_exact(4) {
        let a = px[3] as u64;
        match px[3] {
            255 => {}
            0 => {
                if alpha == Alpha::Opaque {
                    alpha = Alpha::Cutout
                }
            }
            _ => alpha = Alpha::Translucent,
        }
        r += px[0] as u64 * a;
        g += px[1] as u64 * a;
        b += px[2] as u64 * a;
        w += a;
    }
    let avg = if w == 0 {
        [0, 0, 0]
    } else {
        [(r / w) as u8, (g / w) as u8, (b / w) as u8]
    };
    TexInfo { alpha, avg }
}

/// Decodes a PNG to 16×16 RGBA. Animated strips use their first frame;
/// other sizes are resampled with nearest-neighbor.
fn decode_first_frame(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return None,
    };
    let stride = info.line_size;
    let frame = w.min(h);
    let mut out = vec![0u8; LAYER_BYTES];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let sx = x * frame / SIZE;
            let sy = y * frame / SIZE;
            let s = &buf[sy * stride + sx * channels..][..channels];
            let px = match channels {
                1 => [s[0], s[0], s[0], 255],
                2 => [s[0], s[0], s[0], s[1]],
                3 => [s[0], s[1], s[2], 255],
                _ => [s[0], s[1], s[2], s[3]],
            };
            out[(y * SIZE + x) * 4..][..4].copy_from_slice(&px);
        }
    }
    Some(out)
}

/// Decodes a 256×256 colormap PNG to RGB.
pub fn decode_colormap(bytes: &[u8]) -> Option<Vec<[u8; 3]>> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let channels = info.color_type.samples();
    if info.width != 256 || info.height != 256 || channels < 3 {
        return None;
    }
    Some(
        (0..256 * 256)
            .map(|i| {
                let s = &buf[(i / 256) * info.line_size + (i % 256) * channels..];
                [s[0], s[1], s[2]]
            })
            .collect(),
    )
}
