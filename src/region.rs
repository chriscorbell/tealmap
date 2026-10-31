//! Anvil region files (`r.X.Z.mca`): header parsing and chunk decompression.

use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const SECTOR: usize = 4096;

/// An open region file. Only the 8 KiB header is kept in memory; chunks are
/// read with positional reads on demand.
pub struct Region {
    pub x: i32,
    pub z: i32,
    path: PathBuf,
    file: std::fs::File,
    header: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkEntry {
    pub sector: u32,
    pub sectors: u8,
    pub timestamp: u32,
}

impl Region {
    pub fn open(path: &Path, x: i32, z: i32) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut header = vec![0u8; 2 * SECTOR];
        let n = read_at_most(&file, &mut header, 0)?;
        header.truncate(if n < 2 * SECTOR { 0 } else { n });
        Ok(Region {
            x,
            z,
            path: path.to_owned(),
            file,
            header,
        })
    }

    /// Header entry for a chunk at local coords (0..32).
    pub fn entry(&self, lx: usize, lz: usize) -> ChunkEntry {
        let i = (lx & 31) + (lz & 31) * 32;
        if self.header.len() < 2 * SECTOR {
            return ChunkEntry::default();
        }
        let loc = &self.header[i * 4..i * 4 + 4];
        let ts = &self.header[SECTOR + i * 4..SECTOR + i * 4 + 4];
        ChunkEntry {
            sector: u32::from_be_bytes([0, loc[0], loc[1], loc[2]]),
            sectors: loc[3],
            timestamp: u32::from_be_bytes(ts.try_into().unwrap()),
        }
    }

    /// Reads a chunk's raw (still compressed) payload into `buf` and returns its
    /// compression type. Used to hash chunks without decompressing them.
    pub fn raw_chunk(&self, lx: usize, lz: usize, buf: &mut Vec<u8>) -> Result<Option<u8>> {
        let e = self.entry(lx, lz);
        if e.sector < 2 || e.sectors == 0 {
            return Ok(None);
        }
        let start = e.sector as u64 * SECTOR as u64;
        buf.clear();
        buf.resize(e.sectors as usize * SECTOR, 0);
        let n = read_at_most(&self.file, buf, start)?;
        if n < 5 {
            bail!("chunk {lx},{lz} points past end of {}", self.path.display());
        }
        let len = u32::from_be_bytes(buf[0..4].try_into().unwrap()) as usize;
        if len == 0 || 4 + len > n {
            bail!("chunk {lx},{lz} has bad length in {}", self.path.display());
        }
        let kind = buf[4];
        buf.truncate(4 + len);
        buf.drain(..5);
        Ok(Some(kind))
    }

    /// Decompresses a chunk into `out`. Returns false if the chunk doesn't exist.
    pub fn read_chunk(
        &self,
        lx: usize,
        lz: usize,
        raw: &mut Vec<u8>,
        out: &mut Vec<u8>,
    ) -> Result<bool> {
        out.clear();
        let Some(kind) = self.raw_chunk(lx, lz, raw)? else {
            return Ok(false);
        };
        if kind & 0x80 != 0 {
            // Oversized chunk stored next to the region as c.X.Z.mcc.
            let cx = self.x * 32 + lx as i32;
            let cz = self.z * 32 + lz as i32;
            let ext = self.path.with_file_name(format!("c.{cx}.{cz}.mcc"));
            let data = std::fs::read(&ext).with_context(|| format!("reading {}", ext.display()))?;
            decompress(kind & 0x7f, &data, out)?;
        } else {
            decompress(kind, raw, out)?;
        }
        Ok(true)
    }
}

fn read_at_most(file: &std::fs::File, buf: &mut [u8], offset: u64) -> Result<usize> {
    use std::os::unix::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        match file.read_at(&mut buf[done..], offset + done as u64) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(done)
}

pub fn decompress(kind: u8, data: &[u8], out: &mut Vec<u8>) -> Result<()> {
    match kind {
        1 => {
            flate2::read::GzDecoder::new(data).read_to_end(out)?;
        }
        2 => {
            flate2::read::ZlibDecoder::new(data).read_to_end(out)?;
        }
        3 => out.extend_from_slice(data),
        4 => lz4_block_stream(data, out)?,
        _ => bail!("unsupported chunk compression type {kind}"),
    }
    Ok(())
}

/// lz4-java's LZ4BlockOutputStream format, which Minecraft uses for type 4.
fn lz4_block_stream(mut data: &[u8], out: &mut Vec<u8>) -> Result<()> {
    const MAGIC: &[u8] = b"LZ4Block";
    while !data.is_empty() {
        if data.len() < 21 || &data[..8] != MAGIC {
            bail!("bad LZ4 block header");
        }
        let token = data[8];
        let clen = u32::from_le_bytes(data[9..13].try_into().unwrap()) as usize;
        let dlen = u32::from_le_bytes(data[13..17].try_into().unwrap()) as usize;
        data = &data[21..];
        if clen == 0 && dlen == 0 {
            break;
        }
        if clen > data.len() {
            bail!("truncated LZ4 block");
        }
        let block = &data[..clen];
        match token & 0xf0 {
            0x10 => out.extend_from_slice(block),
            0x20 => {
                let start = out.len();
                out.resize(start + dlen, 0);
                let n = lz4_flex::block::decompress_into(block, &mut out[start..])?;
                out.truncate(start + n);
            }
            m => bail!("unknown LZ4 block method {m:#x}"),
        }
        data = &data[clen..];
    }
    Ok(())
}

/// Lists `r.X.Z.mca` files in a region directory.
pub fn list_regions(dir: &Path) -> Result<Vec<(i32, i32, PathBuf)>> {
    let mut out = Vec::new();
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e).with_context(|| format!("listing {}", dir.display())),
    };
    for ent in rd {
        let ent = ent?;
        let name = ent.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some((x, z)) = parse_region_name(name) {
            out.push((x, z, ent.path()));
        }
    }
    out.sort_by_key(|&(x, z, _)| (z, x));
    Ok(out)
}

pub fn parse_region_name(name: &str) -> Option<(i32, i32)> {
    let rest = name.strip_prefix("r.")?.strip_suffix(".mca")?;
    let (x, z) = rest.split_once('.')?;
    Some((x.parse().ok()?, z.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_names() {
        assert_eq!(parse_region_name("r.-3.12.mca"), Some((-3, 12)));
        assert_eq!(parse_region_name("r.0.0.mcc"), None);
        assert_eq!(parse_region_name("c.0.0.mcc"), None);
    }

    #[test]
    fn lz4_stream_roundtrip() {
        let raw = b"hello hello hello hello hello tealmap".repeat(20);
        let comp = lz4_flex::block::compress(&raw);
        let mut stream = Vec::new();
        stream.extend_from_slice(b"LZ4Block");
        stream.push(0x20 | 6);
        stream.extend((comp.len() as u32).to_le_bytes());
        stream.extend((raw.len() as u32).to_le_bytes());
        stream.extend(0u32.to_le_bytes());
        stream.extend_from_slice(&comp);
        stream.extend_from_slice(b"LZ4Block");
        stream.push(0x10);
        stream.extend([0u8; 12]);
        let mut out = Vec::new();
        decompress(4, &stream, &mut out).unwrap();
        assert_eq!(out, raw);
    }
}
