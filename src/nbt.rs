//! Minimal, zero-copy NBT reader.
//!
//! Arrays and strings borrow from the input buffer, and [`Reader`] walks the
//! data without allocating, so callers pay only for the tags they read.

use anyhow::{Result, bail};

// Every tag type is parsed so unknown data can be skipped or kept intact,
// even where tealmap never reads the value.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Value<'a> {
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    ByteArray(&'a [u8]),
    /// Raw modified-UTF-8 bytes. Block and biome ids are plain ASCII.
    String(&'a [u8]),
    List(Vec<Value<'a>>),
    Compound(Vec<(&'a [u8], Value<'a>)>),
    /// Raw big-endian bytes, 4 per element.
    IntArray(&'a [u8]),
    /// Raw big-endian bytes, 8 per element.
    LongArray(&'a [u8]),
}

impl<'a> Value<'a> {
    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        match self {
            Value::Compound(entries) => entries
                .iter()
                .find(|(k, _)| *k == key.as_bytes())
                .map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Value::Byte(v) => Some(v as i64),
            Value::Short(v) => Some(v as i64),
            Value::Int(v) => Some(v as i64),
            Value::Long(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&'a str> {
        match *self {
            Value::String(s) => std::str::from_utf8(s).ok(),
            _ => None,
        }
    }

    pub fn as_compound(&self) -> Option<&[(&'a [u8], Value<'a>)]> {
        match self {
            Value::Compound(v) => Some(v),
            _ => None,
        }
    }
}

/// A cursor over NBT data, for walking it without building a tree.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    /// Moves past the root tag and its name.
    pub fn enter_root(&mut self) -> Result<()> {
        if self.u8()? != 10 {
            bail!("nbt: root is not a compound");
        }
        self.string()?;
        Ok(())
    }

    /// The next entry of the current compound as (tag, name), or None at its end.
    #[inline]
    pub fn next_entry(&mut self) -> Result<Option<(u8, &'a [u8])>> {
        let tag = self.u8()?;
        if tag == 0 {
            return Ok(None);
        }
        Ok(Some((tag, self.string()?)))
    }

    /// A list's element tag and length.
    #[inline]
    pub fn list_header(&mut self) -> Result<(u8, usize)> {
        let elem = self.u8()?;
        Ok((elem, self.len()?))
    }

    /// The payload of a byte array (tag 7) or a long array (tag 12), still in
    /// file byte order.
    #[inline]
    pub fn array(&mut self, tag: u8) -> Result<&'a [u8]> {
        let n = self.len()?;
        match tag {
            7 => self.take(n),
            12 => self.take(n.checked_mul(8).ok_or_else(too_long)?),
            _ => bail!("nbt: tag {tag} is not an array"),
        }
    }

    /// Skips one payload and returns its raw bytes.
    #[inline]
    pub fn raw(&mut self, tag: u8) -> Result<&'a [u8]> {
        let start = self.pos;
        self.skip(tag, 1)?;
        Ok(&self.buf[start..self.pos])
    }

    /// Skips one payload.
    #[inline]
    pub fn skip_payload(&mut self, tag: u8) -> Result<()> {
        self.skip(tag, 1)
    }

    /// Parses one payload into a tree.
    pub fn value(&mut self, tag: u8) -> Result<Value<'a>> {
        self.payload(tag, 1)
    }

    #[inline]
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        match end {
            Some(end) => {
                let s = &self.buf[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => bail!("nbt: unexpected end of data"),
        }
    }
    #[inline]
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    #[inline]
    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }
    #[inline]
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    #[inline]
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn len(&mut self) -> Result<usize> {
        let n = self.i32()?;
        if n < 0 {
            bail!("nbt: negative length");
        }
        Ok(n as usize)
    }
    pub fn string(&mut self) -> Result<&'a [u8]> {
        let n = self.i16()? as u16 as usize;
        self.take(n)
    }

    fn payload(&mut self, tag: u8, depth: u32) -> Result<Value<'a>> {
        if depth > 512 {
            bail!("nbt: nesting too deep");
        }
        Ok(match tag {
            1 => Value::Byte(self.u8()? as i8),
            2 => Value::Short(self.i16()?),
            3 => Value::Int(self.i32()?),
            4 => Value::Long(self.i64()?),
            5 => Value::Float(f32::from_bits(self.i32()? as u32)),
            6 => Value::Double(f64::from_bits(self.i64()? as u64)),
            7 => {
                let n = self.len()?;
                Value::ByteArray(self.take(n)?)
            }
            8 => Value::String(self.string()?),
            9 => {
                let elem = self.u8()?;
                let n = self.len()?;
                let mut items = Vec::with_capacity(n.min(4096));
                for _ in 0..n {
                    items.push(self.payload(elem, depth + 1)?);
                }
                Value::List(items)
            }
            10 => {
                let mut entries = Vec::new();
                loop {
                    let t = self.u8()?;
                    if t == 0 {
                        break;
                    }
                    let name = self.string()?;
                    entries.push((name, self.payload(t, depth + 1)?));
                }
                Value::Compound(entries)
            }
            11 => {
                let n = self.len()?;
                Value::IntArray(self.take(n * 4)?)
            }
            12 => {
                let n = self.len()?;
                Value::LongArray(self.take(n * 8)?)
            }
            _ => bail!("nbt: unknown tag {tag}"),
        })
    }

    fn skip(&mut self, tag: u8, depth: u32) -> Result<()> {
        if depth > 512 {
            bail!("nbt: nesting too deep");
        }
        match tag {
            1 => self.pos += 1,
            2 => self.pos += 2,
            3 | 5 => self.pos += 4,
            4 | 6 => self.pos += 8,
            7 => {
                let n = self.len()?;
                self.take(n)?;
            }
            8 => {
                self.string()?;
            }
            9 => {
                let elem = self.u8()?;
                let n = self.len()?;
                match elem {
                    0 => {}
                    1 => {
                        self.take(n)?;
                    }
                    2 => {
                        self.take(n * 2)?;
                    }
                    3 | 5 => {
                        self.take(n * 4)?;
                    }
                    4 | 6 => {
                        self.take(n * 8)?;
                    }
                    _ => {
                        for _ in 0..n {
                            self.skip(elem, depth + 1)?;
                        }
                    }
                }
            }
            10 => loop {
                let t = self.u8()?;
                if t == 0 {
                    break;
                }
                self.string()?;
                self.skip(t, depth + 1)?;
            },
            11 => {
                let n = self.len()?;
                self.take(n * 4)?;
            }
            12 => {
                let n = self.len()?;
                self.take(n * 8)?;
            }
            _ => bail!("nbt: unknown tag {tag}"),
        }
        if self.pos > self.buf.len() {
            bail!("nbt: unexpected end of data");
        }
        Ok(())
    }
}

fn too_long() -> anyhow::Error {
    anyhow::anyhow!("nbt: array too long")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<u8> {
        let mut b = vec![10, 0, 0];
        // Int "a" = 7
        b.extend([3, 0, 1, b'a', 0, 0, 0, 7]);
        // List "skip" of 2 compounds, each with a string
        b.extend([9, 0, 4, b's', b'k', b'i', b'p', 10, 0, 0, 0, 2]);
        for _ in 0..2 {
            b.extend([8, 0, 1, b'n', 0, 2, b'h', b'i', 0]);
        }
        // LongArray "l" = [1, -1]
        b.extend([12, 0, 1, b'l', 0, 0, 0, 2]);
        b.extend(1u64.to_be_bytes());
        b.extend(u64::MAX.to_be_bytes());
        b.push(0);
        b
    }

    /// Walks a root compound, skipping `skip` and collecting the rest.
    fn walk(data: &[u8]) -> Result<Vec<(Vec<u8>, Value<'_>)>> {
        let mut r = Reader::new(data);
        r.enter_root()?;
        let mut out = Vec::new();
        while let Some((tag, name)) = r.next_entry()? {
            if name == b"skip" {
                r.skip_payload(tag)?;
            } else {
                out.push((name.to_vec(), r.value(tag)?));
            }
        }
        Ok(out)
    }

    #[test]
    fn walks_and_skips() {
        let data = sample();
        let entries = walk(&data).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, b"a");
        assert_eq!(entries[0].1.as_i64(), Some(7));
        assert!(matches!(entries[1].1, Value::LongArray(raw) if raw.len() == 16));
    }

    #[test]
    fn truncated_input_errors() {
        let data = sample();
        assert!(walk(&data[..data.len() - 5]).is_err());
    }
}
