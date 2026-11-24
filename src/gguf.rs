use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

/// What `/api/show` needs out of a GGUF header. Missing keys stay missing; the
/// API falls back to its own defaults rather than guessing a value.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Metadata {
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub file_type: Option<u32>,
    pub parameter_count: Option<u64>,
}

const MAGIC: &[u8; 4] = b"GGUF";
const MAX_KEYS: u64 = 1 << 20;
const MAX_STRING_BYTES: u64 = 1 << 20;
const MAX_ARRAY_ITEMS: u64 = 1 << 28;

const TYPE_U8: u32 = 0;
const TYPE_I8: u32 = 1;
const TYPE_U16: u32 = 2;
const TYPE_I16: u32 = 3;
const TYPE_U32: u32 = 4;
const TYPE_I32: u32 = 5;
const TYPE_F32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_U64: u32 = 10;
const TYPE_I64: u32 = 11;
const TYPE_F64: u32 = 12;

/// Scan the key/value header of a GGUF file, stopping before the tensor table.
pub fn read(path: &Path) -> io::Result<Metadata> {
    read_from(&mut BufReader::new(File::open(path)?))
}

fn read_from<R: BufRead>(reader: &mut R) -> io::Result<Metadata> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("not a GGUF file"));
    }
    let version = read_u32(reader)?;
    if !(2..=3).contains(&version) {
        return Err(invalid(&format!("unsupported GGUF version {version}")));
    }
    let _tensor_count = read_u64(reader)?;
    let key_count = read_u64(reader)?;
    if key_count > MAX_KEYS {
        return Err(invalid("GGUF metadata has too many keys"));
    }

    let mut metadata = Metadata::default();
    let mut contexts = Vec::new();
    for _ in 0..key_count {
        let key = read_string(reader)?;
        let kind = read_u32(reader)?;
        match kind {
            TYPE_STRING => {
                if key == "general.architecture" {
                    metadata.architecture = Some(read_string(reader)?);
                } else {
                    skip_string(reader)?;
                }
            }
            TYPE_ARRAY => skip_array(reader)?,
            _ => {
                let number = read_number(reader, kind)?;
                match key.as_str() {
                    "general.file_type" => {
                        metadata.file_type = number.and_then(|n| u32::try_from(n).ok());
                    }
                    "general.parameter_count" => metadata.parameter_count = number,
                    key if key.ends_with(".context_length") => {
                        if let Some(n) = number.and_then(|n| u32::try_from(n).ok()) {
                            contexts.push((key.to_owned(), n));
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    metadata.context_length = metadata.architecture.as_ref().and_then(|architecture| {
        let key = format!("{architecture}.context_length");
        contexts
            .iter()
            .find(|(candidate, _)| candidate == &key)
            .map(|(_, length)| *length)
    });
    Ok(metadata)
}

fn read_number<R: Read>(reader: &mut R, kind: u32) -> io::Result<Option<u64>> {
    Ok(match kind {
        TYPE_U8 | TYPE_BOOL => Some(u64::from(read_u8(reader)?)),
        TYPE_I8 => u64::try_from(i64::from(read_u8(reader)? as i8)).ok(),
        TYPE_U16 => Some(u64::from(read_u16(reader)?)),
        TYPE_I16 => u64::try_from(i64::from(read_u16(reader)? as i16)).ok(),
        TYPE_U32 | TYPE_F32 => Some(u64::from(read_u32(reader)?)),
        TYPE_I32 => u64::try_from(i64::from(read_u32(reader)? as i32)).ok(),
        TYPE_U64 | TYPE_F64 => Some(read_u64(reader)?),
        TYPE_I64 => u64::try_from(read_u64(reader)? as i64).ok(),
        _ => return Err(invalid(&format!("unknown GGUF value type {kind}"))),
    })
}

fn skip_array<R: BufRead>(reader: &mut R) -> io::Result<()> {
    let element = read_u32(reader)?;
    let count = read_u64(reader)?;
    if count > MAX_ARRAY_ITEMS {
        return Err(invalid("GGUF array has too many items"));
    }
    if element == TYPE_ARRAY {
        return Err(invalid("nested GGUF arrays are not supported"));
    }
    if element == TYPE_STRING {
        for _ in 0..count {
            skip_string(reader)?;
        }
        return Ok(());
    }
    let width = element_width(element).ok_or_else(|| invalid("unknown GGUF array type"))?;
    let bytes = count
        .checked_mul(u64::from(width))
        .ok_or_else(|| invalid("GGUF array is too large"))?;
    skip_bytes(reader, bytes)
}

fn element_width(kind: u32) -> Option<u8> {
    Some(match kind {
        TYPE_U8 | TYPE_I8 | TYPE_BOOL => 1,
        TYPE_U16 | TYPE_I16 => 2,
        TYPE_U32 | TYPE_I32 | TYPE_F32 => 4,
        TYPE_U64 | TYPE_I64 | TYPE_F64 => 8,
        _ => return None,
    })
}

fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let len = read_u64(reader)?;
    if len > MAX_STRING_BYTES {
        return Err(invalid("GGUF string is too long"));
    }
    let mut bytes = vec![0u8; len as usize];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| invalid("GGUF string is not UTF-8"))
}

fn skip_string<R: BufRead>(reader: &mut R) -> io::Result<()> {
    let len = read_u64(reader)?;
    skip_bytes(reader, len)
}

fn skip_bytes<R: BufRead>(reader: &mut R, mut bytes: u64) -> io::Result<()> {
    while bytes > 0 {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            return Err(invalid("GGUF metadata ends early"));
        }
        let take = usize::try_from(bytes).map_or(buffer.len(), |left| buffer.len().min(left));
        reader.consume(take);
        bytes -= take as u64;
    }
    Ok(())
}

fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
    let mut bytes = [0u8; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u16<R: Read>(reader: &mut R) -> io::Result<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
pub(crate) mod test_support {
    pub enum Value<'a> {
        Str(&'a str),
        StrArray(&'a [&'a str]),
        U32(u32),
        U64(u64),
    }

    /// A GGUF header with only a key/value table, enough for the reader under test.
    pub fn gguf(keys: &[(&str, Value<'_>)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&(keys.len() as u64).to_le_bytes());
        for (key, value) in keys {
            string(&mut out, key);
            match value {
                Value::Str(value) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    string(&mut out, value);
                }
                Value::StrArray(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&8u32.to_le_bytes());
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for item in *items {
                        string(&mut out, item);
                    }
                }
                Value::U32(value) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&value.to_le_bytes());
                }
                Value::U64(value) => {
                    out.extend_from_slice(&10u32.to_le_bytes());
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        out
    }

    fn string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{Value, gguf};
    use super::*;

    #[test]
    fn reads_the_keys_the_api_reports_past_skipped_arrays() {
        let bytes = gguf(&[
            ("general.architecture", Value::Str("qwen3")),
            ("tokenizer.ggml.tokens", Value::StrArray(&["<s>", "hello"])),
            ("qwen3.context_length", Value::U32(262_144)),
            ("general.file_type", Value::U32(15)),
            ("general.parameter_count", Value::U64(7_615_000_000)),
        ]);

        let metadata = read_from(&mut std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(metadata.architecture.as_deref(), Some("qwen3"));
        assert_eq!(metadata.context_length, Some(262_144));
        assert_eq!(metadata.file_type, Some(15));
        assert_eq!(metadata.parameter_count, Some(7_615_000_000));
    }

    #[test]
    fn a_context_key_that_does_not_match_the_architecture_is_ignored() {
        let bytes = gguf(&[
            ("general.architecture", Value::Str("llama")),
            ("qwen3.context_length", Value::U32(262_144)),
        ]);

        let metadata = read_from(&mut std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(metadata.context_length, None);
    }

    #[test]
    fn rejects_a_file_that_is_not_gguf() {
        assert!(read_from(&mut std::io::Cursor::new(b"model bytes".to_vec())).is_err());
    }
}
