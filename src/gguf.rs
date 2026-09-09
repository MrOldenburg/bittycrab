// Minimal GGUF v3 reader, ported from gguf.mojo. Same format assumptions:
// little-endian, tensor data aligned to `general.alignment` after the
// tensor-info block. I2_S tensors: [n/4 packed bytes][f32 scale][pad].

use memmap2::{Advice, Mmap};
use std::fs::File;

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_I2_S: u32 = 36;

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let f: f32 = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            // subnormal
            (mant as f32) * 2f32.powi(-24)
        }
    } else if exp == 0x1f {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 {
        -f
    } else {
        f
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.buf[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.buf[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        v
    }
    fn string(&mut self) -> String {
        let n = self.u64() as usize;
        let s = String::from_utf8_lossy(&self.buf[self.p..self.p + n]).into_owned();
        self.p += n;
        s
    }
    fn string_array(&mut self) -> Vec<String> {
        let it = self.u32();
        assert_eq!(it, 8, "expected string array");
        let cnt = self.u64() as usize;
        (0..cnt).map(|_| self.string()).collect()
    }
    fn scalar_size(vt: u32) -> usize {
        match vt {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4 | 5 | 6 => 4,
            10 | 11 | 12 => 8,
            _ => 0,
        }
    }
    fn skip_value(&mut self, vt: u32) {
        if vt == 8 {
            let n = self.u64() as usize;
            self.p += n;
        } else if vt == 9 {
            let it = self.u32();
            let cnt = self.u64() as usize;
            for _ in 0..cnt {
                self.skip_value(it);
            }
        } else {
            self.p += Self::scalar_size(vt);
        }
    }
}

pub struct Gguf {
    pub mmap: Mmap,
    pub data_base: usize,
    pub names: Vec<String>,
    pub types: Vec<u32>,
    pub offsets: Vec<u64>,
    pub shapes: Vec<Vec<u64>>,
    pub tokens: Vec<String>,
    pub merges: Vec<String>,
}

impl Gguf {
    pub fn index_of(&self, name: &str) -> usize {
        self.names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("tensor not found: {}", name))
    }

    pub fn n_elements(&self, i: usize) -> usize {
        self.shapes[i].iter().map(|d| *d as usize).product()
    }

    fn abs_off(&self, i: usize) -> usize {
        self.data_base + self.offsets[i] as usize
    }

    pub fn load_i2s(&self, name: &str) -> (&[u8], f32) {
        let i = self.index_of(name);
        assert_eq!(self.types[i], GGML_TYPE_I2_S, "{} is not I2_S", name);
        let n_elems = self.n_elements(i);
        let packed_len = n_elems / 4;
        let off = self.abs_off(i);
        let packed = &self.mmap[off..off + packed_len];
        let scale_bytes = &self.mmap[off + packed_len..off + packed_len + 4];
        let scale = f32::from_le_bytes(scale_bytes.try_into().unwrap());
        (packed, scale)
    }

    /// Byte range (offset, len) of an I2_S tensor's packed data within the
    /// mmap, plus its scale -- for callers that want to keep a lightweight,
    /// Copy-able reference into the mmap instead of an owned copy of the
    /// weight bytes (see model.rs's LayerWeights: it used to `.to_vec()`
    /// every I2_S tensor, permanently doubling ~432MB of resident memory
    /// between the mmap's page cache and the heap copy).
    pub fn i2s_range(&self, name: &str) -> (usize, usize, f32) {
        let i = self.index_of(name);
        assert_eq!(self.types[i], GGML_TYPE_I2_S, "{} is not I2_S", name);
        let n_elems = self.n_elements(i);
        let packed_len = n_elems / 4;
        let off = self.abs_off(i);
        let scale_bytes = &self.mmap[off + packed_len..off + packed_len + 4];
        let scale = f32::from_le_bytes(scale_bytes.try_into().unwrap());
        (off, packed_len, scale)
    }

    /// Resolve a (offset, len) byte range from `i2s_range` back into a slice.
    pub fn slice_at(&self, off: usize, len: usize) -> &[u8] {
        &self.mmap[off..off + len]
    }

    pub fn load_f32(&self, name: &str) -> Vec<f32> {
        let i = self.index_of(name);
        assert_eq!(self.types[i], GGML_TYPE_F32, "{} is not F32", name);
        let n_elems = self.n_elements(i);
        let off = self.abs_off(i);
        let raw = &self.mmap[off..off + n_elems * 4];
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    /// Raw F16 tensor bytes (widen lazily at use site -- for the tied
    /// embedding table we don't want to eagerly widen 656MB to F32).
    pub fn f16_tensor_bytes(&self, name: &str) -> &[u8] {
        let i = self.index_of(name);
        assert_eq!(self.types[i], GGML_TYPE_F16, "{} is not F16", name);
        let n_elems = self.n_elements(i);
        let off = self.abs_off(i);
        &self.mmap[off..off + n_elems * 2]
    }
}

pub fn open_gguf(path: &str) -> Gguf {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    // Every token touches essentially all of this file (full-vocab lm_head
    // scan + every layer's weights), so prefetch it into the page cache up
    // front instead of faulting it in row-by-row across the first several
    // decode steps; HugePage cuts TLB pressure on the multi-hundred-MB
    // tensors if the kernel honors it for a file-backed mapping (silently a
    // no-op otherwise -- both are advisory, errors are not fatal).
    let _ = mmap.advise(Advice::WillNeed);
    let _ = mmap.advise(Advice::HugePage);

    let mut c = Cursor { buf: &mmap[..], p: 0 };
    let magic = c.u32();
    assert_eq!(magic, 0x46554747, "not a GGUF file (bad magic): {}", path);
    let version = c.u32();
    assert_eq!(version, 3, "unsupported GGUF version {} (want 3)", version);
    let tensor_count = c.u64() as usize;
    let kv_count = c.u64() as usize;

    let mut alignment: usize = 32;
    let mut tokens = Vec::new();
    let mut merges = Vec::new();

    for _ in 0..kv_count {
        let key = c.string();
        let vt = c.u32();
        match (key.as_str(), vt) {
            ("general.alignment", 4) => alignment = c.u32() as usize,
            ("tokenizer.ggml.tokens", 9) => tokens = c.string_array(),
            ("tokenizer.ggml.merges", 9) => merges = c.string_array(),
            _ => c.skip_value(vt),
        }
    }

    let mut names = Vec::with_capacity(tensor_count);
    let mut types = Vec::with_capacity(tensor_count);
    let mut offsets = Vec::with_capacity(tensor_count);
    let mut shapes = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        names.push(c.string());
        let nd = c.u32() as usize;
        let dims: Vec<u64> = (0..nd).map(|_| c.u64()).collect();
        shapes.push(dims);
        types.push(c.u32());
        offsets.push(c.u64());
    }

    let pad = c.p % alignment;
    let data_base = if pad == 0 { c.p } else { c.p + (alignment - pad) };

    Gguf {
        mmap,
        data_base,
        names,
        types,
        offsets,
        shapes,
        tokens,
        merges,
    }
}
