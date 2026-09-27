//! A self-contained SHA-256, for the job content hash.
//!
//! # Why not a dependency
//!
//! `sha2` exists in the workspace manifest but not in this crate's frozen
//! dependency list, and the ownership contract says a missing dependency is an
//! architect change, not an agent's unilateral `Cargo.toml` edit. A content
//! hash is ~70 lines of well-known arithmetic; a new third-party dependency is
//! a much larger permanent cost, so the arithmetic lives here and is verified
//! against the NIST vectors in the tests below.
//!
//! # What the content hash is for
//!
//! Transcribing a two-hour recording takes minutes, so the app must not do it
//! twice for the same file. The rule is deliberately *conservative*: the hash
//! covers the file's bytes **and** its size, and it is stored on the job. A
//! match means "this exact file has already been transcribed", which is the
//! only claim that is safe to make from a hash. It is explicitly **not** used
//! to detect tampering: an attacker who can change a file can change its hash.
//!
//! Hashing the whole file is a linear read. For a 2 GB video that is a few
//! seconds of I/O against the same file the media stage is about to read, and
//! it happens once per job; correctness of the "already transcribed?" question
//! is worth more than the seconds, so there is no size cap.

use std::io::Read;
use std::path::Path;

use vaani_core::Result;

use crate::error;

/// Round constants: the first 32 bits of the fractional parts of the cube
/// roots of the first 64 primes.
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// Streaming SHA-256, so a multi-gigabyte file never has to be resident.
#[derive(Debug, Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    /// Total bytes fed so far, as a `u64` bit-length counter.
    length_bits: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// A fresh hasher.
    pub fn new() -> Self {
        Self {
            state: H0,
            buffer: [0u8; 64],
            buffered: 0,
            length_bits: 0,
        }
    }

    /// Feed bytes. Any number of calls, in any chunk size.
    pub fn update(&mut self, mut data: &[u8]) {
        self.length_bits = self.length_bits.wrapping_add((data.len() as u64) * 8);
        if self.buffered > 0 {
            let need = 64 - self.buffered;
            let take = need.min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        // Early out: without this, an empty tail would reset `buffered` to 0 and
        // silently discard whatever was still sitting in the buffer.
        if data.is_empty() {
            return;
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for block in blocks {
            let mut b = [0u8; 64];
            b.copy_from_slice(block);
            self.compress(&b);
        }
        self.buffer[..rest.len()].copy_from_slice(rest);
        self.buffered = rest.len();
    }

    /// Finish and return the digest as 64 lowercase hex characters.
    pub fn finish_hex(mut self) -> String {
        // Padding: 0x80, then zeros, then the 64-bit big-endian bit length.
        let length_bits = self.length_bits;
        self.update_raw(&[0x80]);
        while self.buffered != 56 {
            self.update_raw(&[0x00]);
        }
        let len_bytes = length_bits.to_be_bytes();
        self.update_raw(&len_bytes);
        debug_assert_eq!(self.buffered, 0);
        let mut out = String::with_capacity(64);
        for word in self.state {
            out.push_str(&format!("{word:08x}"));
        }
        out
    }

    /// Padding bytes must not count towards the length.
    fn update_raw(&mut self, data: &[u8]) {
        for &byte in data {
            self.buffer[self.buffered] = byte;
            self.buffered += 1;
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
}

/// SHA-256 of a byte slice, as 64 lowercase hex characters.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.finish_hex()
}

/// SHA-256 of a file's contents, read in 1 MiB chunks.
///
/// A file that cannot be opened or read is a [`vaani_core::ErrorCode::StorageFailed`]
/// error naming the file the user chose, which is a value they already see.
pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| error::from_io(&format!("opening {}", path.display()), e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buf)
            .map_err(|e| error::from_io(&format!("reading {}", path.display()), e))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hasher.finish_hex())
}

/// Hash text, for tests and for keys that are not files.
pub fn hash_str(text: &str) -> String {
    sha256_hex(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nist_vectors() {
        // FIPS 180-4 / NIST CAVP SHA-256 short messages.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn streaming_in_arbitrary_chunks_matches_one_shot() {
        let data: Vec<u8> = (0..3000u32).map(|i| (i % 251) as u8).collect();
        let one_shot = sha256_hex(&data);
        for chunk in [1usize, 7, 63, 64, 65, 128, 1000] {
            let mut h = Sha256::new();
            for part in data.chunks(chunk) {
                h.update(part);
            }
            assert_eq!(h.finish_hex(), one_shot, "chunk size {chunk}");
        }
    }

    #[test]
    fn update_after_finish_is_impossible_but_buffer_is_reused_safely() {
        // Exercises the "buffer partially full, then more data" path, where a
        // naive implementation resets the buffer count and loses what was in it.
        let mut h = Sha256::new();
        h.update(b"a");
        h.update(b"bc");
        h.update(b"defghijklmnopqrstuvwxyz");
        let mut expected = Vec::from(&b"abcdefghijklmnopqrstuvwxyz"[..]);
        expected.extend_from_slice(&[0u8; 100]);
        h.update(&[0u8; 100]);
        assert_eq!(h.finish_hex(), sha256_hex(&expected));
    }

    #[test]
    fn hash_file_matches_hash_of_contents() {
        let dir = crate::testing::TempDir::new("hash");
        let path = dir.path().join("payload.bin");
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 256) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        assert_eq!(hash_file(&path).unwrap(), sha256_hex(&data));
    }

    #[test]
    fn hash_file_of_a_missing_file_is_an_error_not_a_panic() {
        let dir = crate::testing::TempDir::new("hash-missing");
        let e = hash_file(&dir.path().join("nope.wav")).unwrap_err();
        assert_eq!(e.code(), vaani_core::ErrorCode::StorageFailed);
    }
}
