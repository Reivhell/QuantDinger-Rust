//! Port of `backend_api_python/app/utils/credential_crypto.py`.
//!
//! Fernet encryption for persisted credentials, std-only (no `cryptography`
//! crate; crates.io is unreachable): AES-128-CBC + HMAC-SHA256 + PKCS7 +
//! base64url, with the key derived as
//! `base64url(sha256(secret))` — byte-identical to `_fernet`.
//!
//! Scope: the pure crypto. Secret *resolution* (`CREDENTIAL_ENCRYPTION_KEY`
//! vs `SECRET_KEY` vs `Config`, env reads) stays Python-side; the port takes
//! explicit secret strings. [`decrypt_credential_blob`] replicates the
//! fallback order (dedup preserving order), the empty-input shortcut, and
//! the exact error cases.
//!
//! Faithful corners:
//! - `decrypt` performs no TTL check (Python calls `decrypt` without `ttl`).
//! - Version byte must be `0x80`, HMAC verified in full before decryption,
//!   PKCS7 padding fully validated — any violation is [`DecryptError`],
//!   mirroring `InvalidToken`.
//! - `stored.encode("ascii")` happens *before* key checks in Python —
//!   non-ASCII input raises even when keys are set. [`decrypt_credential_blob`]
//!   preserves that order.
//! - `encrypt(None)` → encrypts `""` (None becomes empty string).

use crate::snapshot::sha256_bytes;

// ---------------------------------------------------------------------------
// base64url (Fernet alphabet, padding kept on encode)
// ---------------------------------------------------------------------------

const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

pub fn b64url_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_ALPHABET[((n >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_ALPHABET[(n & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn b64val(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some((c - b'A') as u32),
        b'a'..=b'z' => Some((c - b'a' as u8 + 26) as u32),
        b'0'..=b'9' => Some((c - b'0' + 52) as u32),
        b'-' => Some(62),
        b'_' => Some(63),
        // Accept standard alphabet on decode (tolerant reader).
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// URL-safe base64 decode; `=` padding optional (Fernet tokens carry none,
/// keys carry one `=`).
pub fn b64url_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut clean: Vec<u8> = s.bytes().filter(|&b| b != b'=').collect();
    // `str.strip()` in Python removes surrounding whitespace before decode.
    if clean.iter().any(|&b| b.is_ascii_whitespace()) {
        clean.retain(|b| !b.is_ascii_whitespace());
    }
    if clean.len() % 4 == 1 {
        return Err("bad base64 length".to_string());
    }
    while clean.len() % 4 != 0 {
        clean.push(b'A');
    }
    let mut out = Vec::with_capacity(clean.len() / 4 * 3);
    for q in clean.chunks(4) {
        let v = [b64val(q[0]), b64val(q[1]), b64val(q[2]), b64val(q[3])];
        if v.iter().any(|x| x.is_none()) {
            return Err("bad base64 char".to_string());
        }
        let n = (v[0].unwrap() << 18) | (v[1].unwrap() << 12) | (v[2].unwrap() << 6) | v[3].unwrap();
        out.push((n >> 16) as u8);
        out.push((n >> 8) as u8);
        out.push(n as u8);
    }
    // Trim padding-implied bytes: length mod 4 == 2 → 1 byte, == 3 → 2 bytes.
    let rem = s.bytes().filter(|&b| b != b'=').count() % 4;
    match rem {
        2 => out.truncate(out.len() - 2),
        3 => out.truncate(out.len() - 1),
        _ => {}
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// AES-128 (FIPS-197)
// ---------------------------------------------------------------------------

const SBOX: [u8; 256] = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
];

const INV_SBOX: [u8; 256] = [
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d,
];

const RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

fn xtime(x: u8) -> u8 {
    if x & 0x80 == 0 { x << 1 } else { (x << 1) ^ 0x1b }
}

fn mix_column(c: &mut [u8; 4]) {
    let (a0, a1, a2, a3) = (c[0], c[1], c[2], c[3]);
    c[0] = xtime(a0) ^ (xtime(a1) ^ a1) ^ a2 ^ a3;
    c[1] = a0 ^ xtime(a1) ^ (xtime(a2) ^ a2) ^ a3;
    c[2] = a0 ^ a1 ^ xtime(a2) ^ (xtime(a3) ^ a3);
    c[3] = (xtime(a0) ^ a0) ^ a1 ^ a2 ^ xtime(a3);
}

fn mul(a: u8, b: u8) -> u8 {
    let mut p = 0u8;
    let (mut a, mut b) = (a, b);
    for _ in 0..8 {
        if b & 1 == 1 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    p
}

fn inv_mix_column(c: &mut [u8; 4]) {
    let (a0, a1, a2, a3) = (c[0], c[1], c[2], c[3]);
    c[0] = mul(a0, 0x0e) ^ mul(a1, 0x0b) ^ mul(a2, 0x0d) ^ mul(a3, 0x09);
    c[1] = mul(a0, 0x09) ^ mul(a1, 0x0e) ^ mul(a2, 0x0b) ^ mul(a3, 0x0d);
    c[2] = mul(a0, 0x0d) ^ mul(a1, 0x09) ^ mul(a2, 0x0e) ^ mul(a3, 0x0b);
    c[3] = mul(a0, 0x0b) ^ mul(a1, 0x0d) ^ mul(a2, 0x09) ^ mul(a3, 0x0e);
}

fn expand_key(key: &[u8; 16]) -> [[u8; 16]; 11] {
    let mut rk = [[0u8; 16]; 11];
    rk[0] = *key;
    for i in 1..11 {
        let mut t = [rk[i - 1][13], rk[i - 1][14], rk[i - 1][15], rk[i - 1][12]];
        for b in t.iter_mut() {
            *b = SBOX[*b as usize];
        }
        t[0] ^= RCON[i - 1];
        for j in 0..4 {
            rk[i][j] = rk[i - 1][j] ^ t[j];
        }
        for j in 4..16 {
            rk[i][j] = rk[i - 1][j] ^ rk[i][j - 4];
        }
    }
    rk
}

fn add_round_key(s: &mut [u8; 16], rk: &[u8; 16]) {
    for i in 0..16 {
        s[i] ^= rk[i];
    }
}

fn shift_rows(s: &mut [u8; 16]) {
    // state is column-major: s[row + 4*col]
    let t = *s;
    s[1] = t[5];
    s[5] = t[9];
    s[9] = t[13];
    s[13] = t[1];
    s[2] = t[10];
    s[6] = t[14];
    s[10] = t[2];
    s[14] = t[6];
    s[3] = t[15];
    s[7] = t[3];
    s[11] = t[7];
    s[15] = t[11];
}

fn inv_shift_rows(s: &mut [u8; 16]) {
    let t = *s;
    s[1] = t[13];
    s[5] = t[1];
    s[9] = t[5];
    s[13] = t[9];
    s[2] = t[10];
    s[6] = t[14];
    s[10] = t[2];
    s[14] = t[6];
    s[3] = t[7];
    s[7] = t[11];
    s[11] = t[15];
    s[15] = t[3];
}

fn aes128_encrypt_block(key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
    let rk = expand_key(key);
    let mut s = *block;
    add_round_key(&mut s, &rk[0]);
    for r in 1..10 {
        for b in s.iter_mut() {
            *b = SBOX[*b as usize];
        }
        shift_rows(&mut s);
        for c in 0..4 {
            let mut col = [s[c * 4], s[c * 4 + 1], s[c * 4 + 2], s[c * 4 + 3]];
            mix_column(&mut col);
            s[c * 4..c * 4 + 4].copy_from_slice(&col);
        }
        add_round_key(&mut s, &rk[r]);
    }
    for b in s.iter_mut() {
        *b = SBOX[*b as usize];
    }
    shift_rows(&mut s);
    add_round_key(&mut s, &rk[10]);
    s
}

fn aes128_decrypt_block(key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
    let rk = expand_key(key);
    let mut s = *block;
    add_round_key(&mut s, &rk[10]);
    for r in (1..10).rev() {
        inv_shift_rows(&mut s);
        for b in s.iter_mut() {
            *b = INV_SBOX[*b as usize];
        }
        add_round_key(&mut s, &rk[r]);
        for c in 0..4 {
            let mut col = [s[c * 4], s[c * 4 + 1], s[c * 4 + 2], s[c * 4 + 3]];
            inv_mix_column(&mut col);
            s[c * 4..c * 4 + 4].copy_from_slice(&col);
        }
    }
    inv_shift_rows(&mut s);
    for b in s.iter_mut() {
        *b = INV_SBOX[*b as usize];
    }
    add_round_key(&mut s, &rk[0]);
    s
}

fn aes128_cbc_encrypt(key: &[u8; 16], iv: &[u8; 16], plain: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(plain.len());
    let mut prev = *iv;
    for chunk in plain.chunks(16) {
        let mut block = [0u8; 16];
        block[..chunk.len()].copy_from_slice(chunk);
        for i in 0..16 {
            block[i] ^= prev[i];
        }
        let enc = aes128_encrypt_block(key, &block);
        out.extend_from_slice(&enc);
        prev = enc;
    }
    out
}

fn aes128_cbc_decrypt(key: &[u8; 16], iv: &[u8; 16], cipher: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(cipher.len());
    let mut prev = *iv;
    for chunk in cipher.chunks(16) {
        let mut block = [0u8; 16];
        block.copy_from_slice(chunk);
        let dec = aes128_decrypt_block(key, &block);
        for i in 0..16 {
            out.push(dec[i] ^ prev[i]);
        }
        prev = block;
    }
    out
}

// ---------------------------------------------------------------------------
// HMAC-SHA256
// ---------------------------------------------------------------------------

pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut kb = [0u8; 64];
    if key.len() > 64 {
        kb[..32].copy_from_slice(&sha256_bytes(key));
    } else {
        kb[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= kb[i];
        opad[i] ^= kb[i];
    }
    let mut inner = Vec::with_capacity(64 + msg.len());
    inner.extend_from_slice(&ipad);
    inner.extend_from_slice(msg);
    let ih = sha256_bytes(&inner);
    let mut outer = Vec::with_capacity(64 + 32);
    outer.extend_from_slice(&opad);
    outer.extend_from_slice(&ih);
    sha256_bytes(&outer)
}

// ---------------------------------------------------------------------------
// Fernet tokens
// ---------------------------------------------------------------------------

/// Errors mirroring `cryptography.fernet.InvalidToken` plus input-shape
/// errors that surface differently in Python (`UnicodeEncodeError`,
/// `binascii.Error`, `ValueError` from missing keys).
#[derive(Debug, Clone, PartialEq)]
pub enum DecryptError {
    InvalidToken,
    NonAsciiInput,
    NoKeys,
    CannotDecrypt,
}

/// Derive the 32 Fernet key bytes from a secret, like `_fernet`.
pub fn fernet_key_bytes(secret: &str) -> [u8; 32] {
    sha256_bytes(secret.as_bytes())
}

/// Read 16 random bytes from the OS (Linux `/dev/urandom`), falling back to
/// a time/pid hash when unavailable.
pub fn random_iv() -> [u8; 16] {
    use std::io::Read;
    // NOTE: never `fs::read` /dev/urandom — it never EOFs. Bounded read only.
    if let Ok(f) = std::fs::File::open("/dev/urandom") {
        let mut take = f.take(16);
        let mut iv = [0u8; 16];
        if take.read_exact(&mut iv).is_ok() {
            return iv;
        }
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut seed = now.wrapping_add(std::process::id() as u128 * 0x9E3779B97F4A7C15);
    let mut iv = [0u8; 16];
    for b in iv.iter_mut() {
        // splitmix64 step
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        *b = (z >> 56) as u8;
    }
    iv
}

fn pkcs7_pad(data: &[u8]) -> Vec<u8> {
    let pad = 16 - (data.len() % 16);
    let mut out = Vec::with_capacity(data.len() + pad);
    out.extend_from_slice(data);
    out.extend(std::iter::repeat(pad as u8).take(pad));
    out
}

/// Build a Fernet token: `v80 | timestamp BE | iv | CBC | HMAC`, base64url.
pub fn fernet_encrypt_with(secret: &str, plaintext: &[u8], timestamp: u64, iv: [u8; 16]) -> String {
    let key = fernet_key_bytes(secret);
    let (signing, enc): ([u8; 16], [u8; 16]) = {
        let mut s = [0u8; 16];
        let mut e = [0u8; 16];
        s.copy_from_slice(&key[..16]);
        e.copy_from_slice(&key[16..]);
        (s, e)
    };
    let cipher = aes128_cbc_encrypt(&enc, &iv, &pkcs7_pad(plaintext));
    let mut data = Vec::with_capacity(1 + 8 + 16 + cipher.len() + 32);
    data.push(0x80);
    data.extend_from_slice(&timestamp.to_be_bytes());
    data.extend_from_slice(&iv);
    data.extend_from_slice(&cipher);
    let sig = hmac_sha256(&signing, &data);
    data.extend_from_slice(&sig);
    b64url_encode(&data)
}

pub fn fernet_encrypt(secret: &str, plaintext: &[u8]) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    fernet_encrypt_with(secret, plaintext, ts, random_iv())
}

/// Decrypt a Fernet token (no TTL check, like Python's plain `decrypt`).
pub fn fernet_decrypt(secret: &str, token: &str) -> Result<Vec<u8>, DecryptError> {
    if !token.is_ascii() {
        return Err(DecryptError::NonAsciiInput);
    }
    let data = b64url_decode(token.trim()).map_err(|_| DecryptError::InvalidToken)?;
    if data.len() < 1 + 8 + 16 + 16 + 32 || data[0] != 0x80 {
        return Err(DecryptError::InvalidToken);
    }
    let key = fernet_key_bytes(secret);
    let (signing, enc): ([u8; 16], [u8; 16]) = {
        let mut s = [0u8; 16];
        let mut e = [0u8; 16];
        s.copy_from_slice(&key[..16]);
        e.copy_from_slice(&key[16..]);
        (s, e)
    };
    let (body, sig) = data.split_at(data.len() - 32);
    let expect = hmac_sha256(&signing, body);
    // Constant-time compare.
    let mut diff = 0u8;
    for (a, b) in sig.iter().zip(expect.iter()) {
        diff |= a ^ b;
    }
    if diff != 0 {
        return Err(DecryptError::InvalidToken);
    }
    let iv: [u8; 16] = body[9..25].try_into().map_err(|_| DecryptError::InvalidToken)?;
    let plain = aes128_cbc_decrypt(&enc, &iv, &body[25..]);
    // PKCS7 unpad, fully validated.
    let &pad = plain.last().ok_or(DecryptError::InvalidToken)?;
    if pad == 0 || pad > 16 || plain.len() < pad as usize {
        return Err(DecryptError::InvalidToken);
    }
    if plain[plain.len() - pad as usize..].iter().any(|&b| b != pad) {
        return Err(DecryptError::InvalidToken);
    }
    Ok(plain[..plain.len() - pad as usize].to_vec())
}

/// Mirrors `encrypt_credential_blob`: `None` → `""`, then encrypt.
pub fn encrypt_credential_blob(secret: &str, plaintext: Option<&str>) -> String {
    fernet_encrypt(secret, plaintext.unwrap_or("").as_bytes())
}

/// Mirrors `decrypt_credential_blob` given the already-resolved candidate
/// secrets in order (`[_credential_key(), _secret_key()]`, deduped).
/// `stored` is the raw DB value (`None` → `Ok("")`).
pub fn decrypt_credential_blob(stored: Option<&str>, secrets: &[&str]) -> Result<String, DecryptError> {
    let s = match stored {
        None => return Ok(String::new()),
        Some(v) => v,
    };
    if !s.is_ascii() {
        return Err(DecryptError::NonAsciiInput);
    }
    let t = s.trim();
    if t.is_empty() {
        return Ok(String::new());
    }
    let mut uniq: Vec<&str> = Vec::new();
    for sec in secrets {
        // Python strips env values before the truthiness check.
        let tsec = sec.trim();
        if !tsec.is_empty() && !uniq.contains(&tsec) {
            uniq.push(tsec);
        }
    }
    if uniq.is_empty() {
        return Err(DecryptError::NoKeys);
    }
    for sec in uniq {
        match fernet_decrypt(sec, t) {
            Ok(bytes) => return String::from_utf8(bytes).map_err(|_| DecryptError::InvalidToken),
            Err(DecryptError::InvalidToken) => continue,
            Err(e) => return Err(e),
        }
    }
    Err(DecryptError::CannotDecrypt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aes128_matches_fips_vector() {
        // FIPS-197 Appendix B.
        let key: [u8; 16] = [0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c];
        let pt: [u8; 16] = [0x32,0x43,0xf6,0xa8,0x88,0x5a,0x30,0x8d,0x31,0x31,0x98,0xa2,0xe0,0x37,0x07,0x34];
        let ct = aes128_encrypt_block(&key, &pt);
        assert_eq!(ct, [0x39,0x25,0x84,0x1d,0x02,0xdc,0x09,0xfb,0xdc,0x11,0x85,0x97,0x19,0x6a,0x0b,0x32]);
        assert_eq!(aes128_decrypt_block(&key, &ct), pt);
    }

    #[test]
    fn hmac_matches_rfc4231() {
        // RFC 4231 test case 1.
        let mac = hmac_sha256(&[0x0b; 20], b"Hi There");
        let expect = hex("b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
        assert_eq!(mac.as_slice(), expect.as_slice());
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    #[test]
    fn b64url_round_trip_with_padding_shapes() {
        for len in [0usize, 1, 2, 3, 15, 16, 17, 32, 57] {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(b64url_decode(&b64url_encode(&data)).unwrap(), data);
        }
        assert_eq!(b64url_decode("Zg==").unwrap(), b"f");
        assert!(b64url_decode("***").is_err());
    }

    #[test]
    fn fernet_round_trip_and_tamper() {
        let tok = fernet_encrypt_with("s3cret", b"{\"a\":1}", 1_767_225_600, [7u8; 16]);
        assert_eq!(fernet_decrypt("s3cret", &tok).unwrap(), b"{\"a\":1}");
        assert_eq!(fernet_decrypt("wrong", &tok), Err(DecryptError::InvalidToken));
        // Flip a ciphertext char (keep alphabet valid).
        let mut ch: Vec<char> = tok.chars().collect();
        ch[20] = if ch[20] == 'A' { 'B' } else { 'A' };
        assert_eq!(
            fernet_decrypt("s3cret", &ch.into_iter().collect::<String>()).unwrap_err(),
            DecryptError::InvalidToken
        );
        // Bad version byte.
        let mut raw = b64url_decode(&tok).unwrap();
        raw[0] = 0x81;
        assert_eq!(fernet_decrypt("s3cret", &b64url_encode(&raw)), Err(DecryptError::InvalidToken));
        assert_eq!(fernet_decrypt("s3cret", "caf\u{00e9}"), Err(DecryptError::NonAsciiInput));
    }

    #[test]
    fn blob_semantics() {
        assert_eq!(decrypt_credential_blob(None, &["a"]), Ok(String::new()));
        assert_eq!(decrypt_credential_blob(Some("  "), &["a"]), Ok(String::new()));
        assert_eq!(decrypt_credential_blob(Some("x"), &[]), Err(DecryptError::NoKeys));
        assert_eq!(decrypt_credential_blob(Some("x"), &["", "  "]), Err(DecryptError::NoKeys));
        let tok = encrypt_credential_blob("k", None);
        assert_eq!(decrypt_credential_blob(Some(&tok), &["k"]), Ok(String::new()));
    }

    #[test]
    fn fallback_order_dedupes() {
        let tok = encrypt_credential_blob("new", Some("secret-data"));
        // new key second still wins; dupes collapsed.
        assert_eq!(decrypt_credential_blob(Some(&tok), &["old", "old", "new"]), Ok("secret-data".to_string()));
        assert_eq!(decrypt_credential_blob(Some(&tok), &["old"]), Err(DecryptError::CannotDecrypt));
    }
}
