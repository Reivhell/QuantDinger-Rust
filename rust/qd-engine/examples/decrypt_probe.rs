//! Decrypt probe for parity: given candidate secrets + stored tokens on argv,
//! print `decrypt_credential_blob` outcomes as JSON.
//!
//! Usage: `decrypt_probe <secrets-csv> <token>...`
//! Each token result: `{"ok": plaintext}` or `{"err": "InvalidToken"|...}`.
//! Empty argv token means `None` (DB NULL).

use qd_engine::credential_crypto::decrypt_credential_blob;

fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let secrets: Vec<&str> = if args.is_empty() {
        vec![]
    } else {
        args[0].split(',').collect()
    };
    let mut rows = Vec::new();
    for tok in args.iter().skip(1) {
        let stored: Option<&str> = if tok.is_empty() { None } else { Some(tok) };
        let row = match decrypt_credential_blob(stored, &secrets) {
            Ok(pt) => format!("{{\"ok\":{}}}", jstr(&pt)),
            Err(e) => format!("{{\"err\":{}}}", jstr(&format!("{e:?}"))),
        };
        rows.push(row);
    }
    println!("[{}]", rows.join(","));
}
