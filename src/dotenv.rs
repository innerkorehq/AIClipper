//! Minimal `.env` support: `KEY=VALUE` lines become environment variables.
//!
//! The pipeline and its Python sidecar both read their settings from the environment
//! (`AICLIPPER_*`, `HF_TOKEN`), so the file is loaded into the process environment once,
//! at startup, before anything else runs. Rules:
//!
//! * a variable that is already set in the real environment is never overridden;
//! * `.env` in the current directory is read first, then the one in the project root
//!   (the directory the binary was built from), so the same settings apply wherever the
//!   command is run from;
//! * lines are `KEY=VALUE` or `export KEY=VALUE`; `#` starts a comment; a value may be
//!   quoted with `"` or `'` (quotes are removed, and a quoted value may contain `#`).

use std::path::{Path, PathBuf};

/// Parse the text of a `.env` file into `(key, value)` pairs, in file order.
pub fn parse(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").map(str::trim_start).unwrap_or(line);
        let Some((key, rest)) = line.split_once('=') else { continue };
        let key = key.trim();
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        let rest = rest.trim();
        let value = match rest.chars().next() {
            Some(q @ ('"' | '\'')) => match rest[1..].find(q) {
                Some(end) => rest[1..1 + end].to_string(),
                None => rest[1..].to_string(), // unterminated: take the rest
            },
            // unquoted: an inline comment starts at " #"
            _ => rest.split(" #").next().unwrap_or("").trim().to_string(),
        };
        out.push((key.to_string(), value));
    }
    out
}

/// Where `.env` files are looked for, in priority order.
fn candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        v.push(cwd.join(".env"));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".env");
    if !v.contains(&root) {
        v.push(root);
    }
    v
}

/// Load `.env` files into the process environment. Returns the files read and how many
/// variables each one set. Call before any other thread is started.
pub fn load() -> Vec<(PathBuf, usize)> {
    let mut loaded = Vec::new();
    for path in candidates() {
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let mut set = 0;
        for (k, v) in parse(&text) {
            if std::env::var_os(&k).is_none() {
                std::env::set_var(&k, &v);
                set += 1;
            }
        }
        loaded.push((path, set));
    }
    loaded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_export_quoted_and_commented_lines() {
        let t = "# a comment\n\nAICLIPPER_WHISPER_BACKEND=hf\nexport A_B=1 # inline\nQ=\"two words # kept\"\nS='single'\n  SPACED = value  \nbad line\n=nokey\n1-BAD=x\nEMPTY=\n";
        let got = parse(t);
        let get = |k: &str| got.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        assert_eq!(get("AICLIPPER_WHISPER_BACKEND"), Some("hf"));
        assert_eq!(get("A_B"), Some("1"));
        assert_eq!(get("Q"), Some("two words # kept"));
        assert_eq!(get("S"), Some("single"));
        assert_eq!(get("SPACED"), Some("value"));
        assert_eq!(get("EMPTY"), Some(""));
        assert_eq!(got.len(), 6, "comments, blanks and malformed lines are skipped: {got:?}");
    }
}
