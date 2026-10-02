//! Port of `backend_api_python/app/utils/language.py`.
//!
//! AI-output language follows the frontend UI language. `_normalize_lang`
//! is pure string logic; `detect_request_language` reads the Flask request,
//! which arrives here as plain header/arg/body maps (no Flask needed).

/// Canonical supported tags. Mirrors `SUPPORTED_LANGS`.
pub const SUPPORTED_LANGS: &[&str] = &[
    "en-US", "zh-CN", "zh-TW", "ja-JP", "ko-KR", "vi-VN", "th-TH", "ar-SA",
    "fr-FR", "de-DE", "ru-RU",
];

/// Mirrors `_normalize_lang`: first `Accept-Language` entry, strip `;q=`,
/// short-tag folding, then canonical-case match.
pub fn normalize_lang(raw: &str) -> Option<String> {
    let mut s = raw.trim().to_string();
    if s.is_empty() {
        return None;
    }
    if let Some((first, _)) = s.split_once(',') {
        s = first.trim().to_string();
    }
    if let Some((first, _)) = s.split_once(';') {
        s = first.trim().to_string();
    }
    let lower = s.to_lowercase();
    match lower.as_str() {
        "en" | "en-us" => return Some("en-US".to_string()),
        "zh" | "zh-cn" | "zh-hans" => return Some("zh-CN".to_string()),
        "zh-tw" | "zh-hant" => return Some("zh-TW".to_string()),
        _ => {}
    }
    for lang in SUPPORTED_LANGS {
        if lang.to_lowercase() == lower {
            return Some(lang.to_string());
        }
    }
    None
}

/// Request language sources. Mirrors the Flask request surface
/// (`headers.get("X-App-Lang")`, `body["language"]`, `args["language"]`,
/// `headers.get("Accept-Language")`).
#[derive(Debug, Clone, Default)]
pub struct RequestParts<'a> {
    pub header_app_lang: Option<&'a str>,
    pub body_language: Option<&'a str>,
    pub query_language: Option<&'a str>,
    pub header_accept_language: Option<&'a str>,
}

/// Mirrors `detect_request_language`: header → body/query → Accept-Language
/// → default.
pub fn detect_request_language(req: &RequestParts<'_>, default: &str) -> String {
    if let Some(raw) = req.header_app_lang {
        if let Some(lang) = normalize_lang(raw) {
            return lang;
        }
    }
    if let Some(raw) = req.body_language {
        if let Some(lang) = normalize_lang(raw) {
            return lang;
        }
    }
    if let Some(raw) = req.query_language {
        if let Some(lang) = normalize_lang(raw) {
            return lang;
        }
    }
    if let Some(raw) = req.header_accept_language {
        if let Some(lang) = normalize_lang(raw) {
            return lang;
        }
    }
    default.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_folds_short_and_quality_tags() {
        assert_eq!(normalize_lang("en"), Some("en-US".into()));
        assert_eq!(normalize_lang("en-US,en;q=0.9"), Some("en-US".into()));
        assert_eq!(normalize_lang(" zh-hans "), Some("zh-CN".into()));
        assert_eq!(normalize_lang("zh-Hant"), Some("zh-TW".into()));
        assert_eq!(normalize_lang("JA-jp"), Some("ja-JP".into()));
        assert_eq!(normalize_lang("fr-FR; q=0.8"), Some("fr-FR".into()));
        assert_eq!(normalize_lang(""), None);
        assert_eq!(normalize_lang("   "), None);
        assert_eq!(normalize_lang("xx-YY"), None);
        assert_eq!(normalize_lang("en;"), Some("en-US".into()));
    }

    #[test]
    fn detect_follows_priority() {
        let base = RequestParts {
            header_app_lang: Some("ja-JP"),
            body_language: Some("fr-FR"),
            query_language: Some("de-DE"),
            header_accept_language: Some("ko-KR"),
        };
        assert_eq!(detect_request_language(&base, "en-US"), "ja-JP");
        let no_header = RequestParts { header_app_lang: Some("xx"), ..base.clone() };
        assert_eq!(detect_request_language(&no_header, "en-US"), "fr-FR");
        let query_only = RequestParts {
            query_language: Some("th-TH"),
            ..Default::default()
        };
        assert_eq!(detect_request_language(&query_only, "en-US"), "th-TH");
        let empty: RequestParts = Default::default();
        assert_eq!(detect_request_language(&empty, "zh-CN"), "zh-CN");
    }
}
