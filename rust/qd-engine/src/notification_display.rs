//! Port of `backend_api_python/app/utils/notification_display.py`.
//!
//! `with_display` attaches locale-neutral vue-i18n display metadata to a
//! notification payload. Payloads are [`crate::json_helpers::JsonVal`]
//! objects; legacy rows without `display` fall back to title/message, which
//! stays reader-side (nothing to port).

use crate::json_helpers::JsonVal;

/// Mirrors `with_display(payload, template, params)`: copies the payload and
/// sets `display = {"template": ..., "params": {...}}`.
pub fn with_display(
    payload: &[(String, JsonVal)],
    template: &str,
    params: &[(String, JsonVal)],
) -> Vec<(String, JsonVal)> {
    let mut out: Vec<(String, JsonVal)> = payload.to_vec();
    out.retain(|(k, _)| k != "display");
    out.push((
        "display".to_string(),
        JsonVal::Obj(vec![
            ("template".to_string(), JsonVal::Str(template.to_string())),
            ("params".to_string(), JsonVal::Obj(params.to_vec())),
        ]),
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attaches_display_and_overwrites_stale() {
        let payload = vec![
            ("title".to_string(), JsonVal::Str("T".into())),
            ("display".to_string(), JsonVal::Str("stale".into())),
        ];
        let params = vec![("symbol".to_string(), JsonVal::Str("BTC".into()))];
        let out = with_display(&payload, "fill", &params);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], payload[0]);
        assert_eq!(
            out[1].1,
            JsonVal::Obj(vec![
                ("template".to_string(), JsonVal::Str("fill".into())),
                (
                    "params".to_string(),
                    JsonVal::Obj(vec![("symbol".to_string(), JsonVal::Str("BTC".into()))])
                ),
            ])
        );
    }

    #[test]
    fn none_payload_starts_empty() {
        let out = with_display(&[], "t", &[]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "display");
    }
}
