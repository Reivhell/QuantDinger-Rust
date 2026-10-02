//! Port of `backend_api_python/app/utils/local_brokers.py`.
//!
//! Deployment policy for IBKR local-desktop access. The environment is read
//! through [`Env`] so the logic stays pure and testable; [`Env::live`]
//! reads the real process environment, exactly like Python's `os.getenv`.
//!
//! Faithful corner: an empty-but-set variable parses to `""`, which is not
//! in the truthy set — so `ALLOW_LOCAL_DESKTOP_BROKERS=""` disables access,
//! exactly like Python (only *unset* falls back to `"true"`).

use std::collections::HashMap;

/// Rejection message. Mirrors `desktop_broker_cloud_reject_message`.
pub const CLOUD_REJECT_MESSAGE: &str = "This server has disabled IBKR local desktop broker access \
    (requires local TWS or IB Gateway). Deploy QuantDinger on your own \
    machine or private server and install IBKR TWS/Gateway.";

/// Environment snapshot. `None` value = variable unset.
#[derive(Debug, Clone, Default)]
pub struct Env {
    vars: HashMap<String, String>,
}

impl Env {
    pub fn live() -> Self {
        let mut vars = HashMap::new();
        if let Ok(v) = std::env::var("ALLOW_LOCAL_DESKTOP_BROKERS") {
            vars.insert("ALLOW_LOCAL_DESKTOP_BROKERS".to_string(), v);
        }
        Self { vars }
    }

    pub fn with(mut self, name: &str, value: &str) -> Self {
        self.vars.insert(name.to_string(), value.to_string());
        self
    }
}

/// Mirrors `local_desktop_brokers_allowed`.
pub fn local_desktop_brokers_allowed(env: &Env) -> bool {
    let raw = env.vars.get("ALLOW_LOCAL_DESKTOP_BROKERS").map(String::as_str).unwrap_or("true");
    matches!(raw.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

/// Mirrors `require_local_desktop_brokers_allowed` (`Err` carries the
/// `PermissionError` message).
pub fn require_local_desktop_brokers_allowed(env: &Env) -> Result<(), String> {
    if local_desktop_brokers_allowed(env) {
        Ok(())
    } else {
        Err(CLOUD_REJECT_MESSAGE.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_truthy_spellings_allow() {
        assert!(local_desktop_brokers_allowed(&Env::default()));
        for v in ["1", "true", "YES", " On ", "TrUe"] {
            assert!(
                local_desktop_brokers_allowed(&Env::default().with("ALLOW_LOCAL_DESKTOP_BROKERS", v)),
                "{v}"
            );
        }
        assert!(require_local_desktop_brokers_allowed(&Env::default()).is_ok());
    }

    #[test]
    fn falsy_and_empty_deny_with_exact_message() {
        for v in ["0", "false", "no", "", "  "] {
            let env = Env::default().with("ALLOW_LOCAL_DESKTOP_BROKERS", v);
            assert!(!local_desktop_brokers_allowed(&env), "{v:?}");
            assert_eq!(require_local_desktop_brokers_allowed(&env).unwrap_err(), CLOUD_REJECT_MESSAGE);
        }
    }
}
