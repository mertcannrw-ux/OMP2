//! Durable provider registry.
//!
//! A session can reach more than one gateway: each record here is a named way to
//! reach a model catalog (adapter, endpoint, the *name* of the environment
//! variable holding its key, and a default model). The active one is still
//! selected through the `ai_*` convars, so everything that reads those — request
//! derivation, compaction budgets, replication — keeps working unchanged; the
//! registry is what lets a session switch without retyping an endpoint and what
//! lets the model list say which provider a model came from.

use crate::ids::ElementId;
use crate::patch::{ElementSnapshot, TypedValue};
use serde::{Deserialize, Serialize};

/// Container element that holds the registry.
pub const PROVIDERS_CONTAINER: &str = "providers";
/// Element kind of one registry entry.
pub const PROVIDER_KIND: &str = "provider";
/// Attribute carrying the catalog fetched for this entry, bounded by the caller.
pub const PROVIDER_MODELS_ATTRIBUTE: &str = "models";

/// One named way to reach a provider.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRecord {
    pub name: String,
    pub adapter: String,
    pub endpoint: String,
    /// Name of the host environment variable holding the credential. Never the
    /// credential itself: records are journaled and replicate with the session.
    pub key_env: String,
    /// Model this provider should select when it becomes active; empty means the
    /// first advertised model.
    pub model: String,
}

impl ProviderRecord {
    /// Host part of the endpoint, used as a label when a provider is unregistered.
    pub fn host(&self) -> String {
        endpoint_host(&self.endpoint)
    }

    pub fn to_element(&self, id: ElementId) -> ElementSnapshot {
        let mut element = ElementSnapshot::new(id, PROVIDER_KIND);
        for (attribute, value) in [
            ("name", self.name.as_str()),
            ("adapter", self.adapter.as_str()),
            ("endpoint", self.endpoint.as_str()),
            ("key_env", self.key_env.as_str()),
            ("model", self.model.as_str()),
        ] {
            element
                .attributes
                .insert(attribute.into(), TypedValue::String(value.to_string()));
        }
        element
    }

    pub fn from_element(element: &ElementSnapshot) -> Option<Self> {
        if element.kind != PROVIDER_KIND {
            return None;
        }
        let string = |name: &str| match element.attributes.get(name) {
            Some(TypedValue::String(value)) => value.clone(),
            _ => String::new(),
        };
        let name = string("name");
        let endpoint = string("endpoint");
        if name.is_empty() || endpoint.is_empty() {
            return None;
        }
        Some(Self {
            name,
            adapter: string("adapter"),
            endpoint,
            key_env: string("key_env"),
            model: string("model"),
        })
    }

    /// True when this record points at the same place as an active configuration.
    pub fn matches_endpoint(&self, adapter: &str, endpoint: &str) -> bool {
        normalize_endpoint(&self.endpoint) == normalize_endpoint(endpoint)
            && (self.adapter.is_empty() || adapter.is_empty() || self.adapter == adapter)
    }
}

/// Host part of a URL, without scheme, port or path.
pub fn endpoint_host(endpoint: &str) -> String {
    endpoint
        .split("://")
        .nth(1)
        .unwrap_or(endpoint)
        .split(['/', '?'])
        .next()
        .unwrap_or("")
        .rsplit('@')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("")
        .to_string()
}

/// Comparison form of an endpoint: scheme and trailing separators dropped,
/// lowercased. Host, port and path all stay significant, so two gateways on the
/// same host are never mistaken for each other.
pub fn normalize_endpoint(endpoint: &str) -> String {
    let without_scheme = endpoint.split("://").nth(1).unwrap_or(endpoint);
    let trimmed = without_scheme
        .split(['?', '#'])
        .next()
        .unwrap_or(without_scheme)
        .trim_end_matches('/');
    let trimmed = trimmed
        .strip_suffix("/chat/completions")
        .or_else(|| trimmed.strip_suffix("/models"))
        .unwrap_or(trimmed)
        .trim_end_matches('/');
    trimmed.to_ascii_lowercase()
}

/// Name of the provider serving `endpoint`, falling back to its host so an
/// unregistered endpoint is still labelled with something truthful.
pub fn label_for_endpoint(records: &[ProviderRecord], adapter: &str, endpoint: &str) -> String {
    let registered = records
        .iter()
        .find(|record| record.matches_endpoint(adapter, endpoint))
        .map(|record| record.name.clone());
    match registered {
        Some(name) => name,
        None => {
            let host = endpoint_host(endpoint);
            if host.is_empty() {
                "unconfigured".to_string()
            } else {
                host
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> ProviderRecord {
        ProviderRecord {
            name: "opencode-go".into(),
            adapter: "openai_compatible".into(),
            endpoint: "https://opencode.ai/zen/go/v1".into(),
            key_env: "OPENCODE_GO_API_KEY".into(),
            model: "glm-5.3-flash".into(),
        }
    }

    #[test]
    fn records_round_trip_through_the_dom() {
        let entry = record();
        let element = entry.to_element(ElementId::new("provider-opencode-go").unwrap());
        assert_eq!(element.kind, PROVIDER_KIND);
        assert_eq!(ProviderRecord::from_element(&element), Some(entry));

        // A record without a name or endpoint is not a provider.
        let mut nameless = ElementSnapshot::new(ElementId::mint(), PROVIDER_KIND);
        nameless
            .attributes
            .insert("endpoint".into(), TypedValue::String("https://x/v1".into()));
        assert_eq!(ProviderRecord::from_element(&nameless), None);
    }

    #[test]
    fn labels_prefer_the_registered_name_and_fall_back_to_the_host() {
        let records = vec![record()];
        assert_eq!(
            label_for_endpoint(&records, "openai_compatible", "https://opencode.ai/zen/go/v1"),
            "opencode-go"
        );
        // Trailing slash / path differences still match.
        assert_eq!(
            label_for_endpoint(&records, "openai_compatible", "https://opencode.ai/zen/go/v1/"),
            "opencode-go"
        );
        // An endpoint nobody registered is labelled by its host, never guessed.
        assert_eq!(
            label_for_endpoint(&records, "openai_compatible", "https://api.theclawbay.com/v1"),
            "api.theclawbay.com"
        );
        assert_eq!(label_for_endpoint(&records, "", ""), "unconfigured");
    }

    #[test]
    fn two_gateways_on_one_host_are_distinct() {
        let left = ProviderRecord {
            name: "left".into(),
            adapter: "openai_compatible".into(),
            endpoint: "http://127.0.0.1:8000/v1".into(),
            key_env: String::new(),
            model: String::new(),
        };
        let right = ProviderRecord {
            name: "right".into(),
            endpoint: "http://127.0.0.1:9000/v1".into(),
            ..left.clone()
        };
        assert!(left.matches_endpoint("openai_compatible", "http://127.0.0.1:8000/v1"));
        assert!(!left.matches_endpoint("openai_compatible", "http://127.0.0.1:9000/v1"));
        assert!(right.matches_endpoint("openai_compatible", "http://127.0.0.1:9000/v1/"));
        // The endpoint a client actually calls still identifies its provider.
        assert!(left.matches_endpoint("openai_compatible", "http://127.0.0.1:8000/v1/chat/completions"));
        // Same host, different path is a different gateway.
        assert!(!left.matches_endpoint("openai_compatible", "http://127.0.0.1:8000/openai/v1"));
    }

    #[test]
    fn endpoint_host_survives_the_shapes_people_type() {
        assert_eq!(endpoint_host("https://opencode.ai/zen/go/v1"), "opencode.ai");
        assert_eq!(endpoint_host("https://user@host.example:8443/v1"), "host.example");
        assert_eq!(endpoint_host("http://127.0.0.1:8000/v1"), "127.0.0.1");
        assert_eq!(endpoint_host("api.example.com/v1"), "api.example.com");
    }
}
