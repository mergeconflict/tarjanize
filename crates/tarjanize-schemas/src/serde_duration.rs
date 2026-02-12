//! Serde helpers for `Duration` ↔ f64 millisecond serialization.
//!
//! JSON consumers (frontend JS, regression math) work with f64
//! milliseconds. Internally we use `Duration` for exact integer
//! nanosecond arithmetic, avoiding floating-point comparison pitfalls.
//! These modules bridge the gap at the serialization boundary.

use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Serialize a `Duration` as f64 milliseconds.
pub fn serialize<S: Serializer>(
    dur: &Duration,
    s: S,
) -> Result<S::Ok, S::Error> {
    dur.as_secs_f64().mul_add(1000.0, 0.0).serialize(s)
}

/// Deserialize f64 milliseconds into a `Duration`.
pub fn deserialize<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<Duration, D::Error> {
    let ms = f64::deserialize(d)?;
    Ok(Duration::from_secs_f64(ms / 1000.0))
}

/// Serializes `Option<Duration>` as `Option<f64>` milliseconds.
///
/// Apply via `#[serde(with = "...::option")]` on an
/// `Option<Duration>` field. `None` maps to JSON `null`.
pub mod option {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    /// Serialize an `Option<Duration>` as `Option<f64>` milliseconds.
    pub fn serialize<S: Serializer>(
        dur: &Option<Duration>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match dur {
            Some(d) => s.serialize_some(&d.as_secs_f64().mul_add(1000.0, 0.0)),
            None => s.serialize_none(),
        }
    }

    /// Deserialize `Option<f64>` milliseconds into `Option<Duration>`.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Duration>, D::Error> {
        let opt = Option::<f64>::deserialize(d)?;
        Ok(opt.map(|ms| Duration::from_secs_f64(ms / 1000.0)))
    }
}
