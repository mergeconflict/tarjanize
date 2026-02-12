//! Schedule data types — re-exported from `tarjanize-schedule`.
//!
//! The canonical definitions live in `tarjanize_schedule::data`. This module
//! re-exports them so existing code in `tarjanize-viz` (html.rs, schedule.rs)
//! continues to compile without changing import paths.

pub use tarjanize_schedule::data::{ScheduleData, Summary, TargetData};
