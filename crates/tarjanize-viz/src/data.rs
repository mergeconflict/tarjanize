//! Schedule data types — re-exported from `tarjanize-schedule`.
//!
//! The canonical definitions live in `tarjanize_schedule::data`. This module
//! re-exports them so `tarjanize-viz` can use them without fully qualifying
//! the `tarjanize_schedule` path.

pub use tarjanize_schedule::data::{ScheduleData, Summary, TargetData};
