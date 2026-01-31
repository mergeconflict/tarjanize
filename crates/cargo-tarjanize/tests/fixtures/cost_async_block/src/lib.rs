//! Test: Async block cost aggregates to containing function.

use std::future::Future;

pub fn with_async_block() -> impl Future<Output = i32> {
    async { 42 }
}
