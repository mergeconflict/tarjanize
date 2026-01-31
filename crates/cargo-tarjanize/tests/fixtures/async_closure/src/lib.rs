//! Test: Async closure creates edge.

pub async fn helper() {}

pub fn caller() {
    let _closure = async || helper().await;
}
