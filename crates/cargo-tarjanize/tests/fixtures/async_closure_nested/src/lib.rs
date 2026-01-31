//! Test: Nested async closure creates edge.

pub async fn helper() {}

pub fn caller() {
    let _outer = async || {
        let _inner = async || helper().await;
    };
}
