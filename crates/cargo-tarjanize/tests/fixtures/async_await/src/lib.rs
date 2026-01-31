//! Test: Async/await expressions create edge.

pub async fn helper() {}

pub async fn caller() {
    helper().await;
}
