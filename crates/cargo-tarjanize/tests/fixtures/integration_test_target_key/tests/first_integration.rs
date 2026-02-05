//! First integration test file.
//!
//! This compiles as a separate crate with `--crate-name first_integration`.

use integration_test_target_key::add;

#[test]
fn integration_test_add() {
    assert_eq!(add(10, 20), 30);
}
