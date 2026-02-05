//! Second integration test file.
//!
//! This compiles as a separate crate with `--crate-name second_integration`.

use integration_test_target_key::add;

#[test]
fn integration_test_add_negative() {
    assert_eq!(add(-5, 10), 5);
}
