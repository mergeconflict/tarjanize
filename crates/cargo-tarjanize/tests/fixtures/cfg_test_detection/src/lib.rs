// Regular function - should be in lib target.
pub fn regular_function() -> i32 {
    42
}

// Test module - should only be in test target.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regular() {
        assert_eq!(regular_function(), 42);
    }

    // Helper function only available in tests.
    fn test_helper() -> i32 {
        100
    }
}

// Test-only function at module level.
#[cfg(test)]
pub fn test_only_function() -> i32 {
    999
}

// Regular struct.
pub struct RegularStruct {
    pub value: i32,
}

// Test-only struct.
#[cfg(test)]
pub struct TestOnlyStruct {
    pub test_value: i32,
}
