/// A simple function in the library.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// A unit test in the lib (compiled with lib as --test).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }
}
