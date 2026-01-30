/// A function in crate_a that crate_b depends on.
pub fn function_in_a() -> i32 {
    42
}

#[cfg(test)]
mod tests {
    /// This test uses crate_b via the dev-dependency.
    /// The dependency edge crate_a -> crate_b should be skipped
    /// because it would create a cycle.
    #[test]
    fn test_using_crate_b() {
        assert_eq!(crate_b::function_in_b(), 42);
    }
}
