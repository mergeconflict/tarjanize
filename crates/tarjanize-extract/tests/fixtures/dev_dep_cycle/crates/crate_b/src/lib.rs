/// A function in crate_b that depends on crate_a.
pub fn function_in_b() -> i32 {
    crate_a::function_in_a()
}
