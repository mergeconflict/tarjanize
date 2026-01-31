//! Test: Parent cost includes nested function costs.

pub fn outer_with_nested() -> i32 {
    fn nested_helper() -> i32 {
        // Do some work to generate measurable cost
        let mut sum = 0;
        for i in 0..50 {
            sum += i;
        }
        sum
    }
    nested_helper() + 1
}
