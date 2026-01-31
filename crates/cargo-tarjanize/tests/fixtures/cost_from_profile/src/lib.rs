//! Test: Cost populated from profile data.

pub fn profiled_fn() -> i32 {
    // Some computation to generate profile timing
    let mut sum = 0;
    for i in 0..100 {
        sum += i;
    }
    sum
}
