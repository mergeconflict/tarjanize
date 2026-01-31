//! Test: Parent cost includes closure costs.

pub fn outer_with_closures() -> i32 {
    let add = |a: i32, b: i32| -> i32 {
        // Do some work
        let mut result = a;
        for _ in 0..10 {
            result += b;
        }
        result
    };

    let mul = |a: i32, b: i32| -> i32 {
        let mut result = 0;
        for _ in 0..a {
            result += b;
        }
        result
    };

    add(1, 2) + mul(3, 4)
}
