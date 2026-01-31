//! Test: Closure in impl method aggregates to impl block.

pub struct MyType;

impl MyType {
    pub fn method_with_closure(&self) -> i32 {
        let closure = |x: i32| x + self.get_value();
        closure(10)
    }

    fn get_value(&self) -> i32 {
        5
    }
}
