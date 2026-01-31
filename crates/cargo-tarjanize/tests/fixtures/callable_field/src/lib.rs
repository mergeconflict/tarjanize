//! Test: Calling a field that is a function creates edge.

pub fn target_fn() {}

pub struct Container {
    pub callback: fn(),
}

pub fn caller(c: Container) {
    (c.callback)();
}

pub fn make_container() -> Container {
    Container { callback: target_fn }
}
