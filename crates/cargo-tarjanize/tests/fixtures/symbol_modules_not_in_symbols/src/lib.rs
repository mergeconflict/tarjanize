//! Test: Modules are not in the symbol list (they are in submodules).

pub mod my_module {
    pub fn inner_fn() {}
}

pub fn outer_fn() {}
