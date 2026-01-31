//! Test: Macro-generated symbols have file paths.

macro_rules! make_fn {
    ($name:ident) => {
        pub fn $name() {}
    };
}

make_fn!(generated_function);
