//! Test: Macro-generated struct has path.

macro_rules! make_struct {
    ($name:ident) => {
        pub struct $name;
    };
}

make_struct!(GeneratedStruct);
