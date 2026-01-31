//! Test: Macro-generated struct in module has path.

pub mod inner {
    macro_rules! make_struct {
        ($name:ident) => {
            pub struct $name;
        };
    }

    make_struct!(NestedGeneratedStruct);
}
