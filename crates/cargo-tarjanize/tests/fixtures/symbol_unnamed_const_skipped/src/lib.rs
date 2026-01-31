//! Test: Unnamed const items are skipped.

const _: () = {
    // This is an unnamed const, typically used for compile-time assertions.
};

pub const NAMED_CONST: i32 = 42;
