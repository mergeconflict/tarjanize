//! Test: Inline asm sym operands create edges to the referenced items.
//!
//! When using `sym` operands in inline assembly, the referenced functions
//! and statics should be captured as dependencies.

use std::arch::asm;

pub static TARGET_STATIC: i32 = 42;

pub fn target_fn() -> i32 {
    42
}

#[cfg(target_arch = "x86_64")]
pub fn uses_asm_sym_fn() {
    unsafe {
        // Reference a function via sym operand.
        asm!(
            "/* reference to {0} */",
            sym target_fn,
            options(nomem, nostack)
        );
    }
}

#[cfg(target_arch = "x86_64")]
pub fn uses_asm_sym_static() {
    unsafe {
        // Reference a static via sym operand.
        asm!(
            "/* reference to {0} */",
            sym TARGET_STATIC,
            options(readonly, nostack)
        );
    }
}

// Fallback for non-x86_64 architectures (like aarch64).
#[cfg(target_arch = "aarch64")]
pub fn uses_asm_sym_fn() {
    unsafe {
        asm!(
            "/* reference to {0} */",
            sym target_fn,
            options(nomem, nostack)
        );
    }
}

#[cfg(target_arch = "aarch64")]
pub fn uses_asm_sym_static() {
    unsafe {
        asm!(
            "/* reference to {0} */",
            sym TARGET_STATIC,
            options(readonly, nostack)
        );
    }
}
