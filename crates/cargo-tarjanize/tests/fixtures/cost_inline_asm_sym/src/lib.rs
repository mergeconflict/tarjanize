//! Test: Inline asm `sym` operands create dependencies.
//!
//! Note: This test is architecture-specific (x86_64).

#[cfg(target_arch = "x86_64")]
pub fn target_fn() -> i32 {
    42
}

#[cfg(target_arch = "x86_64")]
pub fn uses_asm_sym() -> i32 {
    let result: i32;
    unsafe {
        std::arch::asm!(
            "call {fn_ptr}",
            fn_ptr = sym target_fn,
            out("eax") result,
            clobber_abi("C"),
        );
    }
    result
}

// Fallback for non-x86_64
#[cfg(not(target_arch = "x86_64"))]
pub fn target_fn() -> i32 {
    42
}

#[cfg(not(target_arch = "x86_64"))]
pub fn uses_asm_sym() -> i32 {
    target_fn()
}
