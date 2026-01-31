//! Build script to set rpath for `rustc_private` libraries.
//!
//! When using `rustc_private`, the binary needs to find `librustc_driver` and
//! other rustc libraries at runtime. This build script queries rustc for its
//! sysroot and sets the rpath so the binary can find these libraries without
//! needing `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` to be set manually.

use std::process::Command;

fn main() {
    // Query rustc for its sysroot.
    let output = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .expect("failed to run rustc --print sysroot");

    let sysroot = String::from_utf8(output.stdout)
        .expect("rustc sysroot is not valid UTF-8")
        .trim()
        .to_string();

    // The rustc libraries are in {sysroot}/lib.
    let lib_path = format!("{sysroot}/lib");

    // Tell cargo to set the rpath so the binary can find rustc libraries.
    // On macOS this is @loader_path relative, on Linux it's $ORIGIN relative,
    // but using an absolute path works for both during development.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_path}");

    // Re-run if the toolchain changes.
    println!("cargo:rerun-if-env-changed=RUSTUP_TOOLCHAIN");
}
