//! Build script for tarjanize-viz.
//!
//! Runs esbuild twice:
//! 1. `renderer.ts` (imports `logic.ts`) -> `bundle.js` (ESM, pixi.js external)
//! 2. `sidebar.ts` (imports `tree.ts`) -> `sidebar.js` (IIFE, self-contained)
//!
//! External imports (pixi.js) are kept external in the renderer bundle --
//! the HTML import map resolves them at runtime from CDN. The sidebar bundle
//! is fully self-contained (IIFE format) to avoid variable collisions with
//! the ESM renderer. Both outputs go to `OUT_DIR` where `server.rs` picks
//! them up via `include_str!`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let project_root = PathBuf::from(&manifest_dir)
        .join("../..")
        .canonicalize()
        .expect("could not resolve project root");

    // Prefer the project-local esbuild binary installed via `npm install`.
    let esbuild = project_root.join("node_modules/.bin/esbuild");

    // -----------------------------------------------------------------------
    // Bundle 1: renderer.ts (Gantt chart, ESM format)
    // -----------------------------------------------------------------------

    let entry = PathBuf::from(&manifest_dir).join("templates/renderer.ts");
    let output = PathBuf::from(&out_dir).join("bundle.js");

    let status = Command::new(&esbuild)
        .arg(entry.to_str().expect("non-UTF8 path"))
        .arg("--bundle")
        .arg("--format=esm")
        // pixi.js is loaded via the HTML import map, not bundled.
        .arg("--external:pixi.js")
        .arg(format!("--outfile={}", output.display()))
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "failed to run esbuild at {}: {e}\n\
                 Run `npm install` in the project root to install it.",
                esbuild.display(),
            );
        });

    assert!(status.success(), "esbuild bundling of renderer.ts failed");

    // -----------------------------------------------------------------------
    // Bundle 2: sidebar.ts (event wiring, IIFE format)
    // -----------------------------------------------------------------------

    let sidebar_entry =
        PathBuf::from(&manifest_dir).join("templates/sidebar.ts");
    let sidebar_output = PathBuf::from(&out_dir).join("sidebar.js");

    let status = Command::new(&esbuild)
        .arg(sidebar_entry.to_str().expect("non-UTF8 path"))
        .arg("--bundle")
        // IIFE avoids top-level await and variable collisions with the
        // ESM renderer bundle.
        .arg("--format=iife")
        .arg(format!("--outfile={}", sidebar_output.display()))
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "failed to run esbuild at {}: {e}\n\
                 Run `npm install` in the project root to install it.",
                esbuild.display(),
            );
        });

    assert!(status.success(), "esbuild bundling of sidebar.ts failed");

    // Rerun when any TS source file changes.
    println!("cargo:rerun-if-changed=templates/renderer.ts");
    println!("cargo:rerun-if-changed=templates/logic.ts");
    println!("cargo:rerun-if-changed=templates/color.ts");
    println!("cargo:rerun-if-changed=templates/tooltip.ts");
    println!("cargo:rerun-if-changed=templates/edges.ts");
    println!("cargo:rerun-if-changed=templates/constants.ts");
    println!("cargo:rerun-if-changed=templates/sidebar.ts");
    println!("cargo:rerun-if-changed=templates/tree.ts");
}
