//! Build script for tarjanize-viz.
//!
//! Runs esbuild to bundle `renderer.ts` (imports `logic.ts`) -- the Gantt
//! chart renderer, shared between static HTML and web server modes.
//!
//! External imports (pixi.js) are kept external -- the HTML import map
//! resolves them at runtime from CDN. The bundled output goes to `OUT_DIR`
//! where `html.rs` and `server.rs` pick it up via `include_str!`.

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
    // Bundle: renderer.ts (Gantt chart)
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

    // Rerun when the TS source files change.
    println!("cargo:rerun-if-changed=templates/logic.ts");
    println!("cargo:rerun-if-changed=templates/renderer.ts");
}
