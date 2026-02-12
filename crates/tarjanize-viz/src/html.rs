//! HTML generation via askama template rendering.
//!
//! The TS source files (`logic.ts`, `renderer.ts`) are proper ES modules for
//! testability. At build time, `build.rs` runs esbuild to bundle them into a
//! single module (`bundle.js`). This module inlines the bundle and CSS into
//! the askama template, producing a self-contained HTML file whose only
//! external dependency is the `PixiJS` CDN (resolved via an HTML import map).

use std::io::Write;

use askama::Template;

use crate::data::ScheduleData;
use crate::error::VizError;

/// Bundled JS produced by esbuild during `build.rs`. Contains renderer.ts
/// with logic.ts inlined, pixi.js kept as an external import.
const BUNDLE_JS: &str = include_str!(concat!(env!("OUT_DIR"), "/bundle.js"));

/// Raw CSS, included at compile time. A single file with no imports, so no
/// bundling needed.
const STYLE_CSS: &str = include_str!("../templates/style.css");

/// Askama template for the visualization HTML.
///
/// The template receives the esbuild bundle, CSS, and serialized schedule
/// data. All assets are inlined, producing a self-contained HTML file with
/// no external dependencies except the `PixiJS` CDN.
#[derive(Template)]
#[template(path = "viz.html")]
struct VizTemplate {
    schedule_json: String,
    style_css: String,
    bundle_js: String,
}

/// Renders the visualization HTML to the given writer.
///
/// Serializes the schedule data to JSON and embeds it in the HTML template
/// alongside the esbuild bundle and CSS. The output is a self-contained HTML
/// file.
pub(crate) fn generate(
    data: &ScheduleData,
    mut output: impl Write,
) -> Result<(), VizError> {
    let json = serde_json::to_string(data).map_err(VizError::serialize)?;
    let template = VizTemplate {
        schedule_json: json,
        style_css: STYLE_CSS.to_owned(),
        bundle_js: BUNDLE_JS.to_owned(),
    };
    let rendered = template.render().map_err(VizError::template)?;
    output
        .write_all(rendered.as_bytes())
        .map_err(VizError::io)?;
    Ok(())
}
