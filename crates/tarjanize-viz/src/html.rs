//! HTML generation via askama template rendering.
//!
//! Compiles `viz.html` (which `{% include %}`s CSS and JS) into Rust code
//! at build time. The template embeds the serialized schedule JSON, producing
//! a self-contained HTML file with no external dependencies except the
//! `PixiJS` CDN.

use std::io::Write;

use askama::Template;

use crate::data::ScheduleData;
use crate::error::VizError;

/// Askama template for the visualization HTML.
///
/// The `schedule_json` field is serialized `ScheduleData` embedded directly
/// into the `<script>` block as a JS object literal.
#[derive(Template)]
#[template(path = "viz.html")]
struct VizTemplate {
    schedule_json: String,
}

/// Renders the visualization HTML to the given writer.
///
/// Serializes the schedule data to JSON and embeds it in the HTML template.
/// The output is a self-contained HTML file.
pub(crate) fn generate(
    data: &ScheduleData,
    mut output: impl Write,
) -> Result<(), VizError> {
    let json = serde_json::to_string(data).map_err(VizError::serialize)?;
    let template = VizTemplate {
        schedule_json: json,
    };
    let rendered = template.render().map_err(VizError::template)?;
    output
        .write_all(rendered.as_bytes())
        .map_err(VizError::io)?;
    Ok(())
}
