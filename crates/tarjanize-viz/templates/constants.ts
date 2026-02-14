// constants.ts — Named constants for the tarjanize Gantt chart renderer.
//
// Extracted from renderer.ts to keep the main module under the max-lines
// limit while giving every magic number a meaningful name.

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

// Left padding in the chart area to avoid bars starting at the very edge.
export const CHART_PADDING_LEFT = 20;

// Right padding so the last bar doesn't hug the viewport boundary.
export const CHART_PADDING_RIGHT = 40;

// Height in pixels reserved for the fixed time axis at the bottom.
export const AXIS_HEIGHT = 30;

// Bar height as a fraction of lane height, leaving a small gap between lanes.
export const BAR_HEIGHT_RATIO = 0.8;

// Minimum / maximum lane height bounds for the adaptive calculation.
export const MIN_LANE_HEIGHT = 1;
export const MAX_LANE_HEIGHT = 20;

// Minimum screen-space lane height (pixels) to show labels.
export const LABEL_MIN_LANE_PX = 14;

// Padding between the bar edge and its text label, in screen pixels.
export const LABEL_PAD_PX = 3;

// Padding between the tooltip and the cursor, in pixels.
export const TOOLTIP_PAD = 12;

// Font size for bar labels and axis ticks.
export const FONT_SIZE = 10;

// Resolution multiplier for text rasterization (4x keeps text crisp at 4x zoom).
export const TEXT_RESOLUTION = 4;

// Number of decimal places for the parallelism ratio display.
export const PARALLELISM_DECIMALS = 2;

// Tick mark height in pixels for the time axis.
export const TICK_HEIGHT = 6;

// ---------------------------------------------------------------------------
// Alpha values
// ---------------------------------------------------------------------------

// Alpha for non-critical-path bars when nothing is hovered.
export const DEFAULT_ALPHA = 0.4;

// Alpha for bars on the critical path through the hovered target.
export const HIGHLIGHT_ALPHA = 1;

// Alpha for dimmed bars during hover (not on the local critical path).
export const DIM_ALPHA = 0.15;

// Alpha for one-hop dependency edges (not on critical path).
export const EDGE_OFFPATH_ALPHA = 0.25;

// Alpha for critical-path edges.
export const CRITICAL_EDGE_ALPHA = 0.8;

// ---------------------------------------------------------------------------
// Zoom
// ---------------------------------------------------------------------------

// Zoom speed factor. Each wheel tick multiplies or divides the scale by this.
export const ZOOM_FACTOR = 1.1;

// Minimum scale factor for zoom (prevents zoom-out below 1:1).
export const MIN_SCALE = 1;

// ---------------------------------------------------------------------------
// PixiJS hex color constants
// ---------------------------------------------------------------------------

export const COLOR_BACKGROUND = 0x1A_1A_2E;
export const COLOR_WHITE = 0xFF_FF_FF;
export const COLOR_AXIS_BG = 0x16_21_3E;
export const COLOR_AXIS_BORDER = 0x33_33_33;
export const COLOR_AXIS_TICK = 0x66_66_66;
export const COLOR_AXIS_LABEL = 0x99_99_99;
export const COLOR_EDGE_CRITICAL = 0x00_D4_FF;
export const COLOR_EDGE_OFFPATH = 0x88_88_88;
export const COLOR_FALLBACK_WHITE = 0xFF_FF_FF;

// ---------------------------------------------------------------------------
// Edge drawing
// ---------------------------------------------------------------------------

// Stroke width for critical-path edges.
export const CRITICAL_EDGE_WIDTH = 2;

// Stroke width for axis and off-path elements.
export const THIN_STROKE_WIDTH = 1;

// Minimum edge length below which we skip drawing (avoids zero-length edges).
export const MIN_EDGE_LENGTH = 1;

// Dashed line segment length for off-path edges.
export const DASH_LENGTH = 6;

// Gap length between dashed segments for off-path edges.
export const GAP_LENGTH = 4;

// ---------------------------------------------------------------------------
// HSL conversion
// ---------------------------------------------------------------------------

// Divisors for converting HSL integer values to 0..1 range.
export const HUE_DIVISOR = 360;
export const PERCENT_DIVISOR = 100;

// RGB channel scale factor and bit-shift offsets.
export const RGB_SCALE = 255;
export const RED_SHIFT = 16;
export const GREEN_SHIFT = 8;

// HSL-to-RGB fractional thresholds and multipliers.
export const HUE_SIXTH = 1 / 6;
export const HUE_HALF = 1 / 2;
export const HUE_TWO_THIRDS = 2 / 3;
export const HUE_THIRD_OFFSET = 1 / 3;
export const HUE_RGB_FACTOR = 6;
export const LIGHTNESS_MIDPOINT = 0.5;

// Multiplier used in several HSL/coordinate calculations (dividing by 2).
export const HALF_DIVISOR = 2;

// HSL regex group indices for positional match groups.
export const HSL_GROUP_HUE = 1;
export const HSL_GROUP_SAT = 2;
export const HSL_GROUP_LIT = 3;

// ---------------------------------------------------------------------------
// Numeric identity constants (for lint compliance in PixiJS API calls)
// ---------------------------------------------------------------------------

// Origin coordinate for Graphics.rect() and Container positioning.
export const ORIGIN = 0;

// Unit value for scale factors, loop offsets, and additive identity.
export const UNIT = 1;

// Negative unit for zoom direction calculation.
export const NEGATIVE_UNIT = -1;

// Zoom direction threshold (deltaY comparison).
export const ZOOM_DIRECTION_POSITIVE = 0;
