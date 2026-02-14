// color.ts — HSL-to-hex color conversion for the Gantt chart renderer.
//
// Converts CSS HSL strings (from slackColor in logic.ts) to packed
// 0xRRGGBB integers that PixiJS uses for fill colors.

import {
  HUE_DIVISOR, PERCENT_DIVISOR, RGB_SCALE, RED_SHIFT, GREEN_SHIFT,
  HUE_SIXTH, HUE_HALF, HUE_TWO_THIRDS, HUE_THIRD_OFFSET, HUE_RGB_FACTOR,
  LIGHTNESS_MIDPOINT, HALF_DIVISOR, HSL_GROUP_HUE, HSL_GROUP_SAT,
  HSL_GROUP_LIT, COLOR_FALLBACK_WHITE, ORIGIN, UNIT,
} from "./constants.ts";

const HSL_REGEX = /hsl\((?<hue>\d+),\s*(?<sat>\d+)%,\s*(?<lit>\d+)%\)/v;

const hue2rgb = (p: number, q: number, tInput: number): number => {
  let t = tInput;
  if (t < ORIGIN) t += UNIT;
  if (t > UNIT) t -= UNIT;
  if (t < HUE_SIXTH) return p + (q - p) * HUE_RGB_FACTOR * t;
  if (t < HUE_HALF) return q;
  if (t < HUE_TWO_THIRDS) return p + (q - p) * (HUE_TWO_THIRDS - t) * HUE_RGB_FACTOR;
  return p;
};

const channelToInt = (v: number): number => Math.round(v * RGB_SCALE);

// Convert an HSL string like "hsl(120, 50%, 60%)" to a packed 0xRRGGBB integer.
export function hslToHex(hslString: string): number {
  const match = HSL_REGEX.exec(hslString);
  if (match === null) return COLOR_FALLBACK_WHITE;
  const hue = Number.parseInt(match[HSL_GROUP_HUE], 10) / HUE_DIVISOR;
  const saturation = Number.parseInt(match[HSL_GROUP_SAT], 10) / PERCENT_DIVISOR;
  const lightness = Number.parseInt(match[HSL_GROUP_LIT], 10) / PERCENT_DIVISOR;
  if (saturation === ORIGIN) {
    return (channelToInt(lightness) << RED_SHIFT) | (channelToInt(lightness) << GREEN_SHIFT) | channelToInt(lightness);
  }
  const q = lightness < LIGHTNESS_MIDPOINT ? lightness * (UNIT + saturation) : lightness + saturation - lightness * saturation;
  const p = HALF_DIVISOR * lightness - q;
  const red = hue2rgb(p, q, hue + HUE_THIRD_OFFSET);
  const green = hue2rgb(p, q, hue);
  const blue = hue2rgb(p, q, hue - HUE_THIRD_OFFSET);
  return (channelToInt(red) << RED_SHIFT) | (channelToInt(green) << GREEN_SHIFT) | channelToInt(blue);
}
