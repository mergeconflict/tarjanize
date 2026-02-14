// Global type augmentations for the tarjanize viz runtime.
//
// renderer.ts and app.html inject these properties onto the global scope.
// This declaration makes them visible to TypeScript without `as any` casts.

import type { ScheduleData } from "./generated/ScheduleData";

declare global {
  interface Window {
    /** Current schedule data, set by app.html on load. */
    DATA: ScheduleData;
    /** Replace schedule data and rebuild the Gantt chart. Set by renderer.ts. */
    updateSchedule: (data: ScheduleData) => void;
    /** Format a millisecond duration as a human-readable string. Set by renderer.ts. */
    formatMs: (ms: number) => string;
  }

  // Augment globalThis so `globalThis.DATA` etc. resolve without casts.
  // Uses `var` declarations which TypeScript merges into the global scope.
  var DATA: ScheduleData;
  var updateSchedule: (data: ScheduleData) => void;
  var formatMs: (ms: number) => string;
}
