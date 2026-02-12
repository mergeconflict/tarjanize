// Global teardown: stop the viz web server started by global-setup.

import { readFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";

const STATE_FILE = join(import.meta.dirname, ".server-state.json");

export default async function globalTeardown() {
  try {
    const state = JSON.parse(readFileSync(STATE_FILE, "utf-8"));
    if (state.pid) {
      process.kill(state.pid, "SIGTERM");
    }
  } catch {
    // Server already stopped or state file missing — nothing to do.
  } finally {
    try {
      unlinkSync(STATE_FILE);
    } catch {
      // Ignore if already cleaned up.
    }
  }
}
