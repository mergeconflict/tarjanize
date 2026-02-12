// Shared helpers for e2e tests.
//
// Reads the server URL from the state file written by global-setup.ts.

import { readFileSync } from "node:fs";
import { join } from "node:path";

const STATE_FILE = join(import.meta.dirname, ".server-state.json");

// Read the base URL from the state file left by global-setup.
export function getBaseUrl(): string {
  const state = JSON.parse(readFileSync(STATE_FILE, "utf-8"));
  return state.url;
}
