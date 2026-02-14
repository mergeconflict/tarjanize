// Shared helpers for e2e tests.
//
// Reads the server URL from the state file written by global-setup.ts.

import { readFileSync } from "node:fs";
import path from "node:path";

const STATE_FILE = path.join(import.meta.dirname, ".server-state.json");

/** Shape of the JSON state file written by global-setup. */
interface ServerState {
  pid: number;
  url: string;
}

/** Type guard for the server state JSON written by global-setup. */
function isServerState(value: unknown): value is ServerState {
  if (typeof value !== "object" || value === null) return false;
  return (
    "pid" in value &&
    typeof value.pid === "number" &&
    "url" in value &&
    typeof value.url === "string"
  );
}

// Read the base URL from the state file left by global-setup.
export function getBaseUrl(): string {
  const parsed: unknown = JSON.parse(readFileSync(STATE_FILE, "utf8"));
  if (!isServerState(parsed)) {
    throw new Error("Invalid server state file");
  }
  return parsed.url;
}
