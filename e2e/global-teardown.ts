// Global teardown: stop the viz web server started by global-setup.

import { readFileSync, unlinkSync } from "node:fs";
import path from "node:path";

const STATE_FILE = path.join(import.meta.dirname, ".server-state.json");

/** PID value indicating no process was recorded. */
const NO_PID = 0;

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

export default function globalTeardown(): void {
  try {
    const parsed: unknown = JSON.parse(readFileSync(STATE_FILE, "utf8"));
    if (isServerState(parsed) && parsed.pid !== NO_PID) {
      process.kill(parsed.pid, "SIGTERM");
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
