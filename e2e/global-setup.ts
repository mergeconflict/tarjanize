// Global setup: build tarjanize and start the viz web server.
//
// The server binds to an ephemeral port and prints "Listening on
// http://127.0.0.1:PORT" to stderr. We capture that to determine
// the URL, then store the server PID and URL in a temp file so
// global-teardown and tests can use them.
//
// The fixture is stored as fixture.json.gz to keep the repo
// manageable. We decompress it on the fly via stdin pipe.

import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import path from "node:path";

const STATE_FILE = path.join(import.meta.dirname, ".server-state.json");
const FIXTURE = path.join(import.meta.dirname, "fixture.json.gz");

// How long to wait for the server to print its listening URL.
const STARTUP_TIMEOUT_MS = 120_000;

/** Regex to capture the "Listening on http://..." line from server stderr. */
const LISTENING_RE = /Listening on (?<url>http:\/\/[^\s]+)/v;

/** Exit code indicating a normal shutdown. */
const SUCCESS_EXIT_CODE = 0;

/** Index of the first regex capture group match. */
const FIRST_CAPTURE_GROUP = 1;

export default async function globalSetup(): Promise<void> {
  // Decompress the fixture and pipe into the viz server's stdin.
  const gunzip = spawn("gunzip", ["-c", FIXTURE], {
    stdio: ["ignore", "pipe", "ignore"],
  });

  const server = spawn(
    "cargo",
    ["run", "--bin", "tarjanize", "--", "viz"],
    {
      cwd: path.join(import.meta.dirname, ".."),
      stdio: [gunzip.stdout, "pipe", "pipe"],
      // Prevent the server from opening the browser automatically.
      env: { ...process.env, BROWSER: "echo" },
    },
  );

  // Collect stderr to find the "Listening on ..." line.
  const url = await waitForListeningUrl(server, gunzip);

  // Write server PID and URL so teardown and tests can find them.
  writeFileSync(
    STATE_FILE,
    JSON.stringify({ pid: server.pid, url }),
  );

  // Also set environment variable for the test process.
  process.env.BASE_URL = url;
}

/**
 * Wait for the server to print its listening URL on stderr.
 *
 * Returns the URL string, or rejects if the server fails to start
 * within the configured timeout.
 */
async function waitForListeningUrl(
  server: ReturnType<typeof spawn>,
  gunzip: ReturnType<typeof spawn>,
): Promise<string> {
  // Wrapping event-based Node.js APIs (child_process events) into a
  // promise — there is no async/await alternative for ChildProcess
  // event listeners.
  // eslint-disable-next-line promise/avoid-new -- wrapping event emitters requires explicit Promise construction
  return await new Promise<string>((resolve, reject) => {
    let stderr = "";

    const timeout = setTimeout(() => {
      server.kill();
      gunzip.kill();
      reject(
        new Error(
          `Server did not start within ${String(STARTUP_TIMEOUT_MS)}ms.\nStderr: ${stderr}`,
        ),
      );
    }, STARTUP_TIMEOUT_MS);

    server.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
      const match = LISTENING_RE.exec(stderr);
      if (match?.[FIRST_CAPTURE_GROUP] !== undefined) {
        clearTimeout(timeout);
        resolve(match[FIRST_CAPTURE_GROUP]);
      }
    });

    server.on("error", (error: Error) => {
      clearTimeout(timeout);
      reject(new Error(`Failed to start server: ${error.message}`));
    });

    server.on("exit", (code: number | null) => {
      clearTimeout(timeout);
      if (code !== SUCCESS_EXIT_CODE) {
        reject(
          new Error(
            `Server exited with code ${String(code)} before ready.\nStderr: ${stderr}`,
          ),
        );
      }
    });
  });
}
