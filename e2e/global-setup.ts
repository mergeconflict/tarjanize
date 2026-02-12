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
import { join } from "node:path";

const STATE_FILE = join(import.meta.dirname, ".server-state.json");
const FIXTURE = join(import.meta.dirname, "fixture.json.gz");

// How long to wait for the server to print its listening URL.
const STARTUP_TIMEOUT_MS = 120_000;

export default async function globalSetup() {
  // Decompress the fixture and pipe into the viz server's stdin.
  const gunzip = spawn("gunzip", ["-c", FIXTURE], {
    stdio: ["ignore", "pipe", "ignore"],
  });

  const server = spawn(
    "cargo",
    ["run", "--bin", "tarjanize", "--", "viz"],
    {
      cwd: join(import.meta.dirname, ".."),
      stdio: [gunzip.stdout, "pipe", "pipe"],
      // Prevent the server from opening the browser automatically.
      env: { ...process.env, BROWSER: "echo" },
    },
  );

  // Collect stderr to find the "Listening on ..." line.
  const url = await new Promise<string>((resolve, reject) => {
    let stderr = "";

    const timeout = setTimeout(() => {
      server.kill();
      gunzip.kill();
      reject(
        new Error(
          `Server did not start within ${STARTUP_TIMEOUT_MS}ms.\nStderr: ${stderr}`,
        ),
      );
    }, STARTUP_TIMEOUT_MS);

    server.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
      const match = stderr.match(/Listening on (http:\/\/[^\s]+)/);
      if (match) {
        clearTimeout(timeout);
        resolve(match[1]);
      }
    });

    server.on("error", (err) => {
      clearTimeout(timeout);
      reject(new Error(`Failed to start server: ${err.message}`));
    });

    server.on("exit", (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        reject(
          new Error(
            `Server exited with code ${code} before ready.\nStderr: ${stderr}`,
          ),
        );
      }
    });
  });

  // Write server PID and URL so teardown and tests can find them.
  writeFileSync(STATE_FILE, JSON.stringify({ pid: server.pid, url }));

  // Also set environment variable for the test process.
  process.env.BASE_URL = url;
}
