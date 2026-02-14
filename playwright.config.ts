import { defineConfig } from "@playwright/test";

// Per-test timeout in milliseconds.
const TEST_TIMEOUT_MS = 30_000;

// No automatic retries -- failures should be investigated immediately.
const RETRY_COUNT = 0;

// Playwright configuration for tarjanize end-to-end browser tests.
//
// The global setup script (e2e/global-setup.ts) starts the tarjanize
// viz web server on an ephemeral port and writes the URL to a temp
// file. Tests read the URL from the BASE_URL environment variable.
export default defineConfig({
  testDir: "./e2e",
  timeout: TEST_TIMEOUT_MS,
  retries: RETRY_COUNT,
  globalSetup: "./e2e/global-setup.ts",
  globalTeardown: "./e2e/global-teardown.ts",
  use: {
    browserName: "chromium",
    headless: true,
    trace: "on-first-retry",
  },
});
