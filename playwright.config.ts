import { defineConfig } from "@playwright/test";

// Playwright configuration for tarjanize end-to-end browser tests.
//
// The global setup script (e2e/global-setup.ts) starts the tarjanize
// viz web server on an ephemeral port and writes the URL to a temp
// file. Tests read the URL from the BASE_URL environment variable.
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: 0,
  globalSetup: "./e2e/global-setup.ts",
  globalTeardown: "./e2e/global-teardown.ts",
  use: {
    browserName: "chromium",
    headless: true,
    trace: "on-first-retry",
  },
});
