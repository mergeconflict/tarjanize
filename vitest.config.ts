// vitest.config.ts — Vitest configuration for JS unit tests.
//
// Excludes e2e tests (run via Playwright) from the vitest runner.

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    exclude: ["e2e/**", "node_modules/**"],
  },
});
