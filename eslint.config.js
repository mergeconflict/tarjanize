import js from "@eslint/js";
import tseslint from "typescript-eslint";
import love from "eslint-config-love";
import unicorn from "eslint-plugin-unicorn";

export default tseslint.config(
  // Global ignores.
  {
    ignores: [
      "node_modules/**",
      "build/**",
      "dist/**",
      "target/**",
      "**/*.min.js",
      "types/generated/**",
    ],
  },

  // Base JS recommended rules.
  js.configs.recommended,

  // Strict type-checked + stylistic type-checked TypeScript rules.
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,

  // eslint-config-love: opinionated strict config from typescript-eslint creator.
  love,

  // eslint-plugin-unicorn: 100+ modern JS/TS conventions.
  unicorn.configs["recommended"],

  // TypeScript file settings: parser, project service, and aggressive rules.
  {
    files: ["**/*.ts"],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Aggressive rules from the strict setup.
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-unnecessary-condition": "error",
      "@typescript-eslint/strict-boolean-expressions": "error",
      "unicorn/no-null": "error",
      "no-console": "error",

      // Keep our unused-vars config.
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
    },
  },

  // e2e tests: allow node globals alongside browser globals.
  {
    files: ["e2e/**/*.ts"],
    rules: {
      // e2e tests use Playwright's test() which handles promises internally.
      "@typescript-eslint/no-floating-promises": "off",
    },
  },

  // Constants file: every line is a named constant, so no-magic-numbers is noise.
  {
    files: ["crates/tarjanize-viz/templates/constants.ts"],
    rules: {
      "@typescript-eslint/no-magic-numbers": "off",
    },
  },

  // Template files that need relaxed any rules (heavy DOM/PixiJS interop).
  {
    files: [
      "crates/tarjanize-viz/templates/tree.ts",
      "crates/tarjanize-viz/templates/sidebar.ts",
    ],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unsafe-argument": "off",
      "@typescript-eslint/no-unsafe-assignment": "off",
      "@typescript-eslint/no-unsafe-call": "off",
      "@typescript-eslint/no-unsafe-member-access": "off",
      "@typescript-eslint/no-unsafe-return": "off",
    },
  },

  // Plain JS files (config files, JS tests) are not part of the TypeScript
  // project so type-checked rules must be disabled to avoid parser errors.
  {
    files: ["**/*.js"],
    ...tseslint.configs.disableTypeChecked,
  },

  // JS test files: numeric literals and null values are test data, not magic
  // numbers or avoidable nulls.
  {
    files: ["js-tests/**/*.js"],
    rules: {
      "@typescript-eslint/no-magic-numbers": "off",
      "unicorn/no-null": "off",
      "unicorn/numeric-separators-style": "off",
    },
  },
);
