// @ts-check
import { defineConfig } from "@playwright/test";

const baseURL = "http://localhost:5173";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60 * 1000,
  use: {
    baseURL,
    headless: true,
  },
  reporter: [["list"]],
});
