import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const ROUTES = ["/", "/models", "/ingest", "/auth", "/license", "/admin/reset"];

test.describe("accessibility", () => {
  for (const route of ROUTES) {
    test(`axe scan ${route}`, async ({ page }) => {
      // hash-based routing, keep consistent anchor navigation
      const target = route === "/" ? "#/" : `#${route}`;
      await page.goto(target);

      const results = await new AxeBuilder({ page }).analyze();
      expect(results.violations).toEqual([]);
    });
  }
});
