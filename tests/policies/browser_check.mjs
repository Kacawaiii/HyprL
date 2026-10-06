// Optional local browser proof. Dependencies and screenshots stay under ignored var/.
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

const base = process.argv[2];
const endpoint = new URL(base);
assert.ok(endpoint.protocol === 'http:' && ['127.0.0.1', 'localhost'].includes(endpoint.hostname), 'local synthetic API only');
const runtime = path.resolve('var/policy-browser');
const temporary = path.join(runtime, 'tmp');
mkdirSync(temporary, { recursive: true });
process.env.TMPDIR = temporary;
const require = createRequire(path.join(runtime, 'package.json'));
const { chromium: playwright } = require('playwright-core');
const chromiumModule = require('@sparticuz/chromium');
const chromium = chromiumModule.default ?? chromiumModule;
const browser = await playwright.launch({ args: chromium.args, executablePath: await chromium.executablePath(), headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1365, height: 900 }, reducedMotion: 'reduce' });
  await context.route('**/*', route => route.request().url().startsWith(base + '/') ? route.continue() : route.abort());
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base + '/policies?product=ETH-USD&start=2026-06-01&end=2026-06-30&model=selected-model');
  await page.getByRole('heading', { name: 'Probability calibration · ETH-USD' }).waitFor();
  assert.equal(await page.getByRole('img', { name: /Reliability diagram/ }).count(), 1);
  assert.equal(await page.getByText('AMBIGUOUS BAR', { exact: true }).count(), 1);
  await page.getByRole('button', { name: 'Expert', exact: true }).click();
  await page.getByRole('table', { name: 'Calibrated reliability bins' }).waitFor();
  assert.ok(page.url().includes('model=selected-model'));
  await page.getByRole('combobox').selectOption('BTC-USD');
  await page.getByRole('heading', { name: 'Probability calibration · BTC-USD' }).waitFor();
  await page.screenshot({ path: path.join(runtime, 'expert.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Beginner', exact: true }).click();
  await page.screenshot({ path: path.join(runtime, 'mobile.png'), fullPage: true });
  const width = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, width: innerWidth }));
  assert.ok(width.scroll <= width.width, JSON.stringify(width));
  await page.getByRole('button', { name: 'Beginner', exact: true }).focus();
  for (let step = 0; step < 3; step++) await page.keyboard.press('Tab');
  assert.equal(await page.getByRole('combobox').evaluate(el => el === document.activeElement), true);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ state: 'PASS', synthetic: true, assertions: 9, viewports: 2, browser: await browser.version(), page_errors: errors.length }));
} finally {
  await browser.close();
}
