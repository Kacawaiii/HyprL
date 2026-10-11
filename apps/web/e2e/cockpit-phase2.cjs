/** Real or synthetic snapshots, read-only screenshots. Never follows source links. */
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.COCKPIT_URL || 'http://127.0.0.1:8792';
if (!['127.0.0.1', 'localhost', '[::1]'].includes(new URL(base).hostname)) throw new Error('Loopback URL required');
const out = process.env.COCKPIT_SCREENSHOTS;
if (!out) throw new Error('Set COCKPIT_SCREENSHOTS to a private output directory outside Git');
const repo = path.resolve(__dirname, '../../..');
if (path.resolve(out) === repo || path.resolve(out).startsWith(repo + path.sep)) throw new Error('Screenshots must stay outside Git');
const checks = [
  ['radar', '/', '.radar-event'],
  ['markets', '/markets', '[aria-label="Marchés et décisions"]'],
  ['news', '/news-value', '.news-comparison'],
  ['system', '/system', '[aria-label="Santé du radar et des unités"]'],
  ['paper', '/paper', '.paper-accounts'],
  ['trader', '/trader?date=2026-10-10', '[aria-label="Décisions IA sourcées"]'],
];
(async () => {
  fs.mkdirSync(out, { recursive: true, mode: 0o700 });
  const browser = await chromium.launch({ headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      for (const [name, route, selector] of checks) {
        const page = await browser.newPage({ viewport: { width, height: 1000 } });
        const errors = [], rejected = [], unavailable = [];
        // Browser access is restricted to the loopback app. Primary/feed links
        // remain ordinary anchors and are never opened by this procedure.
        await page.route('**/*', (r) => {
          const url = new URL(r.request().url());
          if (url.origin === new URL(base).origin) return r.continue();
          rejected.push(url.hostname); return r.abort();
        });
        page.on('pageerror', (e) => errors.push(e.message));
        page.on('response', (r) => { if (r.status() >= 400) unavailable.push({ path: new URL(r.url()).pathname, status: r.status() }); });
        page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push(m.text()); });
        await page.goto(base + route, { waitUntil: 'networkidle' });
        await page.locator(selector).first().waitFor({ timeout: 30000 });
        if (name === 'markets') {
          await page.getByRole('combobox', { name: 'Actif', exact: true }).selectOption('XLE');
          await page.waitForTimeout(100);
          const chart = page.getByRole('group', { name: /prix et événements de décision/ });
          const marker = chart.getByRole('button').first();
          await marker.focus(); await page.keyboard.press('Enter');
          const before = await chart.getByRole('button', { name: /Événements radar/ }).count();
          await page.getByRole('checkbox', { name: 'Événements radar' }).uncheck();
          if (await chart.getByRole('button', { name: /Événements radar/ }).count()) throw new Error('Radar layer did not hide');
          await page.getByRole('checkbox', { name: 'Événements radar' }).check();
          results.push({ check: 'market-filter-and-keyboard', width, events: before });
        }
        if (name === 'paper') {
          const detail = page.locator('.paper-account .decision-detail').first();
          if (await detail.count()) await detail.locator('summary').click();
        }
        const overflow = await page.evaluate(() => {
          const content = document.querySelector('.content');
          return document.documentElement.scrollWidth > window.innerWidth + 1 || (content && content.scrollWidth > content.clientWidth + 1);
        });
        const file = `${name}-${width === 390 ? 'mobile-390' : 'desktop'}.png`;
        await page.setViewportSize({ width, height: width === 390 ? 2000 : 1700 });
        await page.screenshot({ path: path.join(out, file) });
        fs.chmodSync(path.join(out, file), 0o600);
        results.push({ page: name, width, screenshot: file, errors, unavailableEndpoints: unavailable, horizontalOverflow: Boolean(overflow), externalRequests: rejected });
        if (errors.length || overflow || rejected.length) throw new Error(`${name}/${width} browser check failed`);
        await page.close();
      }
    }
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(out, 'browser-checks.json'), JSON.stringify(results, null, 2), { mode: 0o600 });
  }
  console.log(JSON.stringify(results));
})().catch((e) => { console.error(e.message); process.exitCode = 1; });
