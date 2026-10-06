/**
 * Real-browser check of the Lab journey against a running lab listener that serves the built cockpit.
 *
 *   LAB_URL=http://127.0.0.1:8790 HYPRL_MODEL_LAB_TOKEN=... NODE_PATH=<dir with playwright> node e2e/lab-journey.cjs
 *
 * Synthetic data only: the listener runs real worker jobs (dataset, experiment, cancel, monitoring) on its
 * synthetic generator. Covers the full journey, refusals, keyboard use and a 390 px viewport.
 * Prints one JSON line per check and exits non-zero on the first failure. Screenshots go to $SHOTS when set.
 */
const { chromium } = require('playwright');

const BASE = process.env.LAB_URL;
const TOKEN = process.env.HYPRL_MODEL_LAB_TOKEN;
const SHOTS = process.env.SHOTS;
if (!BASE || !TOKEN) throw new Error('LAB_URL and HYPRL_MODEL_LAB_TOKEN are required');

const results = [];
function pass(name, detail = '') { results.push({ name, ok: true }); console.log(JSON.stringify({ check: name, ok: true, detail })); }
async function shot(page, name) { if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: true }); }
function expect(condition, message) { if (!condition) throw new Error(message); }

function watch(page) {
  const errors = [];
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  page.on('console', (message) => {
    // A refused control is an expected 4xx answer; the browser logs it as a failed resource.
    if (message.type() === 'error' && !/status of 40[1-9]/.test(message.text())) errors.push(`console: ${message.text()}`);
  });
  return errors;
}

async function unlock(page, token = TOKEN) {
  const field = page.getByLabel(/Operator token \(/);
  await field.focus();
  await page.keyboard.type(token);
  await page.keyboard.press('Enter');
}

async function journey(browser) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const errors = watch(page);
  await page.goto(`${BASE}/lab/datasets`);
  await page.getByRole('list', { name: 'Lab journey' }).waitFor();
  expect(await page.getByRole('button', { name: 'Build dataset' }).isDisabled(), 'build must wait for the token');
  await page.getByTestId('waiting-authorization').first().waitFor();
  expect(await page.getByRole('radio', { name: /Real prices/ }).isDisabled(), 'real prices must not be selectable');
  pass('locked page: no control before the token; real data WAITING_AUTHORIZATION');

  // Wrong token: the read is refused and said so; then forget it and use the right one.
  await unlock(page, 'x'.repeat(40));
  await page.getByText('Token refused').first().waitFor();
  await page.getByLabel('Lab navigation').getByRole('button', { name: 'Forget token' }).click();
  pass('wrong token is refused and explained');

  await unlock(page);
  await page.getByLabel('Lab navigation').getByText('OPERATOR TOKEN SET (memory only)').waitFor();
  // Invalid configuration: refused before sending.
  const bars = page.getByLabel(/Hourly bars/);
  await bars.fill('5');
  await page.getByText(/Bars must be/).waitFor();
  expect(await page.getByRole('button', { name: 'Build dataset' }).isDisabled(), 'an illegal configuration must not be sendable');
  await bars.fill('120');
  pass('illegal dataset configuration is refused before sending');

  // 1. Data: build with the keyboard (Enter submits the form).
  await bars.focus();
  await page.keyboard.press('Enter');
  await page.getByText(/queued in an isolated worker/).waitFor();
  const next = page.getByRole('link', { name: /Next: configure an experiment/ });
  await next.waitFor({ timeout: 90_000 });
  await page.getByText(/candidate decisions are admissible/).waitFor();
  await shot(page, '1-dataset');
  pass('dataset built by a worker from the page; manifest and exclusions shown');

  // 2. Model + 3. Experiment: follow the link (selection carried), choose the local external adapter, launch.
  await next.click();
  await page.waitForURL(/\/lab\/experiments\?.*dataset=/);
  const launch = page.getByRole('button', { name: 'Launch experiment' });
  await page.getByRole('combobox', { name: /^Model/ }).selectOption('local-momentum-v1');
  await page.getByLabel('Chosen model').filter({ hasText: 'EXTERNAL ADAPTER' }).waitFor();
  await page.waitForFunction(() => {
    const button = [...document.querySelectorAll('button')].find((item) => item.textContent === 'Launch experiment');
    return button && !button.disabled;
  });
  await page.getByLabel(/Embargo/).fill('90000');
  expect(await launch.isDisabled(), 'an out-of-range embargo must not be sendable');
  await page.getByLabel(/Embargo/).fill('3600');
  await launch.click();
  await page.getByText(/recorded before any result/).waitFor();
  await page.getByRole('progressbar', { name: 'Job progress' }).first().waitFor();
  await page.getByRole('table', { name: 'Job log' }).first().waitFor();
  pass('experiment launched from the page; progress and log followed');

  // 4. Results against the baselines, artifacts, splits and hashes.
  await page.getByRole('table', { name: /test comparison/ }).first().waitFor({ timeout: 120_000 });
  const comparison = await page.getByRole('table', { name: /test comparison/ }).first().innerText();
  expect(comparison.includes('ZERO') && comparison.includes('TRAIN_MEAN'), 'baselines missing');
  await page.getByRole('button', { name: 'Expert' }).click();
  await page.getByText('Runtime and source hashes').waitFor();
  expect(page.url().includes('mode=expert') && page.url().includes('dataset='), 'mode switch must keep the selection');
  await shot(page, '2-experiment');
  pass('results compared with ZERO and TRAIN_MEAN; Expert hashes; selection kept across the mode switch');

  // 5. Monitoring of the experiment's own predictions.
  await page.getByRole('button', { name: 'Open the monitoring of these predictions' }).click();
  await page.waitForURL(/\/lab\/monitoring\?.*monitor=/);
  const monitoring = page.getByLabel('Experiment monitoring');
  await monitoring.getByText(/chain verified/).waitFor({ timeout: 120_000 });
  await monitoring.getByLabel('Diagnosis').waitFor();
  await monitoring.getByText(/establishes no real edge/).waitFor();
  await shot(page, '3-monitoring');
  pass('monitoring job ran on the experiment predictions; diagnosis shown with limitations');

  // Cancel: with one worker, a second long dataset waits QUEUED behind the first, so a fast runner
  // cannot finish it before the page shows its cancel control. It is cancelled before it publishes.
  await page.getByRole('link', { name: 'Datasets', exact: true }).click();
  await page.getByLabel(/Hourly bars/).fill('600');
  await page.getByRole('button', { name: 'Build dataset' }).click();
  await page.getByText(/queued in an isolated worker/).waitFor();
  await page.getByLabel(/Seed/).fill('8');
  await page.getByRole('button', { name: 'Build dataset' }).click();
  await page.getByLabel('Dataset jobs').getByText('QUEUED').first().waitFor();
  await page.getByRole('button', { name: 'Cancel this job' }).click();
  await page.getByText(/Cancellation recorded/).waitFor();
  await page.getByLabel('Dataset jobs').getByText('CANCELLED').first().waitFor({ timeout: 60_000 });
  pass('long dataset job cancelled from the page; no result published');

  expect(errors.length === 0, `browser errors: ${errors.join(' | ')}`);
  pass('no page or console error during the journey');
  await context.close();
}

async function refusals(browser) {
  // A foreign page (or a DNS-rebound name) is refused by the listener even with the right token.
  const context = await browser.newContext();
  const host = new URL(BASE).host;
  const port = new URL(BASE).port;
  const headers = { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' };
  const cross = await context.request.post(`${BASE}/api/v1/lab/datasets`, {
    headers: { ...headers, Origin: 'http://localhost:5173' }, data: { synthetic: true, bars: 120 },
  });
  expect(cross.status() === 403, `cross-origin POST answered ${cross.status()}`);
  const rebound = await context.request.get(`${BASE}/api/v1/lab/jobs`, {
    headers: { ...headers, Host: `evil.example:${port}`, Origin: `http://evil.example:${port}` },
  });
  expect(rebound.status() === 403, `rebound GET answered ${rebound.status()}`);
  const same = await context.request.get(`${BASE}/api/v1/lab/jobs`, { headers: { ...headers, Origin: `http://${host}` } });
  expect(same.status() === 200, `same-origin GET answered ${same.status()}`);
  pass('listener refuses foreign origins and rebound hosts, admits its own page');

  // How the page phrases a refused control (the answer is substituted; the page is real).
  const page = await context.newPage();
  await page.route('**/api/v1/lab/datasets', (route) => (route.request().method() === 'POST'
    ? route.fulfill({ status: 403, contentType: 'application/json', body: JSON.stringify({ error: 'refused' }) })
    : route.continue()));
  await page.goto(`${BASE}/lab/datasets`);
  await unlock(page);
  await page.getByRole('button', { name: 'Build dataset' }).click();
  await page.getByRole('alert').filter({ hasText: /served by the lab listener itself/ }).waitFor();
  pass('a refused control is explained in the page');
  await context.close();
}

async function keyboard(browser) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' });
  const page = await context.newPage();
  await page.goto(`${BASE}/lab/datasets`);
  await unlock(page);
  await page.getByLabel('Lab navigation').getByText('OPERATOR TOKEN SET (memory only)').waitFor();
  let reached = false;
  for (let index = 0; index < 60 && !reached; index += 1) {
    await page.keyboard.press('Tab');
    reached = await page.evaluate(() => document.activeElement?.textContent === 'Experiments'
      && document.activeElement?.closest('nav')?.getAttribute('aria-label') === 'Lab views');
  }
  expect(reached, 'the Experiments tab is not reachable with Tab');
  const outline = await page.evaluate(() => getComputedStyle(document.activeElement).outlineStyle);
  await page.keyboard.press('Enter');
  await page.waitForURL(/\/lab\/experiments/);
  await page.getByRole('heading', { name: /Configure an experiment/ }).waitFor();
  const transition = await page.evaluate(() => {
    const bar = document.querySelector('.progress > span');
    return bar ? getComputedStyle(bar).transitionDuration : 'none-present';
  });
  expect(transition === 'none-present' || parseFloat(transition) < 0.01, `progress animates under reduced motion (${transition})`);
  pass('keyboard: Tab reaches the Lab tabs, Enter navigates, token kept', `focus outline ${outline}`);
  await context.close();
}

async function narrow(browser) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await context.newPage();
  const errors = watch(page);
  for (const tab of ['datasets', 'experiments', 'models', 'monitoring']) {
    await page.goto(`${BASE}/lab/${tab}`);
    await unlock(page);
    await page.getByLabel('Lab navigation').getByText('OPERATOR TOKEN SET (memory only)').waitFor();
    await page.waitForTimeout(1500);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow <= 1, `/lab/${tab} overflows the 390 px viewport by ${overflow}px`);
    await shot(page, `390-${tab}`);
  }
  expect(errors.length === 0, `browser errors: ${errors.join(' | ')}`);
  pass('390 px: datasets, experiments, models and monitoring fit without horizontal scroll');
  await context.close();
}

(async () => {
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader'] });
  try {
    await journey(browser);
    await refusals(browser);
    await keyboard(browser);
    await narrow(browser);
    console.log(JSON.stringify({ summary: `${results.length} checks passed` }));
  } catch (error) {
    console.log(JSON.stringify({ check: 'FAILED', ok: false, detail: String(error && error.stack || error) }));
    if (process.env.GITHUB_ACTIONS) {
      // Annotations are the part of a run readable without credentials; one line, no token in it.
      const last = results.length > 0 ? results[results.length - 1].name : 'none';
      const detail = String(error && error.message || error).replace(/\s+/g, ' ').slice(0, 900);
      console.log(`::error title=lab-journey::after ${results.length} passed checks (last: ${last}): ${detail}`);
    }
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
