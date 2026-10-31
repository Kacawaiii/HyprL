'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { createApp } = require('../server/app');
const { loadConfig } = require('../server/config');
const { VERSION, TEXT, assertNewsletterPage } = require('../server/growth/newsletter');
const { CURRENT_VERSION } = require('../server/lib/consent');
const { PublicContent, weekAt, claimId } = require('../server/growth/content');
const { fixture } = require('./member-fixtures');
const quiet = { info() {}, warn() {}, error() {} };
const workerKey = '2'.repeat(64), password = 'synthetic-growth-test-password';
const valid = { email: 'reader@example.test', consent: 'oui', consent_version: CURRENT_VERSION, newsletter: 'oui', newsletter_version: VERSION };
async function setup(t, extra = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-growth-test-'));
  let clock = Date.parse('2026-10-28T12:00:00Z');
  const mails = [], value = fixture(clock);
  value.calendar.push({ ...value.calendar[0], kind: 'earnings', symbol: 'DEMO', date: '2026-10-29' });
  value.claims[0].verdict = 'verified';
  const snapshots = { read() { return structuredClone(value); } };
  const config = loadConfig({ ADMIN_PASSWORD: password, PUBLIC_URL: 'http://localhost:9999', DATA_FILE: path.join(dir, 'subs.json'), NEWSLETTER_ENABLED: 'true', NEWSLETTER_TRACKING_DISABLED: 'true', NEWSLETTER_TOKEN_KEY: '1'.repeat(64), NEWSLETTER_WORKER_KEY: workerKey, SMTP_HOST: 'smtp.example.test', MAIL_FROM: 'letter@example.test', SMTP_EU_APPROVED: 'true', ...extra });
  const content = new PublicContent(config, { now: () => clock, snapshots });
  let app;
  const start = () => createApp(config, { now: () => clock, logger: quiet, publicContent: { snapshots }, mailer: { async send(m) { mails.push(m); if (m.subject.startsWith('La lettre') && extra.failDelivery) throw Error('synthetic SMTP uncertain'); return true; } } });
  app = start();
  let server = http.createServer(app);
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  let base = 'http://127.0.0.1:' + server.address().port;
  t.after(async () => { app.close(); await new Promise(r => server.close(r)); fs.rmSync(dir, { recursive: true, force: true }); });
  const request = (route, body, headers = {}, method = 'POST') => fetch(base + route, { method, headers: { 'Content-Type': 'application/x-www-form-urlencoded', ...headers }, body: method === 'GET' ? undefined : new URLSearchParams(body), redirect: 'manual' });
  return { get app() { return app; }, config, content, value, mails, request,
    get: route => fetch(base + route), advance: ms => { clock += ms; },
    send: () => request('/api/internal/newsletter/send', {}, { Authorization: 'Bearer ' + workerKey }),
    async restart() { app.close(); await new Promise(r => server.close(r)); app = start(); server = http.createServer(app); await new Promise(r => server.listen(0, '127.0.0.1', r)); base = 'http://127.0.0.1:' + server.address().port; },
    async enroll(email = valid.email) { return request('/api/subscribe', { ...valid, email }); },
    confirmation() { const m = mails.findLast(m => m.subject.startsWith('Confirmez')); return new URL(m.text.match(/http:\/\/localhost:9999\/lettre\/confirmation\?token=[\w-]+/)[0]); },
    async confirm() { return request('/lettre/confirmation', { token: this.confirmation().searchParams.get('token') }); },
    async login() { const r = await fetch(base + '/api/admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password }) }); return r.headers.get('set-cookie').split(';')[0]; },
  };
}

test('newsletter consent is versioned, separate, optional, unticked and cannot drift silently', () => {
  const html = fs.readFileSync(path.join(__dirname, '../public/index.html'), 'utf8');
  assertNewsletterPage(html);
  assert.throws(() => assertNewsletterPage(html.replace(TEXT, 'different consent')));
  assert.throws(() => assertNewsletterPage(html.replace('id="newsletter"', 'checked id="newsletter"')));
  assert.throws(() => assertNewsletterPage(html.replace('id="newsletter"', 'required id="newsletter"')));
});

test('unchecked/channel-email/legacy entries never become newsletter subscribers', async t => {
  const c = await setup(t);
  const r = await c.request('/api/subscribe', { ...valid, newsletter: '', channels: 'email' });
  assert.equal(r.status, 303);
  assert.equal(c.app.store.all()[0].newsletter, undefined);
  const send = await c.send(); assert.equal(send.status, 200);
  assert.equal((await send.json()).sent, 0);
  assert.equal(c.mails.filter(m => m.subject.startsWith('Confirmez')).length, 0);
});

test('separate consent is required at the API and fails atomically when version or config is wrong', async t => {
  const c = await setup(t);
  assert.equal((await c.request('/api/subscribe', { ...valid, newsletter_version: 'stale' })).status, 400);
  assert.equal(c.app.store.all().length, 0);
  c.config.newsletter.enabled = false;
  assert.equal((await c.enroll()).status, 503);
  assert.equal(c.app.store.all().length, 0);
});

test('double opt-in uses a hashed 48h token; GET/scanners never activate; POST is one-use', async t => {
  const c = await setup(t);
  await c.enroll();
  const link = c.confirmation(), token = link.searchParams.get('token');
  let sub = c.app.store.all()[0];
  assert.equal(sub.newsletter.status, 'pending');
  assert.equal(sub.newsletter.text, TEXT);
  assert.ok(!JSON.stringify(sub).includes(token));
  assert.equal((await c.get(link.pathname + link.search)).status, 200);
  assert.equal(c.app.store.all()[0].newsletter.status, 'pending');
  assert.equal((await (await c.send()).json()).sent, 0);
  assert.equal((await c.request('/lettre/confirmation', { token }, { Origin: 'https://foreign.example.test' })).status, 403);
  assert.equal((await c.request('/lettre/confirmation', { token }, { Origin: 'null' })).status, 200);
  assert.equal((await c.confirm()).status, 404);
  sub = c.app.store.all()[0];
  assert.equal(sub.newsletter.status, 'confirmed');
  assert.ok(sub.newsletter.confirmed_at);
  assert.equal(sub.newsletter.confirmation_hash, undefined);
  assert.equal((await c.request('/lettre/confirmation', { token: 'x'.repeat(43) })).status, 404);
});

test('expired confirmation, rate limited resend and enrollment for an existing early-access address', async t => {
  const c = await setup(t);
  await c.request('/api/subscribe', { ...valid, newsletter: '' });
  await c.enroll(); await c.enroll();
  assert.equal(c.app.store.all().length, 1);
  assert.equal(c.mails.filter(m => m.subject.startsWith('Confirmez')).length, 1);
  const old = c.confirmation().searchParams.get('token');
  c.advance(48 * 3600000);
  assert.equal((await c.request('/lettre/confirmation', { token: old })).status, 404);
  await c.enroll();
  assert.notEqual(c.confirmation().searchParams.get('token'), old);
  assert.equal((await c.request('/lettre/confirmation', { token: old })).status, 404);
  assert.equal((await c.confirm()).status, 200);
});

test('weekly delivery only confirmed recipients, durable at-most-once across retries and restarts', async t => {
  const c = await setup(t);
  await c.enroll(); await c.confirm(); await c.enroll('pending@example.test');
  const result = await (await c.send()).json();
  assert.equal(result.sent, 1);
  const letter = c.mails.find(m => m.subject.startsWith('La lettre'));
  assert.equal(letter.to, valid.email);
  assert.equal(letter.html, undefined);
  assert.match(letter.text, /DEMO/);
  assert.match(letter.text, /Chance ou pas/);
  assert.match(letter.text, /Source disponible/);
  assert.equal(letter.headers['List-Unsubscribe-Post'], 'List-Unsubscribe=One-Click');
  assert.equal((await (await c.send()).json()).sent, 0);
  await c.restart();
  assert.equal((await (await c.send()).json()).sent, 0);
  assert.equal(c.mails.filter(m => m.subject.startsWith('La lettre')).length, 1);
});

test('SMTP uncertain result is recorded and is never retried automatically', async t => {
  const c = await setup(t, { failDelivery: true });
  await c.enroll(); await c.confirm();
  const result = await (await c.send()).json();
  assert.equal(result.uncertain, 1);
  await c.restart(); await c.send();
  assert.equal(c.mails.filter(m => m.subject.startsWith('La lettre')).length, 1);
});

test('crash reservation survives restart; disk failure refuses SMTP without partially mutating state', async t => {
  const c = await setup(t); await c.enroll(); await c.confirm();
  const sub = c.app.store.all()[0], originalWrite = c.app.store._writeSync;
  c.app.store._writeSync = () => { throw Error('synthetic disk failure'); };
  assert.equal((await c.send()).status, 500);
  assert.equal(c.mails.filter(m => m.subject.startsWith('La lettre')).length, 0);
  assert.deepEqual(c.app.store.all()[0].newsletter.deliveries, {});
  c.app.store._writeSync = originalWrite;
  await c.app.store.update(sub.id, s => { s.newsletter.deliveries['2026-10-26'] = { status: 'reserved', at: '2026-10-28T12:00:00Z' }; return s; });
  await c.restart();
  assert.equal((await (await c.send()).json()).sent, 0);
  assert.equal(c.mails.filter(m => m.subject.startsWith('La lettre')).length, 0);
});

test('concurrent worker requests never duplicate a letter; general deletion removes both purposes', async t => {
  const c = await setup(t); await c.enroll(); await c.confirm();
  const responses = await Promise.all([c.send(), c.send()]);
  assert.ok(responses.every(r => [200, 409].includes(r.status)));
  assert.equal(c.mails.filter(m => m.subject.startsWith('La lettre')).length, 1);
  const early = c.mails.find(m => m.subject.startsWith('Votre inscription'));
  const deletion = new URL(early.list.unsubscribe.url);
  assert.equal((await c.request(deletion.pathname + deletion.search, { 'List-Unsubscribe': 'One-Click' })).status, 200);
  assert.equal(c.app.store.all().length, 0);
  assert.equal((await (await c.send()).json()).sent, 0);
});

test('unsubscribe GET has no effect; one-click POST stops letters and retains early-access consent', async t => {
  const c = await setup(t);
  await c.enroll(); await c.confirm(); await c.send();
  const letter = c.mails.find(m => m.subject.startsWith('La lettre'));
  const link = new URL(letter.list.unsubscribe.url);
  assert.equal((await c.get(link.pathname + link.search)).status, 200);
  assert.equal(c.app.store.all()[0].newsletter.status, 'confirmed');
  assert.equal((await c.request(link.pathname + link.search, { 'List-Unsubscribe': 'One-Click' })).status, 200);
  assert.equal(c.app.store.all()[0].newsletter.status, 'unsubscribed');
  assert.equal(c.app.store.all()[0].consent.given, true);
  assert.equal((await (await c.send()).json()).sent, 0);
  await c.enroll(); assert.equal(c.app.store.all()[0].newsletter.status, 'pending');
});

test('admin exports require authentication and expose consent evidence without any token/hash', async t => {
  const c = await setup(t); await c.enroll(); await c.confirm();
  assert.equal((await c.get('/api/admin/newsletter/export')).status, 401);
  const cookie = await c.login();
  const r = await c.request('/api/admin/newsletter/export', {}, { Cookie: cookie }, 'GET');
  assert.equal(r.status, 200);
  const output = await r.text();
  assert.match(output, /confirmed_at/); assert.match(output, /2026-10-newsletter-v1/);
  assert.ok(!/token|hash|deliveries/.test(output));
  const id = c.app.store.all()[0].id;
  const individual = await c.request('/api/admin/subscribers/' + id + '/export', {}, { Cookie: cookie }, 'GET');
  assert.equal((await individual.json()).subscriber.newsletter.status, 'confirmed');
});

test('worker authentication, SMTP gate and empty/stale content refuse sends', async t => {
  const c = await setup(t); await c.enroll(); await c.confirm();
  assert.equal((await c.request('/api/internal/newsletter/send', {})).status, 403);
  c.config.smtp.euApproved = false;
  assert.equal((await c.send()).status, 503);
  c.config.smtp.euApproved = true;
  c.value.briefs = []; c.value.calendar = []; c.value.claims = [];
  assert.equal((await c.send()).status, 503);
  c.advance(27 * 3600000);
  assert.equal((await c.send()).status, 503);
  assert.equal(c.mails.filter(m => m.subject.startsWith('La lettre')).length, 0);
});

test('public pages render sourced French HTML, JSON-LD, canonicals and sitemap, with no visitor storage', async t => {
  const c = await setup(t);
  for (const route of ['/calendrier-resultats-us', '/chance-ou-pas', '/comprendre-depot-sec', '/parrainage', '/chance-ou-pas/' + claimId(c.value.claims[0])]) {
    const res = await c.get(route); assert.equal(res.status, 200, route); assert.equal(res.headers.get('set-cookie'), null);
    const html = await res.text();
    assert.equal((html.match(/<h1[ >]/g) || []).length, 1);
    assert.match(html, /rel="canonical"/);
    const schema = JSON.parse(html.match(/<script type="application\/ld\+json">([^]*?)<\/script>/)[1]);
    assert.equal(schema.inLanguage, 'fr-FR'); assert.equal(schema.url, c.config.publicUrl + route);
    assert.match(res.headers.get('content-security-policy'), /sha256-/);
  }
  const calendar = await (await c.get('/calendrier-resultats-us')).text();
  assert.match(calendar, /2026-10-29/); assert.match(calendar, /date|Date/); assert.match(calendar, /à confirmer/);
  assert.ok(!calendar.includes('15:30'));
  const sitemap = await (await c.get('/sitemap.xml')).text();
  assert.match(sitemap, /chance-ou-pas\/[a-f0-9]{24}/);
  assert.ok(!sitemap.includes('/lettre/'));
  assert.equal((await c.get('/chance-ou-pas/' + 'a'.repeat(24))).status, 404);
});

test('missing or stale public content is explicit and never fabricates dates or articles', async t => {
  const c = await setup(t);
  c.advance(27 * 3600000);
  const response = await c.get('/calendrier-resultats-us'); assert.equal(response.status, 503);
  assert.equal(response.headers.get('retry-after'), '1800');
  assert.ok(!(await response.text()).includes('2026-10-29'));
  assert.equal((await c.get('/comprendre-depot-sec')).status, 200);
});

test('weekly boundaries follow New York date including Sunday UTC and daylight transitions', () => {
  assert.deepEqual(weekAt(Date.parse('2026-11-02T00:30:00Z')), { start: '2026-10-26', end: '2026-11-01' });
  assert.deepEqual(weekAt(Date.parse('2026-11-02T13:00:00Z')), { start: '2026-11-02', end: '2026-11-08' });
});

test('completed BTC study remains sourced and available without a fresh member export', async t => {
  const c = await setup(t), btc = require('../server/growth/btc-study');
  c.advance(27 * 3600000);
  const res = await c.get(btc.pathname);
  assert.equal(res.status, 200);
  const html = await res.text();
  for (const text of ['55,8', '58,0', '51,2', '60,3', '251/450', 'p = 0,075', 'Aucun avantage vendredi établi', '−7,1 à +2,9', btc.evidenceUrl, btc.study.evidence_sha256]) assert.ok(html.includes(text), text);
  assert.ok(html.includes('ce qui ne confirme pas'));
  const index = await c.get('/chance-ou-pas');
  assert.equal(index.status, 200);
  assert.ok((await index.text()).includes(btc.pathname));
  assert.ok((await (await c.get('/sitemap.xml')).text()).includes(btc.pathname));
});
