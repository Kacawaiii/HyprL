'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path'), http = require('node:http');
const { loadConfig } = require('../server/config');
const { createApp } = require('../server/app');
const { CURRENT_VERSION } = require('../server/lib/consent');
const { VERSION } = require('../server/growth/newsletter');
const { LIMITS, SignupGuard } = require('../server/lib/signup-guard');
const { SubscriberStore } = require('../server/lib/store');
const { validateActions } = require('../server/lib/signup');
const { solve } = require('./signup-client');
const HOUR = 3600000, DAY = 24 * HOUR;
const valid = { email: 'person@example.test', consent: 'oui', consent_version: CURRENT_VERSION, actions: 'AAPL, Bank of America', newsletter: 'oui', newsletter_version: VERSION };
async function setup(t, extra = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-antispam-'));
  let clock = Date.parse('2026-10-31T12:00:00Z'), app, server, base;
  const mails = [], logs = [];
  const config = loadConfig({ FORM_TOKEN_KEY: 'a'.repeat(64), ADMIN_PASSWORD: 'synthetic-antispam-password', DATA_FILE: path.join(dir, 'subs.json'), PUBLIC_URL: 'http://localhost:9999', TRUST_PROXY: 'true', NOTIFY_EMAIL: 'operator@example.test', SMTP_HOST: 'smtp.example.test', MAIL_FROM: 'from@example.test', SMTP_EU_APPROVED: 'true', NEWSLETTER_ENABLED: 'true', NEWSLETTER_TRACKING_DISABLED: 'true', NEWSLETTER_TOKEN_KEY: '1'.repeat(64), NEWSLETTER_WORKER_KEY: '2'.repeat(64) });
  async function start() {
    app = createApp(config, { now: () => clock, logger: { info: s => logs.push(s), warn: s => logs.push(s), error: s => logs.push(s) }, mailer: { async send(m) { mails.push(m); if (extra.uncertain) throw Error('private SMTP detail'); return true; } } });
    server = http.createServer(app); await new Promise(r => server.listen(0, '127.0.0.1', r)); base = 'http://127.0.0.1:' + server.address().port;
  }
  await start();
  const stop = async () => { app.close(); await new Promise(r => server.close(r)); };
  t.after(async () => { await stop(); fs.rmSync(dir, { recursive: true, force: true }); });
  async function challenge(ip = '192.0.2.1') { return (await fetch(base + '/api/form-token', { headers: { 'X-Real-IP': ip } })).json(); }
  async function proof(ip = '192.0.2.1', age = LIMITS.minMs) {
    const c = await challenge(ip); assert.ok(c.token); clock += age;
    return { form_token: c.token, form_work: solve(c.token) };
  }
  const post = (body, { ip = '192.0.2.1', origin = config.publicUrl, route = '/api/subscribe' } = {}) => fetch(base + route, { method: 'POST', redirect: 'manual', headers: { 'Content-Type': 'application/json', ...(origin === undefined || origin === '' ? {} : { Origin: origin }), 'X-Real-IP': ip }, body: JSON.stringify(body) });
  const link = () => new URL(mails.find(m => m.subject.startsWith('Confirmez')).text.match(/http:\/\/localhost:9999\/confirmation\?token=[\w-]+/)[0]);
  return { get app() { return app; }, get base() { return base; }, config, mails, logs, challenge, proof, post, link,
    advance(ms) { clock += ms; }, async restart() { await stop(); await start(); },
    async enroll(fields = {}, ip = '192.0.2.1') { return post({ ...valid, ...fields, ...await proof(ip) }, { ip }); },
    async confirm() { const url = link(); return post({ token: url.searchParams.get('token') }, { route: url.pathname }); },
  };
}

async function success(r) { assert.equal(r.status, 200); assert.deepEqual(await r.json(), { ok: true }); }

test('32 historical-pattern bots and hidden-field scrapers get normal success, zero records and zero email', async t => {
  const c = await setup(t), html = await (await fetch(c.base)).text();
  const hidden = Object.fromEntries([...html.matchAll(/<input type="hidden" name="([^"]+)" value="([^"]*)"/g)].map(m => [m[1], m[2]]));
  assert.ok(!hidden.form_token);
  for (let i = 0; i < 32; i++) await success(await c.post({ ...valid, ...hidden, email: 'bot' + i + '@example.test', actions: 'I wanted to know your price', site_web: '' }, { origin: i % 2 ? c.config.publicUrl : '' }));
  assert.equal(c.app.store.all().length, 0); assert.equal(c.mails.length, 0);
  assert.equal(JSON.parse(fs.readFileSync(c.config.dataFile)).subscribers.length, 0);
  assert.ok(!c.logs.join('').match(/example\.test|wanted|192\.0\.2|token|SMTP detail/));
});

test('exact Origin is mandatory on both subscription routes, including null and lookalike origins', async t => {
  const c = await setup(t);
  const fields = await c.proof();
  for (const route of ['/api/subscribe', '/inscription']) for (const origin of ['', 'null', c.config.publicUrl + '/', 'http://localhost:9999.evil.test']) await success(await c.post({ ...valid, ...fields }, { origin, route }));
  assert.equal(c.mails.length, 0); assert.equal(c.app.store.all().length, 0);
  await success(await c.post({ ...valid, ...fields }));
  assert.equal(c.mails.length, 1);
});

test('scraped tokens require age, signed payload, local work, matching IP, and one-use even after restart', async t => {
  const c = await setup(t), fields = await c.proof('192.0.2.1', 0);
  await success(await c.post({ ...valid, ...fields })); // too fast
  c.advance(LIMITS.minMs);
  await success(await c.post({ ...valid, ...fields, form_token: fields.form_token + '0' }));
  await success(await c.post({ ...valid, ...fields, form_work: 'no-work' }));
  await success(await c.post({ ...valid, ...fields }, { ip: '192.0.2.2' }));
  assert.equal(c.app.store.all().length, 0); assert.equal(c.mails.length, 0);
  await success(await c.post({ ...valid, ...fields }));
  await c.restart();
  await success(await c.post({ ...valid, ...fields, email: 'replay@example.test' }));
  assert.equal(c.app.store.all().length, 1); assert.equal(c.mails.length, 1);
  const expired = await c.proof(); c.advance(LIMITS.ttlMs);
  await success(await c.post({ ...valid, ...expired, email: 'expired@example.test' }));
  assert.equal(c.mails.length, 1);
});

test('actions accept bounded company lists and refuse multilingual spam sentences without storing them', async t => {
  for (const raw of ['AAPL, NVDA, Berkshire Hathaway, Bank of America', 'BRK.B, Mercedes-Benz, Société Générale', '', 'AAPL,\nMSFT']) assert.notEqual(validateActions(raw), null, raw);
  for (const raw of ['I wanted to know your price', 'Je voudrais connaître votre prix', 'Quería saber su precio', 'Ich wollte Ihren Preis wissen', 'Я хотел узнать вашу цену', '我想知道你们的价格', 'AAPL<script>', 'AAPL;MSFT', 'AAPL,,MSFT', 'a'.repeat(61), Array(11).fill('AAPL').join(','), 'AAPL\u0000']) assert.equal(validateActions(raw), null, raw);
  const c = await setup(t);
  const r = await c.enroll({ actions: 'I wanted to know your price' });
  assert.equal(r.status, 400); assert.match((await r.json()).detail, /tickers|noms d’entreprises/);
  assert.equal(c.mails.length, 0); assert.equal(c.app.store.all().length, 0);
});

test('per-IP rolling hourly cap persists across restarts and resets after one hour', async t => {
  const c = await setup(t);
  for (let i = 0; i < LIMITS.ipHour; i++) await c.enroll({ email: 'ip' + i + '@example.test' });
  await c.restart();
  await success(await c.enroll({ email: 'capped@example.test' }));
  assert.equal(c.mails.length, LIMITS.ipHour); assert.equal(c.app.store.all().length, LIMITS.ipHour);
  c.advance(HOUR); await c.enroll({ email: 'later@example.test' });
  assert.equal(c.mails.length, LIMITS.ipHour + 1);
});

test('distributed clients meet global hourly and rolling daily confirmation caps', async t => {
  const c = await setup(t);
  for (let i = 0; i < LIMITS.globalHour; i++) await c.enroll({ email: 'global' + i + '@example.test' }, '192.0.2.' + (i + 1));
  await success(await c.enroll({ email: 'hourcap@example.test' }, '192.0.2.100'));
  assert.equal(c.app.store.findByEmail('hourcap@example.test'), null);
  c.advance(HOUR); await c.restart();
  await success(await c.enroll({ email: 'daycap@example.test' }, '192.0.2.100'));
  assert.equal(c.app.store.findByEmail('daycap@example.test'), null); assert.equal(c.mails.length, LIMITS.mailDay);
  c.advance(DAY); await c.enroll({ email: 'tomorrow@example.test' }, '192.0.2.100');
  assert.equal(c.mails.length, LIMITS.mailDay + 1);
});

test('one message covers only requested purposes; scanners never confirm and explicit POST is single-use', async t => {
  const c = await setup(t); await c.enroll();
  assert.equal(c.mails.length, 1);
  const [sub] = c.app.store.all(), link = c.link(), token = link.searchParams.get('token');
  assert.equal(sub.status, 'pending'); assert.equal(sub.newsletter.status, 'pending');
  assert.ok(!JSON.stringify(sub).includes(token));
  for (const method of ['GET', 'HEAD']) assert.equal((await fetch(c.base + link.pathname + link.search, { method })).status, 200);
  assert.equal(c.app.store.all()[0].status, 'pending');
  assert.equal((await c.app.signup.digest()).sent, 0);
  assert.equal((await c.post({ token }, { route: '/confirmation', origin: 'https://foreign.example.test' })).status, 403);
  assert.equal((await c.confirm()).status, 200);
  assert.equal((await c.confirm()).status, 404);
  assert.equal(c.app.store.all()[0].status, 'confirmed'); assert.equal(c.app.store.all()[0].newsletter.status, 'confirmed');
  await c.enroll({}, '192.0.2.2'); assert.equal(c.mails.length, 1);
});

test('no extra email, including after uncertain SMTP, erasure and restart; ledger has only digests/counters', async t => {
  const c = await setup(t, { uncertain: true }); await c.enroll();
  const id = c.app.store.all()[0].id;
  await c.app.store.remove(id); await c.restart(); await c.enroll();
  assert.equal(c.mails.length, 1); assert.equal(c.app.store.all().length, 0);
  const state = fs.readFileSync(c.config.signupGuardFile, 'utf8');
  assert.ok(!/example\.test|192\.0\.2|AAPL|form_token/.test(state));
  assert.equal(fs.statSync(c.config.signupGuardFile).mode & 0o777, 0o600);
  assert.ok(!c.logs.join('').includes('private SMTP detail'));
});

test('pending records erase exactly at seven days, on request and startup, with no reminders', async t => {
  const c = await setup(t); await c.enroll();
  c.advance(7 * DAY - 1);
  await fetch(c.base + '/api/form-token'); assert.equal(c.app.store.all().length, 1);
  c.advance(1); await fetch(c.base + '/api/form-token'); assert.equal(c.app.store.all().length, 0);
  await c.restart(); await c.enroll(); assert.equal(c.app.store.all().length, 0); assert.equal(c.mails.length, 1);
  await c.enroll({ email: 'second@example.test' });
  c.advance(7 * DAY); await c.restart(); assert.equal(c.app.store.all().length, 0);
  assert.equal(c.mails.length, 2);
});

test('daily operator digest includes confirmed only, once per 24h and never retries uncertain delivery', async t => {
  const c = await setup(t); await c.enroll(); await c.confirm();
  await c.enroll({ email: 'pending@example.test' });
  assert.equal(c.mails.length, 2);
  assert.equal((await c.app.signup.digest()).sent, 0);
  c.advance(DAY);
  await Promise.all([c.app.signup.digest(), c.app.signup.digest()]);
  const digests = () => c.mails.filter(m => m.to === 'operator@example.test');
  assert.equal(digests().length, 1); assert.match(digests()[0].text, /person@example.test/); assert.ok(!digests()[0].text.includes('pending@example.test'));
  await c.restart(); await c.app.signup.digest(); c.advance(DAY); await c.app.signup.digest();
  assert.equal(digests().length, 1);
  // A false/ambiguous send also reserves the sole daily attempt durably.
  const d = await setup(t, { uncertain: true }); await d.enroll(); await d.confirm(); d.advance(DAY);
  await d.app.signup.digest(); await d.restart(); d.advance(DAY); await d.app.signup.digest();
  assert.equal(d.mails.filter(m => m.to === 'operator@example.test').length, 1);
});

test('disk failures fail closed before SMTP and preserve address reservations', async t => {
  const c = await setup(t), write = c.app.guard.write;
  const fields = await c.proof(); c.app.guard.write = () => { throw Error('synthetic disk full'); };
  assert.equal((await c.post({ ...valid, ...fields })).status, 500);
  assert.equal(c.mails.length, 0); assert.equal(c.app.store.all().length, 0);
  c.app.guard.write = write;
  const storeWrite = c.app.store._writeSync; c.app.store._writeSync = () => { throw Error('synthetic disk full'); };
  assert.equal((await c.enroll()).status, 500); assert.equal(c.mails.length, 0);
  c.app.store._writeSync = storeWrite; await c.restart(); await c.enroll();
  assert.equal(c.mails.length, 0); assert.equal(c.app.store.all().length, 0);
});

test('challenge mint caps and concurrent address reservations survive restart; changed key fails closed', async t => {
  const c = await setup(t);
  for (let i = 0; i < LIMITS.challengeIpHour; i++) assert.ok((await c.challenge()).token);
  await c.restart(); assert.deepEqual(await c.challenge(), {});
  assert.ok((await c.challenge('192.0.2.2')).token);
  const results = await Promise.all(Array.from({ length: 5 }, () => c.app.guard.reserveAddress('same@example.test')));
  assert.equal(results.filter(Boolean).length, 1);
  assert.throws(() => new SignupGuard(c.config.signupGuardFile, 'b'.repeat(64)), /key binding/);
  fs.chmodSync(c.config.signupGuardFile, 0o644);
  assert.throws(() => new SignupGuard(c.config.signupGuardFile, c.config.formTokenKey), /Private regular/);
  fs.chmodSync(c.config.signupGuardFile, 0o600);
  fs.symlinkSync(c.config.signupGuardFile, c.config.signupGuardFile + '.alias');
  assert.throws(() => new SignupGuard(c.config.signupGuardFile + '.alias', c.config.formTokenKey), /Private regular/);
});

test('legacy records do not gain confirmation by inference and expire as unverified', async t => {
  const c = await setup(t);
  await c.app.store.add({ id: 'legacy', email: 'legacy@example.test', created_at: '2026-10-30T12:00:00.000Z', channels: [], actions: '' });
  await c.app.signup.digest(); assert.equal(c.mails.length, 0);
  c.advance(6 * DAY); await c.restart();
  assert.equal(new SubscriberStore(c.config.dataFile).load().all().length, 0);
});

test('production requires an independent private HMAC key and external guard state is distinct', () => {
  const base = { ADMIN_PASSWORD: 'synthetic-antispam-password', NODE_ENV: 'production', PUBLIC_URL: 'https://example.test' };
  assert.throws(() => loadConfig(base), /FORM_TOKEN_KEY/);
  assert.throws(() => loadConfig({ ...base, FORM_TOKEN_KEY: 'a'.repeat(64), NEWSLETTER_TOKEN_KEY: 'a'.repeat(64) }), /independent/);
  assert.throws(() => loadConfig({ ...base, FORM_TOKEN_KEY: 'a'.repeat(64), DATA_FILE: 'same.json', SIGNUP_GUARD_FILE: 'same.json' }), /Separate/);
});

test('SMTP kill switch gives a human a clear message and reserves/stores nothing', async t => {
  const c = await setup(t); c.config.smtp.euApproved = false;
  const r = await c.enroll(); assert.equal(r.status, 503);
  assert.match((await r.json()).detail, /suspendu/);
  assert.equal(c.app.store.all().length, 0); assert.equal(c.mails.length, 0);
  assert.equal(Object.keys(c.app.guard.state.addresses).length, 0);
  c.config.smtp.euApproved = true;
  await c.enroll(); assert.equal(c.mails.length, 1);
});
