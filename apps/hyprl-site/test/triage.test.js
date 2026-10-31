'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path'), http = require('node:http'), crypto = require('node:crypto');
const { loadConfig } = require('../server/config');
const { createApp } = require('../server/app');
const { CURRENT_VERSION, CONSENT_TEXTS } = require('../server/lib/consent');
const { LIMITS } = require('../server/lib/signup-guard');
const { solve } = require('./signup-client');
const DAY = 86400000;
const valid = { email: 'person@example.test', consent: 'oui', consent_version: CURRENT_VERSION, channels: ['discord', 'email'], actions: 'AAPL' };
async function setup(t, legacy = [], uncertain = false) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-triage-'));
  const config = loadConfig({ FORM_TOKEN_KEY: 'a'.repeat(64), ADMIN_PASSWORD: 'synthetic-triage-password', DATA_FILE: path.join(dir, 'subscribers.json'), PUBLIC_URL: 'http://localhost:9999', SMTP_HOST: 'smtp.example.test', MAIL_FROM: 'from@example.test', SMTP_EU_APPROVED: 'true' });
  fs.writeFileSync(config.dataFile, JSON.stringify({ subscribers: legacy }), { mode: 0o600 });
  let clock = Date.parse('2026-10-31T12:00:00Z'), app, server, base, cookie;
  const mails = [], logs = [];
  async function start() {
    app = createApp(config, { now: () => clock, logger: { info: s => logs.push(s), warn() {}, error: s => logs.push(s) }, mailer: { async send(m) { mails.push(m); if (uncertain) throw Error('private SMTP'); return true; } } });
    server = http.createServer(app); await new Promise(r => server.listen(0, '127.0.0.1', r)); base = 'http://127.0.0.1:' + server.address().port;
    cookie = (await request('/api/admin/login', 'POST', { password: config.adminPassword })).headers.get('set-cookie').split(';')[0];
  }
  function request(route, method = 'GET', body, auth = false, origin = config.publicUrl) {
    return fetch(base + route, { method, redirect: 'manual', headers: { 'Content-Type': 'application/json', Origin: origin, ...(auth ? { Cookie: cookie } : {}) }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  }
  const stop = async () => { app.close(); await new Promise(r => server.close(r)); };
  await start();
  t.after(async () => { await stop(); fs.rmSync(dir, { recursive: true, force: true }); });
  return { config, mails, logs, request, get app() { return app; }, advance(ms) { clock += ms; },
    async restart() { await stop(); await start(); },
    async proof(age = LIMITS.minMs) { const c = await (await request('/api/form-token')).json(); clock += age; return { form_token: c.token, form_work: solve(c.token) }; },
    async submit(fields = {}, proved = true) { return request('/api/subscribe', 'POST', { ...valid, ...fields, ...(proved ? await this.proof() : {}) }); },
    async list(tab) { const r = await request('/api/admin/subscribers?tab=' + tab, 'GET', undefined, true); assert.equal(r.status, 200); return r.json(); },
    promote(id, auth = true, origin = config.publicUrl) { return request('/api/admin/suspects/' + id + '/promote', 'POST', {}, auth, origin); },
  };
}

function legacy(email, actions = 'AAPL') {
  return { id: crypto.randomUUID(), email, channels: ['discord'], actions, created_at: '2026-09-01T12:00:00.000Z',
    consent: { given: true, version: '2026-10-v1', text: CONSENT_TEXTS['2026-10-v1'], checked_at: '2026-09-01T12:00:00.000Z' } };
}

test('tabs strictly separate confirmed, pending and quarantined, with counts and no bearer/hash exposure', async t => {
  const c = await setup(t);
  await c.submit({ email: 'bot@example.test', actions: 'I wanted to know your price' }, false);
  await c.submit({ email: 'pending@example.test' });
  await c.submit({ email: 'real@example.test' });
  const token = c.mails[1].text.match(/\/confirmation\?token=([\w-]+)/)[1];
  await c.request('/confirmation', 'POST', { token });
  for (const [tab, email] of [['confirmed', 'real@example.test'], ['pending', 'pending@example.test'], ['suspect', 'bot@example.test']]) {
    const result = await c.list(tab);
    assert.deepEqual(result.counts, { confirmed: 1, pending: 1, suspect: 1 });
    assert.deepEqual(result.subscribers.map(s => s.email), [email]);
    assert.ok(!JSON.stringify(result).includes('hash'));
  }
  assert.equal((await c.request('/api/admin/subscribers?tab=suspect')).status, 401);
  assert.equal((await c.request('/api/admin/subscribers?tab=other', 'GET', undefined, true)).status, 400);
  const csv = await c.request('/api/admin/subscribers/export.csv', 'GET', undefined, true);
  assert.equal(csv.status, 200); const text = await csv.text();
  assert.match(text, /real@example.test/); assert.doesNotMatch(text, /bot@example|pending@example/);
  assert.equal((await c.request('/api/admin/subscribers/export.csv')).status, 401);
  const exportNewsletter = await (await c.request('/api/admin/newsletter/export', 'GET', undefined, true)).text();
  assert.doesNotMatch(exportNewsletter, /bot@example|pending@example/);
  const q = c.app.triage.quarantine.all()[0];
  assert.equal((await c.request('/api/admin/subscribers/' + q.id + '/export', 'GET', undefined, true)).status, 404);
});

test('rejections show ordinary success, explain no-token/fast/sentence/cap in French, and send no email', async t => {
  const c = await setup(t);
  assert.deepEqual(await (await c.submit({ email: 'missing@example.test' }, false)).json(), { ok: true });
  const fast = await c.proof(0);
  await c.request('/api/subscribe', 'POST', { ...valid, email: 'fast@example.test', ...fast });
  await c.submit({ email: 'sentence@example.test', actions: 'I wanted to know your price' });
  await c.app.guard.mutate((s, at) => { s.attempts = Array.from({ length: LIMITS.globalHour }, () => ({ at, ip: 'synthetic' })); });
  await c.submit({ email: 'cap@example.test' });
  const rows = (await c.list('suspect')).subscribers;
  assert.deepEqual(new Set(rows.map(s => s.reason_code)), new Set(['no_token', 'too_fast', 'actions', 'cap']));
  assert.ok(rows.every(s => /jeton|vite|Phrase|Plafond/.test(s.reason)));
  assert.equal(c.mails.length, 0); assert.equal(c.app.store.all().length, 0);
  const state = fs.readFileSync(c.config.quarantineFile, 'utf8');
  assert.doesNotMatch(state, /form_token|form_work|192\.0\.2|confirmation_hash/);
  assert.equal(fs.statSync(c.config.quarantineFile).mode & 0o777, 0o600);
});

test('seven-day quarantine deadline is fixed across duplicates/restart, request purge, and startup purge', async t => {
  const c = await setup(t); await c.submit({}, false);
  const q = c.app.triage.quarantine.all()[0];
  c.advance(7 * DAY - 1); await c.submit({}, false); await c.restart();
  assert.equal(c.app.triage.quarantine.all()[0].expires_at, q.expires_at);
  assert.equal((await c.list('suspect')).counts.suspect, 1);
  c.advance(1); assert.equal((await c.list('suspect')).counts.suspect, 0);
  assert.deepEqual(JSON.parse(fs.readFileSync(c.config.quarantineFile)).subscribers, []);
  await c.submit({ email: 'second@example.test' }, false); c.advance(7 * DAY); await c.restart();
  assert.equal(c.app.triage.quarantine.all().length, 0); assert.equal(c.mails.length, 0);
});

test('quarantine expiry timer erases without a visitor request', async t => {
  const c = await setup(t); await c.submit({}, false);
  const q = c.app.triage.quarantine.all()[0];
  await c.app.triage.quarantine.update(q.id, s => { s.expires_at = new Date(Date.parse(s.quarantined_at) + 30).toISOString(); return s; });
  await c.app.triage.run(); c.advance(30);
  await new Promise(r => setTimeout(r, 80));
  assert.equal(c.app.triage.quarantine.all().length, 0);
  assert.equal(JSON.parse(fs.readFileSync(c.config.quarantineFile)).subscribers.length, 0);
});

test('first startup imports old bot pattern before retention, preserves identity/date/reason, and never reimports it', async t => {
  const bot = legacy('oldbot@example.test', 'I wanted to know your price'), other = legacy('oldperson@example.test');
  const c = await setup(t, [bot, other]);
  assert.equal(c.app.store.all().length, 0);
  const rows = (await c.list('suspect')).subscribers;
  assert.equal(rows.length, 2); assert.equal(rows.find(s => s.id === bot.id).reason_code, 'actions');
  assert.equal(rows.find(s => s.id === other.id).reason_code, 'legacy');
  assert.equal(rows.find(s => s.id === bot.id).created_at, bot.created_at);
  const expiry = rows[0].expires_at;
  c.advance(DAY); await c.restart(); assert.equal(c.app.triage.quarantine.all()[0].expires_at, expiry);
  assert.equal(c.mails.length, 0);
});

test('manual move sends exactly one confirmation, logs review, preserves text, and never grants newsletter or confirmation', async t => {
  const c = await setup(t, [legacy('reviewed@example.test', 'I wanted to know your price')]);
  const id = c.app.triage.quarantine.all()[0].id;
  assert.equal((await c.promote(id, false)).status, 401);
  assert.equal((await c.promote(id, true, 'https://foreign.example.test')).status, 403);
  const results = await Promise.all([c.promote(id), c.promote(id)]);
  assert.deepEqual(results.map(r => r.status).sort(), [200, 404]);
  assert.equal((await c.list('suspect')).counts.suspect, 0);
  const [person] = (await c.list('pending')).subscribers;
  assert.equal(person.status, 'pending'); assert.equal(person.confirmed_at, null);
  assert.equal(person.actions, 'I wanted to know your price'); assert.equal(person.newsletter.status, 'not_requested');
  assert.ok(person.triage_reviewed_at); assert.equal(person.confirmation_delivery, 'sent');
  assert.equal(c.mails.length, 1); assert.ok(c.logs.some(s => s.includes('promotion_confirmation_reserved')));
  await c.restart(); assert.equal(c.app.store.all().length, 1); assert.equal(c.app.triage.quarantine.all().length, 0);
  assert.equal((await c.promote(id)).status, 404); assert.equal(c.mails.length, 1);
  const token = c.mails[0].text.match(/\/confirmation\?token=([\w-]+)/)[1];
  assert.equal((await c.request('/confirmation', 'POST', { token })).status, 200);
  assert.equal((await c.list('confirmed')).counts.confirmed, 1);
});

test('promotion respects SMTP kill switch, consent, mail cap and durable suppression after ambiguous SMTP', async t => {
  const c = await setup(t); await c.submit({}, false);
  const id = c.app.triage.quarantine.all()[0].id;
  c.config.smtp.euApproved = false; assert.equal((await c.promote(id)).status, 503); assert.equal(c.mails.length, 0);
  c.config.smtp.euApproved = true;
  await c.app.guard.reserveAddress(valid.email);
  assert.equal((await c.promote(id)).status, 409); assert.equal(c.app.triage.quarantine.all().length, 1);
  const d = await setup(t, [], true); await d.submit({}, false); const q = d.app.triage.quarantine.all()[0];
  assert.equal((await d.promote(q.id)).status, 200); assert.equal(d.mails.length, 1);
  assert.equal(d.app.store.all()[0].confirmation_delivery, 'uncertain');
  await d.app.store.remove(d.app.store.all()[0].id); await d.restart(); await d.submit({}, false);
  assert.equal((await d.promote(d.app.triage.quarantine.all()[0].id)).status, 409); assert.equal(d.mails.length, 1);
  const e = await setup(t); await e.submit({ consent: 'non' }, false);
  assert.equal((await e.promote(e.app.triage.quarantine.all()[0].id)).status, 409); assert.equal(e.mails.length, 0);
});

test('suspect deletion is authenticated, origin-protected, permanent and sends no email', async t => {
  const c = await setup(t); await c.submit({}, false); const id = c.app.triage.quarantine.all()[0].id;
  const route = '/api/admin/suspects/' + id;
  assert.equal((await c.request(route, 'DELETE')).status, 401);
  assert.equal((await c.request(route, 'DELETE', undefined, true, 'https://foreign.example.test')).status, 403);
  assert.equal((await c.request(route, 'DELETE', undefined, true)).status, 204);
  await c.restart(); assert.equal((await c.list('suspect')).counts.suspect, 0); assert.equal(c.mails.length, 0);
});

test('private quarantine rejects public/symlink/aliased state and a disk failure cannot trigger email', async t => {
  const c = await setup(t); await c.submit({}, false);
  const write = c.app.triage.quarantine._writeSync;
  c.app.triage.quarantine._writeSync = () => { throw Error('synthetic disk failure'); };
  assert.equal((await c.submit({ email: 'disk@example.test' }, false)).status, 500); assert.equal(c.mails.length, 0);
  c.app.triage.quarantine._writeSync = write;
  assert.throws(() => loadConfig({ ADMIN_PASSWORD: c.config.adminPassword, DATA_FILE: c.config.dataFile, SIGNUP_QUARANTINE_FILE: c.config.dataFile }), /Separate/);
  const { QuarantineStore } = require('../server/lib/triage');
  fs.chmodSync(c.config.quarantineFile, 0o644);
  assert.throws(() => new QuarantineStore(c.config.quarantineFile).load(), /Private regular/);
  fs.chmodSync(c.config.quarantineFile, 0o600);
  fs.symlinkSync(c.config.quarantineFile, c.config.quarantineFile + '.alias');
  assert.throws(() => new QuarantineStore(c.config.quarantineFile + '.alias').load(), /Private regular/);
});

test('interrupted import and promotion recover by identity without renewal, duplication or email retry', async t => {
  const bot = legacy('crash-import@example.test', 'I wanted to know your price');
  const c = await setup(t, [bot]);
  const q = c.app.triage.quarantine.all()[0];
  // Simulate interruption after quarantine copy, before source removal.
  await c.app.store.add(bot); c.advance(DAY); await c.restart();
  assert.equal(c.app.store.all().length, 0); assert.equal(c.app.triage.quarantine.all().length, 1);
  assert.equal(c.app.triage.quarantine.all()[0].expires_at, q.expires_at);
  // Simulate interruption after pending record persisted, before quarantine removal/SMTP.
  const remove = c.app.triage.quarantine.remove;
  c.app.triage.quarantine.remove = async () => { throw Error('synthetic interruption'); };
  assert.equal((await c.promote(q.id)).status, 500); assert.equal(c.mails.length, 0);
  assert.equal(c.app.store.all().length, 1);
  c.app.triage.quarantine.remove = remove;
  await c.restart();
  assert.equal(c.app.triage.quarantine.all().length, 0); assert.equal(c.app.store.all().length, 1);
  assert.equal((await c.promote(q.id)).status, 404); assert.equal(c.mails.length, 0);
});

test('confirmed-only CSV neutralizes spreadsheet formulas and quotes multi-line cells', async t => {
  const c = await setup(t);
  await c.app.store.add({ ...legacy('=synthetic@example.test', '=HYPERLINK("synthetic")\nAAPL'), status: 'confirmed', confirmed_at: '2026-10-31T12:00:00.000Z' });
  const text = await (await c.request('/api/admin/subscribers/export.csv', 'GET', undefined, true)).text();
  assert.match(text, /"'=synthetic@example\.test"/); assert.match(text, /"'=HYPERLINK\(""synthetic""\)\nAAPL"/);
});

test('full subscriber erasure also erases any earlier quarantined request for the same address', async t => {
  const c = await setup(t); await c.submit({}, false); await c.submit();
  assert.equal(c.app.triage.quarantine.all().length, 1); assert.equal(c.app.store.all().length, 1);
  const id = c.app.store.all()[0].id;
  assert.equal((await c.request('/api/admin/subscribers/' + id, 'DELETE', undefined, true)).status, 204);
  assert.equal(c.app.triage.quarantine.all().length, 0); assert.equal(c.app.store.all().length, 0);
  await c.restart(); assert.equal(c.app.triage.quarantine.all().length, 0);
});

test('truncating an overlong quarantine address never makes it eligible for confirmation', async t => {
  const c = await setup(t);
  await c.submit({ email: 'a'.repeat(245) + '@example.test' }, false);
  const q = c.app.triage.quarantine.all()[0];
  assert.equal(q.email.length, 254); assert.equal(q.email_valid, false);
  assert.equal((await c.promote(q.id)).status, 409); assert.equal(c.mails.length, 0);
});
