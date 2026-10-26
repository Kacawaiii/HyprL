'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { loadConfig } = require('../server/config');
const { createApp } = require('../server/app');
const { SubscriberStore } = require('../server/lib/store');
const { expiresAt, startRetentionJob } = require('../server/lib/retention');
const { createMailer } = require('../server/lib/mail');
const consent = require('../public/consent');
const { CURRENT_VERSION } = require('../server/lib/consent');
const password = 'synthetic-password-for-offline-tests';
const quiet = { info() {}, warn() {}, error() {} };

async function setup(t, extra = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-compliance-'));
  let clock = Date.parse('2026-10-26T12:00:00Z');
  const config = loadConfig({ ADMIN_PASSWORD: password, DATA_FILE: path.join(dir, 'subscribers.json'), PUBLIC_URL: 'https://hyprlcore.com', NODE_ENV: 'production', ...extra });
  const app = createApp(config, { now: () => clock, mailer: { async send() {} }, logger: quiet });
  const server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = 'http://127.0.0.1:' + server.address().port;
  t.after(async () => { app.close(); await new Promise((resolve) => server.close(resolve)); fs.rmSync(dir, { recursive: true, force: true }); });
  async function login() {
    const r = await fetch(base + '/api/admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password }) });
    assert.equal(r.status, 200);
    return r.headers.get('set-cookie').split(';')[0];
  }
  async function subscribe(email = 'person@example.test') {
    const r = await fetch(base + '/api/subscribe', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, consent: 'oui', consent_version: CURRENT_VERSION, channels: ['email'], actions: 'SYNTH' }) });
    assert.ok([200, 201].includes(r.status));
    return app.store.findByEmail(email);
  }
  return { app, config, base, login, subscribe, setClock: (date) => { clock = Date.parse(date); } };
}

test('notification has no default; SMTP is gated by an explicit EU/DPA approval', async () => {
  const config = loadConfig({ ADMIN_PASSWORD: password, SMTP_HOST: 'smtp.invalid', MAIL_FROM: 'sender@example.test' });
  assert.equal(config.notifyEmail, '');
  assert.equal(config.smtp.euApproved, false);
  assert.equal(await createMailer(config, quiet).send({ subject: 'synthetic' }), false);
  assert.equal(loadConfig({ ADMIN_PASSWORD: password, SMTP_EU_APPROVED: 'true' }).smtp.euApproved, true);
});

test('24 calendar months, UTC, leap-day clamping and invalid timestamp rejection', () => {
  assert.equal(expiresAt('2024-02-29T12:34:56.789Z'), '2026-02-28T12:34:56.789Z');
  assert.equal(expiresAt('2026-10-31T00:00:00Z'), '2028-10-31T00:00:00.000Z');
  for (const value of ['2024-02-30T00:00:00Z', '2024-01-01T12:00:00+01:00', 'bad', undefined]) assert.throws(() => expiresAt(value));
});

test('hourly retention is scheduled; startup purges; failures are observable without data leakage', async () => {
  let task, interval, canceled, startup = 0;
  const messages = [];
  const job = startRetentionJob({ purgeExpiredSync: () => startup++, purgeExpired: async () => { throw new Error('private-person@example.test'); } }, {
    now: () => 123, schedule: (fn, ms) => { task = fn; interval = ms; return 7; }, cancel: (id) => { canceled = id; }, logger: { error: (m) => messages.push(m) }
  });
  assert.equal(startup, 1);
  assert.equal(interval, 3600000);
  task();
  await new Promise(setImmediate);
  assert.equal(messages.length, 1);
  assert.ok(!messages[0].includes('example.test'));
  job.close(); assert.equal(canceled, 7);
  assert.throws(() => startRetentionJob({ purgeExpiredSync() { throw new Error('disk'); } }), /disk/);
});

test('purge survives restart and a disk failure never partially mutates memory', async (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-retention-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'store.json');
  const store = new SubscriberStore(file).load();
  await store.add({ id: 'old', email: 'old@example.test', created_at: '2024-02-29T12:00:00Z' });
  await store.add({ id: 'new', email: 'new@example.test', created_at: '2025-01-01T00:00:00Z' });
  const write = store._writeSync;
  store._writeSync = () => { throw new Error('disk full'); };
  await assert.rejects(store.purgeExpired(Date.parse('2026-02-28T12:00:00Z')), /disk full/);
  assert.equal(store.all().length, 2);
  assert.equal(new SubscriberStore(file).load().all().length, 2);
  store._writeSync = write;
  assert.equal(await store.purgeExpired(Date.parse('2026-02-28T12:00:00Z')), 1);
  assert.deepEqual(new SubscriberStore(file).load().all().map((s) => s.id), ['new']);
});

test('individual export is authenticated, isolated, token-free, versioned and UTC', async (t) => {
  const s = await setup(t);
  const person = await s.subscribe();
  await s.subscribe('other@example.test');
  const url = s.base + '/api/admin/subscribers/' + person.id + '/export';
  assert.equal((await fetch(url)).status, 401);
  const cookie = await s.login();
  const r = await fetch(url, { headers: { Cookie: cookie } });
  assert.equal(r.status, 200);
  assert.equal(r.headers.get('cache-control'), 'no-store');
  assert.match(r.headers.get('content-disposition'), /attachment/);
  const body = await r.json();
  assert.equal(body.schema_version, '1.0.0');
  assert.equal(body.exported_at, '2026-10-26T12:00:00.000Z');
  assert.equal(body.retention_expires_at, '2028-10-26T12:00:00.000Z');
  assert.equal(body.subscriber.email, 'person@example.test');
  assert.equal(body.subscriber.consent.checked_at, body.subscriber.created_at);
  assert.ok(!JSON.stringify(body).includes('token'));
  assert.ok(!JSON.stringify(body).includes('other@example.test'));
  assert.equal((await fetch(url, { method: 'POST', headers: { Cookie: cookie } })).status, 405);
  assert.equal((await fetch(s.base + '/api/admin/subscribers/00000000-0000-4000-8000-000000000000/export', { headers: { Cookie: cookie } })).status, 404);
});

test('request-time retention enforces the exact expiry and does not renew on duplicates', async (t) => {
  const s = await setup(t);
  const person = await s.subscribe();
  s.setClock('2028-10-26T11:59:59.999Z');
  const cookie = await s.login();
  const url = s.base + '/api/admin/subscribers/' + person.id + '/export';
  assert.equal((await fetch(url, { headers: { Cookie: cookie } })).status, 200);
  await s.subscribe();
  assert.equal(s.app.store.all()[0].created_at, '2026-10-26T12:00:00.000Z');
  s.setClock('2028-10-26T12:00:00Z');
  assert.equal((await fetch(url, { headers: { Cookie: cookie } })).status, 404);
  assert.equal(s.app.store.all().length, 0);
  assert.equal(new SubscriberStore(s.config.dataFile).load().all().length, 0);
});

test('parallel mutations retain independent signups, deduplicate and do not resurrect deletion', async (t) => {
  const s = await setup(t);
  await Promise.all([s.subscribe('a@example.test'), s.subscribe('b@example.test'), s.subscribe('a@example.test')]);
  assert.equal(s.app.store.all().length, 2);
  const id = s.app.store.findByEmail('a@example.test').id;
  await Promise.all([s.app.store.remove(id), s.app.store.purgeExpired(Date.parse('2026-10-26T12:00:00Z'))]);
  assert.deepEqual(s.app.store.all().map((p) => p.email), ['b@example.test']);
});

test('API failures use RFC 9457, including malformed JSON, method, size and unknown route', async (t) => {
  const s = await setup(t);
  const cases = [
    ['/api/v1/alerts', {}, 404], ['/api/subscribe', {}, 405],
    ['/api/subscribe', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: '' }, 400],
    ['/api/admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{' }, 400],
    ['/api/subscribe', { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: 'no' }, 415],
    ['/api/subscribe', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: 'a'.repeat(17000) }, 413],
    ['/api/subscribe', { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'https://foreign.invalid' }, body: '{}' }, 403]
  ];
  for (const [url, opts, status] of cases) {
    const r = await fetch(s.base + url, opts);
    assert.equal(r.status, status);
    assert.match(r.headers.get('content-type'), /^application\/problem\+json/);
    const p = await r.json();
    assert.equal(p.status, status); assert.equal(p.type, 'about:blank');
    assert.equal(typeof p.detail, 'string'); assert.equal(typeof p.title, 'string');
  }
  const formError = await fetch(s.base + '/inscription', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: '' });
  assert.equal(formError.status, 400);
  assert.match(formError.headers.get('content-type'), /^text\/html/);
  assert.match(await formError.text(), /Consentement requis/);
});

test('public resources never set visitor cookies; HSTS, discovery and env-only contact', async (t) => {
  const s = await setup(t, { NOTIFY_EMAIL: 'private-contact@example.test' });
  for (const route of ['/', '/confidentialite', '/accessibilite', '/contact', '/resiliation', '/securite', '/.well-known/security.txt', '/sitemap.xml', '/robots.txt', '/consent.js']) {
    const r = await fetch(s.base + route);
    assert.equal(r.status, 200, route);
    assert.equal(r.headers.get('set-cookie'), null, route);
    assert.match(r.headers.get('strict-transport-security'), /max-age=31536000/);
  }
  for (const route of ['/docs/internal/REGISTRE_TRAITEMENTS.md', '/server/config.js', '/test/browser-server.js', '/.env', '/package.json']) {
    assert.equal((await fetch(s.base + route)).status, 404, route);
  }
  const txt = await (await fetch(s.base + '/.well-known/security.txt')).text();
  assert.match(txt, /^Contact: https:\/\/hyprlcore.com\/contact\n/);
  assert.match(txt, /Canonical: https:\/\/hyprlcore.com\/\.well-known\/security.txt/);
  const expiration = Date.parse(txt.match(/Expires: (.+)/)[1]);
  assert.equal(expiration - Date.parse('2026-10-26T12:00:00Z'), 180 * 86400000);
  const contact = await (await fetch(s.base + '/contact')).text();
  assert.match(contact, /mailto:private-contact@example.test/);
  const robots = await (await fetch(s.base + '/robots.txt')).text();
  assert.match(robots, /Sitemap: https:\/\/hyprlcore.com\/sitemap.xml/);
  const sitemap = await (await fetch(s.base + '/sitemap.xml')).text();
  assert.match(sitemap, /\/accessibilite/); assert.ok(!sitemap.includes('/admin'));
  const login = await fetch(s.base + '/api/admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password }) });
  assert.match(login.headers.get('set-cookie'), /Secure/);
});

function memoryStorage() {
  const values = new Map();
  return { getItem: (key) => values.get(key) || null, setItem: (key, value) => values.set(key, value), removeItem: (key) => values.delete(key) };
}

test('disabled consent has no storage, banner, or optional-script effects', () => {
  assert.equal(consent.ENABLED, false);
  const storage = new Proxy({}, { get() { throw new Error('storage access forbidden'); } });
  const engine = consent.create({ storage, scripts: [{ purpose: 'functional', load() { assert.fail('script loaded'); } }] });
  assert.equal(engine.current(), null); assert.equal(engine.choose({ functional: true }), null);
  engine.loadAllowed(); assert.equal(consent.mount(null, storage, []), null);
});

test('optional scripts require purpose-specific opt-in; refusal persists equally; revoke stops execution', () => {
  const storage = memoryStorage();
  let functional = 0, analytics = 0, revoked = 0;
  const scripts = [{ purpose: 'functional', load: () => functional++ }, { purpose: 'analytics', load: () => analytics++ }];
  const engine = consent.create({ enabled: true, storage, scripts, onRevoke: () => revoked++ });
  engine.loadAllowed(); assert.equal(functional + analytics, 0);
  engine.choose({ functional: true }); assert.equal(functional, 1); assert.equal(analytics, 0);
  engine.loadAllowed(); assert.equal(functional, 1);
  engine.choose({}); assert.equal(revoked, 1);
  const reload = consent.create({ enabled: true, storage, scripts });
  assert.deepEqual(reload.current().purposes, { functional: false, analytics: false });
  reload.loadAllowed(); assert.equal(analytics, 0);
});

test('six calendar months expiry, 13-month backstop and corrupt/future storage fail closed', () => {
  const storage = memoryStorage();
  let now = Date.parse('2026-08-31T12:00:00Z');
  const engine = consent.create({ enabled: true, storage, now: () => now });
  engine.choose({ analytics: true });
  assert.equal(engine.current().expires_at, '2027-02-28T12:00:00.000Z');
  now = Date.parse('2027-02-28T12:00:00Z');
  assert.equal(engine.current(), null); assert.equal(storage.getItem(consent.KEY), null);
  for (const value of ['{', JSON.stringify({ version: 1, decided_at: '2028-01-01T00:00:00Z', expires_at: '2028-07-01T00:00:00.000Z', purposes: { functional: true, analytics: true } }), JSON.stringify({ version: 1, decided_at: '2025-01-01T00:00:00Z', expires_at: '2099-01-01T00:00:00Z', purposes: { functional: true, analytics: true } })]) {
    storage.setItem(consent.KEY, value);
    assert.equal(consent.create({ enabled: true, storage, now: () => now }).current(), null);
  }
});
