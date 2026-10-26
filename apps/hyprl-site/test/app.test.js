'use strict';
// Tests de bout en bout : node --test test/

const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { loadConfig } = require('../server/config');
const { createApp } = require('../server/app');
const { CONSENT_TEXTS, CURRENT_VERSION } = require('../server/lib/consent');

const PASSWORD = 'mot-de-passe-de-test-tres-long-0123';

async function setup() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-test-'));
  let clock = Date.parse('2026-10-26T13:05:00Z');
  const mails = [];
  const logs = [];
  const config = loadConfig({ ADMIN_PASSWORD: PASSWORD, NOTIFY_EMAIL: 'notification@example.test', DATA_FILE: path.join(dir, 'subs.json'), PUBLIC_URL: 'http://localhost:9999' });
  const app = createApp(config, {
    now: () => clock,
    mailer: { send: async (m) => { mails.push(m); return true; } },
    logger: { info: (s) => logs.push(s), warn: (s) => logs.push(s), error: (s) => logs.push(s) },
  });
  const server = http.createServer(app);
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const base = 'http://127.0.0.1:' + server.address().port;
  return {
    base, mails, logs, app, config, dir,
    advance: (ms) => { clock += ms; },
    close: () => new Promise((r) => { app.close(); server.close(() => { fs.rmSync(dir, { recursive: true, force: true }); r(); }); }),
  };
}

function form(fields) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(fields)) [].concat(v).forEach((x) => p.append(k, x));
  return p.toString();
}

const valid = {
  email: 'subscriber1@example.test', channels: ['discord', 'email', 'inconnu'], actions: 'AAPL, NVDA,\n BTC',
  consent: 'oui', consent_version: CURRENT_VERSION,
};

const post = (base, p, body, headers = {}) => fetch(base + p, {
  method: 'POST', redirect: 'manual', body,
  headers: { 'Content-Type': 'application/x-www-form-urlencoded', ...headers },
});
const tick = () => new Promise((r) => setImmediate(r));

async function loginCookie(t) {
  const r = await fetch(t.base + '/api/admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password: PASSWORD }) });
  assert.equal(r.status, 200);
  return r.headers.get('set-cookie').split(';')[0];
}

test('le texte de consentement affiché correspond exactement à celui enregistré', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'public', 'index.html'), 'utf8');
  const shown = html.match(/<span id="consent-text">([\s\S]*?)<\/span>/)[1];
  assert.equal(shown, CONSENT_TEXTS[CURRENT_VERSION]);
});

test('sans case de consentement cochée, rien n’est enregistré', async () => {
  const t = await setup();
  try {
    const { consent, ...noConsent } = valid;
    const r = await post(t.base, '/api/subscribe', form(noConsent));
    assert.equal(r.status, 400);
    const r2 = await post(t.base, '/api/subscribe', form({ ...valid, consent: 'non' }));
    assert.equal(r2.status, 400);
    assert.equal(t.app.store.all().length, 0);
    const disk = JSON.parse(fs.readFileSync(t.config.dataFile, 'utf8'));
    assert.equal(disk.subscribers.length, 0);
    assert.equal(t.mails.length, 0);
  } finally { await t.close(); }
});

test('inscription : date, heure, texte exact du consentement, e-mails', async () => {
  const t = await setup();
  try {
    const r = await post(t.base, '/api/subscribe', form(valid));
    assert.equal(r.status, 303);
    assert.equal(r.headers.get('location'), '/merci');
    const [s] = t.app.store.all();
    assert.equal(s.email, 'subscriber1@example.test');
    assert.deepEqual(s.channels, ['discord', 'email']);
    assert.equal(s.actions, 'AAPL, NVDA, BTC');
    assert.equal(s.created_at, '2026-10-26T13:05:00.000Z');
    assert.deepEqual(s.consent, { given: true, version: CURRENT_VERSION, text: CONSENT_TEXTS[CURRENT_VERSION], checked_at: '2026-10-26T13:05:00.000Z' });
    assert.ok(/^[0-9a-f]{64}$/.test(s.unsubscribe_token_hash));

    const disk = JSON.parse(fs.readFileSync(t.config.dataFile, 'utf8'));
    assert.equal(disk.subscribers[0].consent.text, CONSENT_TEXTS[CURRENT_VERSION]);
    if (process.platform !== 'win32') assert.equal(fs.statSync(t.config.dataFile).mode & 0o777, 0o600);

    await tick();
    const notif = t.mails.find((m) => m.to === 'notification@example.test');
    assert.ok(notif, 'notification envoyée à notification@example.test');
    assert.equal(notif.text,
      'Date : 26/10/2026 14:05:00 (heure de Paris)\n' +
      'Adresse : subscriber1@example.test\n' +
      'Canaux : Discord, E-mail\n' +
      'Actions suivies : AAPL, NVDA, BTC\n');
    const conf = t.mails.find((m) => m.to === 'subscriber1@example.test');
    assert.match(conf.text, /http:\/\/localhost:9999\/desinscription\?token=[A-Za-z0-9_-]{43}/);
    assert.equal(conf.headers['List-Unsubscribe-Post'], 'List-Unsubscribe=One-Click');
    assert.ok(t.logs.every((l) => !l.includes('subscriber1@example.test')), 'aucune adresse dans les journaux');
  } finally { await t.close(); }
});

test('doublon et pot de miel : aucune nouvelle donnée', async () => {
  const t = await setup();
  try {
    await post(t.base, '/api/subscribe', form(valid));
    await tick();
    const before = t.mails.length;
    const r = await post(t.base, '/api/subscribe', form({ ...valid, actions: 'TSLA' }));
    assert.equal(r.status, 303);
    const r2 = await post(t.base, '/api/subscribe', form({ ...valid, email: 'subscriber2@example.test', site_web: 'http://x' }));
    assert.equal(r2.status, 303);
    await tick();
    assert.equal(t.app.store.all().length, 1);
    assert.equal(t.app.store.all()[0].actions, 'AAPL, NVDA, BTC');
    assert.equal(t.mails.length, before);
  } finally { await t.close(); }
});

test('désinscription : GET demande confirmation, POST efface', async () => {
  const t = await setup();
  try {
    await post(t.base, '/api/subscribe', form(valid));
    await tick();
    const link = t.mails.find((m) => m.to === 'subscriber1@example.test').text.match(/http:\/\/localhost:9999(\/desinscription\?token=\S+)/)[1];
    const g = await fetch(t.base + link);
    assert.equal(g.status, 200);
    assert.match(await g.text(), /Confirmer la désinscription/);
    assert.equal(t.app.store.all().length, 1, 'un GET n’efface rien');

    const token = new URL('http://x' + link).searchParams.get('token');
    const p = await post(t.base, '/desinscription', form({ token }));
    assert.equal(p.status, 200);
    assert.equal(t.app.store.all().length, 0);
    assert.equal(JSON.parse(fs.readFileSync(t.config.dataFile, 'utf8')).subscribers.length, 0);
    const again = await post(t.base, '/desinscription', form({ token }));
    assert.equal(again.status, 404);
  } finally { await t.close(); }
});

test('désinscription en un clic (RFC 8058)', async () => {
  const t = await setup();
  try {
    await post(t.base, '/api/subscribe', form(valid));
    await tick();
    const link = t.mails.find((m) => m.to === 'subscriber1@example.test').list.unsubscribe.url.replace('http://localhost:9999', '');
    const p = await post(t.base, link, 'List-Unsubscribe=One-Click');
    assert.equal(p.status, 200);
    assert.equal(t.app.store.all().length, 0);
  } finally { await t.close(); }
});

test('admin : 5 tentatives par 15 minutes', async () => {
  const t = await setup();
  try {
    const bad = () => fetch(t.base + '/api/admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password: 'faux' }) });
    for (let i = 4; i >= 0; i--) {
      const r = await bad();
      assert.equal(r.status, 401);
      assert.equal((await r.json()).remaining, i);
    }
    const blocked = await fetch(t.base + '/api/admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password: PASSWORD }) });
    assert.equal(blocked.status, 429, 'même le bon mot de passe est refusé pendant le blocage');
    assert.ok(Number(blocked.headers.get('retry-after')) > 0);
    t.advance(15 * 60 * 1000 + 1);
    await loginCookie(t);
  } finally { await t.close(); }
});

test('admin : cookie de session sûr, déconnexion après 30 minutes d’inactivité', async () => {
  const t = await setup();
  try {
    const r = await fetch(t.base + '/api/admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password: PASSWORD }) });
    const sc = r.headers.get('set-cookie');
    assert.match(sc, /HttpOnly/); assert.match(sc, /SameSite=Strict/);
    const cookie = sc.split(';')[0];
    const list = () => fetch(t.base + '/api/admin/subscribers', { headers: { Cookie: cookie } });

    t.advance(29 * 60 * 1000);
    assert.equal((await list()).status, 200, 'activité à 29 min : session prolongée');
    t.advance(29 * 60 * 1000);
    assert.equal((await list()).status, 200, '29 min après la dernière activité : toujours valide');
    t.advance(30 * 60 * 1000 + 1);
    const exp = await list();
    assert.equal(exp.status, 401);
    assert.equal((await exp.json()).error, 'session_expired');
    assert.equal((await list()).status, 401);
  } finally { await t.close(); }
});

test('admin : liste et DELETE /api/admin/subscribers/{id}', async () => {
  const t = await setup();
  try {
    await post(t.base, '/api/subscribe', form(valid));
    await post(t.base, '/api/subscribe', form({ ...valid, email: 'subscriber3@example.test' }));
    const id = t.app.store.findByEmail('subscriber1@example.test').id;

    const anon = await fetch(t.base + '/api/admin/subscribers/' + id, { method: 'DELETE' });
    assert.equal(anon.status, 401);
    assert.equal(t.app.store.all().length, 2);

    const cookie = await loginCookie(t);
    const l = await fetch(t.base + '/api/admin/subscribers', { headers: { Cookie: cookie } });
    const { subscribers } = await l.json();
    assert.equal(subscribers.length, 2);
    assert.ok(subscribers.every((s) => !('unsubscribe_token_hash' in s)));

    const foreign = await fetch(t.base + '/api/admin/subscribers/' + id, { method: 'DELETE', headers: { Cookie: cookie, Origin: 'https://malveillant.example' } });
    assert.equal(foreign.status, 403);

    const d = await fetch(t.base + '/api/admin/subscribers/' + id, { method: 'DELETE', headers: { Cookie: cookie } });
    assert.equal(d.status, 204);
    assert.equal(t.app.store.findByEmail('subscriber1@example.test'), null);
    assert.ok(!fs.readFileSync(t.config.dataFile, 'utf8').includes('subscriber1@example.test'));
    const d2 = await fetch(t.base + '/api/admin/subscribers/' + id, { method: 'DELETE', headers: { Cookie: cookie } });
    assert.equal(d2.status, 404);
  } finally { await t.close(); }
});

test('configuration : mot de passe obligatoire, 20 caractères minimum', () => {
  assert.throws(() => loadConfig({}), /ADMIN_PASSWORD est obligatoire/);
  assert.throws(() => loadConfig({ ADMIN_PASSWORD: 'court' }), /au moins 20/);
});

test('statique : pages, en-têtes de sécurité, pas de traversée de répertoire', async () => {
  const t = await setup();
  try {
    const home = await fetch(t.base + '/');
    assert.equal(home.status, 200);
    assert.match(home.headers.get('content-security-policy'), /script-src 'self'/);
    for (const p of ['/mentions-legales', '/cgv', '/confidentialite', '/merci', '/admin', '/styles.css', '/fonts/ibm-plex-sans-latin-400-normal.woff2']) {
      assert.equal((await fetch(t.base + p)).status, 200, p);
    }
    assert.equal((await fetch(t.base + '/..%2fpackage.json')).status, 404);
    assert.equal((await fetch(t.base + '/%2e%2e/server/config.js')).status, 404);
    assert.equal((await fetch(t.base + '/introuvable')).status, 404);
  } finally { await t.close(); }
});

test('contenu : aucun mot interdit, aucun outil de suivi, aucune animation', () => {
  const pub = path.join(__dirname, '..', 'public');
  const files = fs.readdirSync(pub).filter((f) => /\.(html|css|js)$/.test(f));
  const all = files.map((f) => fs.readFileSync(path.join(pub, f), 'utf8')).join('\n').toLowerCase();
  for (const w of ['signal d’achat', "signal d'achat", 'garanti', 'sans risque', 'des milliers', 'google-analytics', 'googletagmanager', 'gtag(', 'fbq(', 'connect.facebook.net', 'hotjar', 'plausible', 'matomo']) {
    assert.ok(!all.includes(w), 'mot interdit trouvé : ' + w);
  }
  const css = fs.readFileSync(path.join(pub, 'styles.css'), 'utf8');
  assert.ok(!/@keyframes/.test(css));
  assert.ok(!/transition\s*:(?!\s*none)/.test(css));
  assert.ok(!/fonts\.googleapis|fonts\.gstatic/.test(all), 'polices auto-hébergées');
});
