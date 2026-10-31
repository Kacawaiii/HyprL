'use strict';
// Loopback-only synthetic QA. No live env, SMTP, state or provider access.
const fs = require('node:fs'), os = require('node:os'), path = require('node:path'), http = require('node:http'), crypto = require('node:crypto');
const { loadConfig } = require('../server/config');
const { createApp } = require('../server/app');
const { CONSENT_TEXTS } = require('../server/lib/consent');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-triage-browser-'));
const config = loadConfig({ FORM_TOKEN_KEY: 'a'.repeat(64), SMTP_HOST: 'smtp.example.test', MAIL_FROM: 'from@example.test', SMTP_EU_APPROVED: 'true', ADMIN_PASSWORD: 'synthetic-triage-browser-password', DATA_FILE: path.join(dir, 'subscribers.json'), PUBLIC_URL: 'http://127.0.0.1:3819' });
const at = new Date().toISOString();
const record = (email, extra = {}) => ({ id: crypto.randomUUID(), email, created_at: at, channels: ['discord', 'email'], actions: 'AAPL', consent: { given: true, version: '2026-10-v1', text: CONSENT_TEXTS['2026-10-v1'], checked_at: at }, ...extra });
fs.writeFileSync(config.dataFile, JSON.stringify({ subscribers: [
  record('confirmed@example.test', { status: 'confirmed', confirmed_at: at }),
  record('pending@example.test', { status: 'pending', confirmation_delivery: 'sent' }),
  record('review@example.test', { actions: 'I wanted to know your price' }),
  record('delete@example.test', { actions: '<script>synthetic</script>' }),
] }), { mode: 0o600 });
const mails = [];
const app = createApp(config, { mailer: { async send(m) { mails.push(m); return true; } }, logger: { info() {}, warn() {}, error() {} } });
const server = http.createServer((req, res) => {
  if (req.url === '/__qa/mail') { res.setHeader('Content-Type', 'application/json'); return res.end(JSON.stringify(mails)); }
  return app(req, res);
});
server.listen(3819, '127.0.0.1');
function close() { app.close(); server.close(() => { fs.rmSync(dir, { recursive: true, force: true }); process.exit(0); }); }
process.on('SIGTERM', close); process.on('SIGINT', close);
