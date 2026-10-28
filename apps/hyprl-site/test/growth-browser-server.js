'use strict';
// Synthetic loopback fixture. Never read production environment, recipients or stores.
const fs = require('node:fs'), os = require('node:os'), path = require('node:path'), http = require('node:http');
const { createApp } = require('../server/app');
const { loadConfig } = require('../server/config');
const { fixture } = require('./member-fixtures');
const clock = Date.parse('2026-10-28T12:00:00Z');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-growth-browser-')), mails = [];
const config = loadConfig({ ADMIN_PASSWORD: 'synthetic-browser-test-password', DATA_FILE: dir + '/subscribers.json', PUBLIC_URL: 'http://127.0.0.1:3818', NEWSLETTER_ENABLED: 'true', NEWSLETTER_TRACKING_DISABLED: 'true', NEWSLETTER_TOKEN_KEY: '1'.repeat(64), NEWSLETTER_WORKER_KEY: '2'.repeat(64), SMTP_HOST: 'smtp.example.test', MAIL_FROM: 'letter@example.test', SMTP_EU_APPROVED: 'true' });
const value = fixture(clock);
value.calendar.push({ ...value.calendar[0], kind: 'earnings', symbol: 'DEMO', date: '2026-10-29' });
value.claims[0].verdict = 'verified';
const app = createApp(config, { now: () => clock, logger: { info() {}, warn() {}, error() {} }, publicContent: { snapshots: { read: () => structuredClone(value) } }, mailer: { async send(message) { mails.push(message); return true; } } });
const server = http.createServer((req, res) => {
  if (req.url === '/__qa/mail') { res.writeHead(200, { 'Content-Type': 'application/json' }); return res.end(JSON.stringify(mails)); }
  return app(req, res);
});
server.listen(3818, '127.0.0.1');
function close() { app.close(); server.close(() => { fs.rmSync(dir, { recursive: true, force: true }); process.exit(0); }); }
process.on('SIGTERM', close); process.on('SIGINT', close);
