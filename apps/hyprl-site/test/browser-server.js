'use strict';
// Isolated synthetic QA process. Never load the live environment or data file.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const { loadConfig } = require('../server/config');
const { createApp } = require('../server/app');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-browser-'));
const config = loadConfig({ ADMIN_PASSWORD: 'synthetic-browser-test-password', DATA_FILE: path.join(dir, 'subs.json'), PUBLIC_URL: 'http://127.0.0.1:3817' });
const app = createApp(config, { mailer: { async send() {} }, logger: { info() {}, warn() {}, error() {} } });
const server = http.createServer(app);
server.listen(3817, '127.0.0.1');
function close() {
  app.close();
  server.close(() => { fs.rmSync(dir, { recursive: true, force: true }); process.exit(0); });
}
process.on('SIGTERM', close);
process.on('SIGINT', close);
