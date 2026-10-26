'use strict';
// Test-only bootstrap. No live env, Discord, SMTP, research or trader stores.
const fs = require('node:fs'), os = require('node:os'), path = require('node:path'), http = require('node:http');
const { loadConfig } = require('../server/config');
const { createApp } = require('../server/app');
const { fixture, environment, IDS, roleIds } = require('./member-fixtures');
const { canonical } = require('../server/member/snapshots');
const { hash } = require('../server/member/security');
const empty = process.argv.includes('--empty'), port = empty ? 3828 : 3827;
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-member-browser-')), base = 'http://127.0.0.1:' + port;
const config = loadConfig(environment(dir, base));
const data = fixture();
if (empty) for (const key of ['briefs', 'radar', 'calendar', 'forecasts', 'failures', 'series', 'screener', 'claims']) data[key] = [];
else {
  data.forecasts = []; data.series = [];
  for (const [index, agent] of ['HypR-A1', 'HypR-E1'].entries()) {
    let error = 0, hits = 0; const points = [], distribution = [1];
    for (let i = 1; i <= 24; i++) {
      const time = Date.now() - (25 - i) * 86400000, at = new Date(time).toISOString();
      const p = (i + index) % 4 ? .65 : .4, outcome = (i + index) % 3 ? 1 : 0;
      const c = { ...fixture().forecasts[0].commitment, agent, session: at.slice(0, 10), probability: p.toFixed(6), committed_at: new Date(time - 3600000).toISOString(), reveal_at: at, nonce: hash('synthetic-' + agent + '-' + i) };
      data.forecasts.push({ ...fixture().forecasts[0], commitment: c, sha256: hash(canonical(c)), outcome, scored_at: at, source: { ...fixture().forecasts[0].source, published_at: at, available_at: at } });
      error += (p - outcome) ** 2; hits += Number((p >= .5) === Boolean(outcome));
      for (let k = distribution.length; k >= 0; k--) distribution[k] = ((distribution[k] || 0) + (k ? distribution[k - 1] : 0)) / 2;
      const recent = data.forecasts.filter(r => r.commitment.agent === agent).slice(-20);
      points.push({ at, n: i, brier_skill: 1 - error / (i * .25), rolling_hit_rate: recent.filter(r => (Number(r.commitment.probability) >= .5) === Boolean(r.outcome)).length / recent.length, luck_tail: distribution.slice(hits).reduce((a, b) => a + b, 0) });
    }
    data.series.push({ agent, variant: 'baseline-v1', points });
  }
}
const userTiers = new Map();
const discord = { async exchange(code) { const tier = Object.hasOwn(roleIds, code) ? code : 'premium'; userTiers.set(code, tier); return { access_token: code, refresh_token: code, expires_in: 3600 }; }, async user(code) { return { id: ['early', 'premium-mobile'].includes(code) ? IDS.other : IDS.user }; }, async roles(code) { return [roleIds[userTiers.get(code) || 'premium']]; }, async refresh(code) { return { access_token: code, refresh_token: code, expires_in: 3600 }; } };
const destinations = new Map();
const router = { async call(customer, action, fields = {}) {
  const own = destinations.get(customer) || [];
  if (action === 'list') return { destinations: own, email_enabled: false };
  if (action === 'add') { const id = hash(customer + own.length).slice(0, 32); own.push({ id, channel: fields.channel, paused: false, kinds: fields.kinds, watchlist: fields.watchlist }); destinations.set(customer, own); return { id }; }
  const row = own.find(r => r.id === fields.destination); if (!row) throw Object.assign(Error('refused'), { status: 400 });
  if (action === 'pause') row.paused = fields.paused; else destinations.set(customer, own.filter(r => r !== row)); return { ok: true };
} };
const app = createApp(config, { mailer: { async send() {} }, member: { discord, router, snapshots: { read: () => structuredClone(data) } }, logger: { warn() {}, error() {}, info() {} } });
const server = http.createServer(app); server.headersTimeout = 15000; server.requestTimeout = 20000; server.listen(port, '127.0.0.1');
function close() { app.close(); server.close(() => { fs.rmSync(dir, { recursive: true, force: true }); process.exit(0); }); }
process.on('SIGTERM', close); process.on('SIGINT', close);
