'use strict';
// Test-only bootstrap. No live env, Discord, SMTP, research or trader stores.
const fs = require('node:fs'), os = require('node:os'), path = require('node:path'), http = require('node:http');
const { loadConfig } = require('../server/config');
const { createApp } = require('../server/app');
const { fixture, environment, IDS, roleIds } = require('./member-fixtures');
const { hash } = require('../server/member/security');
const empty = process.argv.includes('--empty'), observed = process.argv.includes('--observed'), port = empty ? 3828 : observed ? 3829 : 3827;
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-member-browser-')), base = 'http://127.0.0.1:' + port;
const config = loadConfig(environment(dir, base));
const data = fixture();
if (empty) for (const key of ['briefs', 'radar', 'calendar', 'forecasts', 'failures', 'reveals', 'series', 'screener', 'claims']) data[key] = [];
else {
  const original = structuredClone(data); data.forecasts = []; data.reveals = []; data.failures = [];
  for (const [index, agent] of ['HypR-A1', 'HypR-E1'].entries()) {
    for (let i = 1; i <= 24; i++) {
      const time = Date.now() - (25 - i) * 86400000, at = new Date(time).toISOString(), day = at.slice(0, 10);
      const proof = structuredClone(original.reveals[0]); proof.agent = agent; proof.session = day; proof.id = day + '-' + agent; proof.file = 'reveals/' + day + '/' + agent + '.json'; proof.url = 'https://github.com/Kacawaiii/HyprL/blob/' + proof.ledger_commit + '/' + proof.file;
      proof.published_at = at; proof.available_at = at; proof.commitment.committed_at = new Date(time - 3600000).toISOString(); proof.commitment.published_at = proof.commitment.committed_at; data.reveals.push(proof);
      for (const horizon of ['1d', '5d']) {
        const p = (i + index) % 4 ? .65 : .4, outcome = (i + index) % 3 ? 1 : 0;
        data.forecasts.push({ ...original.forecasts[0], agent, session: day, horizon, probability: p, outcome, scored_at: at, reveal_id: proof.id, decision_time: new Date(time - 7200000).toISOString(), entry_at: new Date(time - 7000000).toISOString() });
      }
    }
  }
  data.series = require('../server/member/snapshots-v2').computeSeries(data);
}
data.series = require('../server/member/snapshots-v2').computeSeries(data);
const snapshot = observed ? JSON.parse(fs.readFileSync(process.env.MEMBER_QA_SNAPSHOT, 'utf8')) : data;
fs.mkdirSync(config.member.snapshotDir, { mode: 0o700 });
fs.writeFileSync(path.join(config.member.snapshotDir, 'member-v2.json'), JSON.stringify(snapshot), { mode: 0o440 });
const snapshots = new (require('../server/member/snapshots').Snapshots)(config.member.snapshotDir, { allowSynthetic: !observed });
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
const app = createApp(config, { mailer: { async send() {} }, member: { discord, router, snapshots }, logger: { warn() {}, error() {}, info() {} } });
const server = http.createServer(app); server.headersTimeout = 15000; server.requestTimeout = 20000; server.listen(port, '127.0.0.1');
function close() { app.close(); server.close(() => { fs.rmSync(dir, { recursive: true, force: true }); process.exit(0); }); }
process.on('SIGTERM', close); process.on('SIGINT', close);
