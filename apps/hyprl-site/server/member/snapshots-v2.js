'use strict';
const fs = require('node:fs');
const path = require('node:path');
const Ajv = require('ajv');
const addFormats = require('ajv-formats');
const { hash, fail } = require('./security');
const schema = require('../../docs/member/snapshot-v2.schema.json');
const canonical = value => JSON.stringify(Array.isArray(value) ? value.map(v => JSON.parse(canonical(v))) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(k => [k, JSON.parse(canonical(value[k]))])) : value);
const SCHEMA_HASH = '12d94734cfdef908afb7d3b4d6282a2d8c607429d7e20b56f85fef0f32340435';
if (hash(canonical(schema)) !== SCHEMA_HASH) throw Error('Member v2 schema binding mismatch');
const ajv = new Ajv({ strict: false }); addFormats(ajv); const valid = ajv.compile(schema);
const repository = 'https://github.com/Kacawaiii/HyprL';
const baselineUrl = repository + '/blob/9afa669596fda2c247a970cc0060bbdbf595afd6/docs/artifacts/replay_duel_test_freeze_v15.json';
const panels = [['a_v1_as_live', '1d'], ['a_v1_as_live', '5d'], ['c_news_small_mid', '2d'], ['c_news_small_mid', '10d']];
const modes = ['REGULAR_0905', 'INTRADAY'], agents = ['HypR-A1', 'HypR-E1'];
function computeSeries(value) {
  const result = [];
  for (const agent of agents) for (const mode of modes) for (const [variant, horizon] of panels) {
    const rows = value.forecasts.filter(r => r.agent === agent && r.mode === mode && r.variant === variant && r.horizon === horizon).sort((a, b) => (a.session < b.session ? -1 : a.session > b.session ? 1 : 0) || (a.decision_time < b.decision_time ? -1 : a.decision_time > b.decision_time ? 1 : 0) || (a.symbol < b.symbol ? -1 : a.symbol > b.symbol ? 1 : 0));
    let error = 0, reference = 0, hits = 0; const recent = [], points = [];
    for (const r of rows) {
      error += (r.probability - r.outcome) ** 2;
      if (value.baseline) reference += (value.baseline.probability - r.outcome) ** 2;
      const hit = Number((r.probability >= .5) === Boolean(r.outcome)); hits += hit; recent.push(hit); if (recent.length > 20) recent.shift();
      points.push({ at: r.scored_at, session: r.session, symbol: r.symbol, n: points.length + 1, hits, brier: error / (points.length + 1), brier_base_rate: value.baseline ? reference / (points.length + 1) : null, brier_skill: reference ? 1 - error / reference : null, rolling_hit_rate: recent.reduce((a, b) => a + b, 0) / recent.length });
    }
    const last = points.at(-1), failures = value.failures.filter(f => f.agent === agent && f.mode === mode).length;
    result.push({ agent, mode, variant, horizon, n: rows.length, hits, brier: last?.brier ?? null, brier_base_rate: last?.brier_base_rate ?? null, brier_skill: last?.brier_skill ?? null, rolling_hit_rate: last?.rolling_hit_rate ?? null, failures, model_failures: failures, trade_n: rows.filter(r => r.trade_hit !== null).length, trade_hits: rows.filter(r => r.trade_hit === true).length, points });
  }
  return result;
}
function same(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) <= 1e-12;
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return a === b;
  return Object.keys(a).sort().join() === Object.keys(b).sort().join() && Object.keys(a).every(k => same(a[k], b[k]));
}
function verify(value, now, allowSynthetic = false) {
  const refuse = () => { throw Error('Invalid member snapshot'); };
  if (!valid(value) || (value.provenance !== 'observed' && !allowSynthetic) || Date.parse(value.generated_at) > now) refuse();
  const cutoff = Date.parse(value.generated_at), base = value.baseline;
  if (base && (base.n !== 1214 || base.successes !== 590 || base.probability !== 591 / 1216 || base.sha256 !== '0693f1dc690d690ea6e3bd1d797e0ba6f8506640173126c1565ae7d2466bb4f4' || base.url !== baselineUrl)) refuse();
  function walk(v) {
    if (!v || typeof v !== 'object') return;
    for (const key of ['available_at', 'published_at', 'scored_at', 'received_at']) if (v[key] && Date.parse(v[key]) > cutoff) refuse();
    if (v.available_at && v.published_at && Date.parse(v.available_at) < Date.parse(v.published_at)) refuse();
    Object.values(v).forEach(walk);
  }
  walk(value);
  const reveals = new Map();
  for (const r of value.reveals) {
    const key = r.session + (r.mode === 'INTRADAY' ? '-intraday' : ''), c = r.commitment;
    const file = 'reveals/' + key + '/' + (r.status === 'REVEALED' ? r.agent + '.json' : 'README.md');
    if (r.id !== key + '-' + r.agent || reveals.has(r.id) || r.file !== file || r.url !== repository + '/blob/' + r.ledger_commit + '/' + file || c.url !== repository + '/blob/' + c.ledger_commit + '/forward-duel-ledger.jsonl' || !(Date.parse(c.committed_at) <= Date.parse(c.published_at) && Date.parse(c.published_at) < Date.parse(r.published_at) && Date.parse(r.published_at) <= Date.parse(r.available_at)) || (r.status === 'REVEALED' && c.sha256 !== r.sha256)) refuse();
    reveals.set(r.id, r);
  }
  const seen = new Set(), failed = new Set();
  for (const r of value.forecasts) {
    const key = [r.agent, r.mode, r.variant, r.horizon, r.session, r.symbol].join('/'), proof = reveals.get(r.reveal_id);
    if (seen.has(key) || !panels.some(([v, h]) => r.variant === v && r.horizon === h) || !proof || proof.status !== 'REVEALED' || proof.sha256 !== r.decision_sha256 || ['agent', 'mode', 'session'].some(k => proof[k] !== r[k]) || Date.parse(r.decision_time) >= Date.parse(proof.published_at) || Date.parse(r.decision_time) >= Date.parse(r.scored_at) || (r.entry_at && !(Date.parse(r.decision_time) < Date.parse(r.entry_at) && Date.parse(r.entry_at) <= Date.parse(r.scored_at)))) refuse();
    seen.add(key);
  }
  for (const r of value.failures) {
    const key = [r.agent, r.mode, r.session].join('/'), proof = reveals.get(r.reveal_id);
    if (failed.has(key) || !proof || proof.status !== 'NO_DECISION' || ['agent', 'mode', 'session'].some(k => proof[k] !== r[k])) refuse();
    failed.add(key);
  }
  const calendar = new Set();
  for (const r of value.calendar) {
    const key = [r.kind, r.symbol, r.date].join('/');
    if (calendar.has(key) || (r.scheduled_at && r.scheduled_at.slice(0, 10) !== r.date)) refuse();
    calendar.add(key);
  }
  if (!same(value.series, computeSeries(value))) refuse();
  return value;
}
class Snapshots {
  constructor(directory, { now = Date.now, allowSynthetic = false } = {}) { Object.assign(this, { directory, now, allowSynthetic }); }
  read() {
    let fd;
    try {
      if (!this.directory || fs.realpathSync(this.directory) !== path.resolve(this.directory)) throw Error();
      fd = fs.openSync(path.join(this.directory, 'member-v2.json'), fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
      const stat = fs.fstatSync(fd);
      if (!stat.isFile() || stat.size > 2 * 1024 * 1024 || (stat.mode & 0o022)) throw Error();
      return verify(JSON.parse(fs.readFileSync(fd, 'utf8')), this.now(), this.allowSynthetic);
    } catch { throw fail(503, 'snapshot_unavailable'); }
    finally { if (fd !== undefined) fs.closeSync(fd); }
  }
}
module.exports = { Snapshots, computeSeries, verify };
