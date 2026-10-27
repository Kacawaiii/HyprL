'use strict';
// Synthetic only; never loaded by the production entry point.
const IDS = { user: '900000000000000001', other: '900000000000000002', guild: '800000000000000001' };
const roleIds = Object.fromEntries(['early', 'essentiel', 'premium', 'fondateur', 'equipe'].map((t, i) => [t, '70000000000000000' + i]));
function environment(dir, publicUrl) {
  return { ADMIN_PASSWORD: 'synthetic-admin-password-long', DATA_FILE: dir + '/subscribers.json', PUBLIC_URL: publicUrl, MEMBER_ENABLED: 'true', DISCORD_CLIENT_ID: '800000000000000002', DISCORD_CLIENT_SECRET: 'synthetic-oauth-secret', DISCORD_GUILD_ID: IDS.guild, MEMBER_IDENTITY_KEY: '1'.repeat(64), MEMBER_STATE_FILE: dir + '/members.json', MEMBER_SNAPSHOT_DIR: dir + '/snapshots', ...Object.fromEntries(Object.entries(roleIds).map(([t, id]) => ['DISCORD_ROLE_' + t.toUpperCase(), id])) };
}
function fixture(now = Date.now()) {
  const at = new Date(now - 60000).toISOString(), day = at.slice(0, 10), source = { url: 'https://example.com/factual-source', sha256: 'a'.repeat(64), published_at: at, available_at: at };
  const repository = 'https://github.com/Kacawaiii/HyprL';
  function reveal(agent, status) {
    const file = 'reveals/' + day + '/' + (status === 'REVEALED' ? agent + '.json' : 'README.md');
    return { id: day + '-' + agent, agent, session: day, mode: 'REGULAR_0905', status, ledger_commit: 'a'.repeat(40), file, sha256: status === 'REVEALED' ? 'b'.repeat(64) : 'c'.repeat(64), url: repository + '/blob/' + 'a'.repeat(40) + '/' + file, published_at: at, available_at: at, commitment: { ledger_commit: 'd'.repeat(40), url: repository + '/blob/' + 'd'.repeat(40) + '/forward-duel-ledger.jsonl', committed_at: new Date(now - 3600000).toISOString(), published_at: new Date(now - 3500000).toISOString(), sha256: status === 'REVEALED' ? 'b'.repeat(64) : 'e'.repeat(64) } };
  }
  const reveals = [reveal('HypR-A1', 'REVEALED'), reveal('HypR-E1', 'NO_DECISION')];
  const forecasts = [{ agent: 'HypR-A1', variant: 'a_v1_as_live', mode: 'REGULAR_0905', horizon: '1d', session: day, symbol: 'DEMO', event: 'outperform_spy', probability: .6, outcome: 1, decision_time: new Date(now - 7200000).toISOString(), entry_at: new Date(now - 7000000).toISOString(), scored_at: at, decision_sha256: 'b'.repeat(64), snapshot_sha256: 'f'.repeat(64), score_sha256: 'f'.repeat(64), source_spec_sha256: 'f'.repeat(64), source_code_sha256: 'f'.repeat(64), reveal_id: reveals[0].id, trade_hit: true }];
  const fact = { kind: 'sec_8k', symbol: 'DEMO', summary: 'Dépôt synthétique réservé aux tests.', source };
  const value = { schema_version: 2, provenance: 'synthetic', generated_at: at, baseline: { probability: 591 / 1216, n: 1214, successes: 590, panel: 'a_v1_as_live:1d', smoothing: 'laplace_1_1', sha256: '0693f1dc690d690ea6e3bd1d797e0ba6f8506640173126c1565ae7d2466bb4f4', url: repository + '/blob/9afa669596fda2c247a970cc0060bbdbf595afd6/docs/artifacts/replay_duel_test_freeze_v15.json' }, briefs: [{ date: day, published_at: at, facts: [fact] }], radar: [fact, { ...fact, kind: 'fed', symbol: null }], calendar: [{ kind: 'fed', symbol: null, date: new Date(now + 86400000).toISOString().slice(0, 10), precision: 'date', scheduled_at: null, source: null, evidence: { kind: 'observed_day_marker', sha256: 'a'.repeat(64), dataset_sha256: 'a'.repeat(64), received_at: at } }], forecasts, failures: [{ agent: 'HypR-E1', mode: 'REGULAR_0905', session: day, status: 'MODEL_FAILURE', code: 'MODEL_USAGE_LIMIT', reveal_id: reveals[1].id }], reveals, revenue: [], claims: [{ claim: 'Affirmation synthétique pour vérifier le parcours.', verdict: 'inconclusive', verified_at: at, source }], screener: [] };
  value.series = require('../server/member/snapshots-v2').computeSeries(value);
  return value;
}
function fakeDiscord() {
  const users = new Map(), roles = new Map(Object.values(IDS).map(id => [id, [roleIds.premium]]));
  return { rolesByUser: roles, calls: [], async exchange(code, verifier) { this.calls.push({ code, verifier }); const user = code === 'other' ? IDS.other : IDS.user; const access = 'synthetic-access-' + user; users.set(access, user); return { access_token: access, refresh_token: access, token_type: 'Bearer', expires_in: 3600 }; }, async user(token) { return { id: users.get(token) }; }, async roles(token) { return roles.get(users.get(token)) || []; }, async refresh(token) { return { access_token: token, refresh_token: token, expires_in: 3600 }; } };
}
module.exports = { IDS, roleIds, environment, fixture, fakeDiscord };
