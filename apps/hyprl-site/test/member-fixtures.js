'use strict';
// Synthetic only; never loaded by the production entry point.
const { hash } = require('../server/member/security');
const { canonical } = require('../server/member/snapshots');
const IDS = { user: '900000000000000001', other: '900000000000000002', guild: '800000000000000001' };
const roleIds = Object.fromEntries(['early', 'essentiel', 'premium', 'fondateur', 'equipe'].map((t, i) => [t, '70000000000000000' + i]));
function environment(dir, publicUrl) {
  return { ADMIN_PASSWORD: 'synthetic-admin-password-long', DATA_FILE: dir + '/subscribers.json', PUBLIC_URL: publicUrl, MEMBER_ENABLED: 'true', DISCORD_CLIENT_ID: '800000000000000002', DISCORD_CLIENT_SECRET: 'synthetic-oauth-secret', DISCORD_GUILD_ID: IDS.guild, MEMBER_IDENTITY_KEY: '1'.repeat(64), MEMBER_STATE_FILE: dir + '/members.json', MEMBER_SNAPSHOT_DIR: dir + '/snapshots', ...Object.fromEntries(Object.entries(roleIds).map(([t, id]) => ['DISCORD_ROLE_' + t.toUpperCase(), id])) };
}
function fixture(now = Date.now()) {
  const at = new Date(now - 60000).toISOString(), source = { url: 'https://example.com/factual-source', sha256: 'a'.repeat(64), published_at: at, available_at: at };
  const c = { agent: 'HypR-A1', variant: 'baseline-v1', session: at.slice(0, 10), symbol: 'DEMO', event: 'close_above_open', probability: '0.600000', base_rate: '0.500000', committed_at: new Date(now - 120000).toISOString(), reveal_at: at, nonce: 'b'.repeat(64) };
  const forecasts = [{ commitment: c, sha256: hash(canonical(c)), receipt_url: 'https://example.com/receipt', outcome: 1, scored_at: at, source }];
  const fact = { kind: 'sec_8k', symbol: 'DEMO', summary: 'Dépôt synthétique réservé aux tests.', source };
  const points = [{ at, n: 1, brier_skill: 0.36, rolling_hit_rate: 1, luck_tail: 0.5 }];
  return { schema_version: 1, provenance: 'synthetic', generated_at: at, briefs: [{ date: at.slice(0, 10), published_at: at, facts: [fact] }], radar: [fact, { ...fact, kind: 'fed', symbol: null }], calendar: [{ kind: 'fed', symbol: null, scheduled_at: new Date(now + 3600000).toISOString(), source }], forecasts, failures: [{ agent: 'HypR-E1', variant: 'baseline-v1', session: c.session, code: 'missing', source }], revenue: [], claims: [{ claim: 'Affirmation synthétique pour vérifier le parcours.', verdict: 'inconclusive', verified_at: at, source }], series: [{ agent: 'HypR-A1', variant: 'baseline-v1', points }], screener: [{ symbol: 'DEMO', cik: '0000000001', concept: 'Revenues', periods: [{ start: '2025-01-01', end: '2025-03-31', revenue: '100', currency: 'USD', source }, { start: '2025-04-01', end: '2025-06-30', revenue: '110', currency: 'USD', source }, { start: '2025-07-01', end: '2025-09-30', revenue: '132', currency: 'USD', source }], growth_previous: 10, growth_latest: 20, acceleration_pp: 10 }] };
}
function fakeDiscord() {
  const users = new Map(), roles = new Map(Object.values(IDS).map(id => [id, [roleIds.premium]]));
  return { rolesByUser: roles, calls: [], async exchange(code, verifier) { this.calls.push({ code, verifier }); const user = code === 'other' ? IDS.other : IDS.user; const access = 'synthetic-access-' + user; users.set(access, user); return { access_token: access, refresh_token: access, token_type: 'Bearer', expires_in: 3600 }; }, async user(token) { return { id: users.get(token) }; }, async roles(token) { return roles.get(users.get(token)) || []; }, async refresh(token) { return { access_token: token, refresh_token: token, expires_in: 3600 }; } };
}
module.exports = { IDS, roleIds, environment, fixture, fakeDiscord };
