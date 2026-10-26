'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { loadConfig } = require('../server/config');
const { createApp } = require('../server/app');
const { Snapshots } = require('../server/member/snapshots');
const { Discord } = require('../server/member/discord');
const { FEATURES, TIERS, hash, memberKey, Budget } = require('../server/member/security');
const { IDS, roleIds, environment, fixture, fakeDiscord } = require('./member-fixtures');
async function setup(t, opts = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-member-test-'));
  const config = loadConfig(environment(dir, 'http://127.0.0.1'));
  let clock = Date.now(); const discord = fakeDiscord(), snapshot = fixture(clock);
  const routerCalls = [], dests = new Map();
  const router = { async call(customer, action, fields = {}) {
    routerCalls.push({ customer, action, fields }); const own = dests.get(customer) || [];
    if (action === 'list') return { destinations: own, email_enabled: false };
    if (action === 'add') { const id = hash(customer).slice(0, 32); own.push({ id, channel: fields.channel, paused: false, kinds: fields.kinds, watchlist: fields.watchlist }); dests.set(customer, own); return { id }; }
    const row = own.find(r => r.id === fields.destination); if (!row) throw Object.assign(Error('router_refused'), { status: 400 });
    if (action === 'pause') row.paused = fields.paused; else dests.set(customer, own.filter(r => r !== row));
    return { ok: true };
  } };
  const app = createApp(config, { now: () => clock, mailer: { async send() {} }, logger: { error() {}, warn() {}, info() {} }, member: { discord, snapshots: { read: () => structuredClone(snapshot) }, router, ...(opts.member || {}) } });
  const server = http.createServer(app); await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = 'http://127.0.0.1:' + server.address().port;
  // The origin is intentionally pinned by configuration; tests supply that origin.
  async function request(route, { method = 'GET', cookies = '', csrf, data, origin = 'http://127.0.0.1' } = {}) {
    return fetch(base + route, { method, redirect: 'manual', headers: { ...(cookies ? { Cookie: cookies } : {}), ...(method !== 'GET' ? { Origin: origin, 'Content-Type': 'application/json' } : {}), ...(csrf ? { 'X-CSRF-Token': csrf } : {}) }, ...(data === undefined ? {} : { body: JSON.stringify(data) }) });
  }
  async function start() { const r = await request('/auth/discord/start'); return { cookie: r.headers.get('set-cookie').split(';')[0], state: new URL(r.headers.get('location')).searchParams.get('state'), response: r }; }
  async function login(code = 'primary') {
    const flow = await start(); const r = await request('/auth/discord/complete', { method: 'POST', cookies: flow.cookie, data: { code, state: flow.state } }); assert.equal(r.status, 200);
    const cookies = r.headers.getSetCookie().find(c => c.startsWith('hyprl_member=')).split(';')[0]; const s = await (await request('/api/member/v1/session', { cookies })).json(); return { cookies, csrf: s.csrf };
  }
  t.after(async () => { app.close(); server.closeAllConnections(); await new Promise(r => server.close(r)); fs.rmSync(dir, { recursive: true, force: true }); });
  return { dir, config, discord, snapshot, routerCalls, request, start, login, app, advance: ms => { clock += ms; } };
}
const features = ['dashboard', 'briefs', 'radar', 'calendar', 'duel', 'luck', 'screener', 'claims', 'alerts', 'account', 'watchlist', 'proofs'];
for (const tier of Object.keys(TIERS)) {
  test('all API and HTML boundaries for Discord tier ' + tier, async t => {
    const s = await setup(t); s.discord.rolesByUser.set(IDS.user, [roleIds[tier]]); const auth = await s.login();
    const pages = { dashboard: '', briefs: '/brief', radar: '/radar', calendar: '/calendrier', duel: '/duel', luck: '/chance', screener: '/screener', claims: '/chance-ou-pas', alerts: '/alertes', account: '/compte', proofs: '/preuves' };
    for (const feature of features) {
      const allowed = TIERS[tier] >= FEATURES[feature]; const route = feature === 'proofs' ? feature + '/' + s.snapshot.forecasts[0].sha256 : feature;
      const r = await s.request('/api/member/v1/' + route, auth); assert.equal(r.status, allowed ? 200 : 403, feature); assert.equal(r.headers.get('cache-control'), 'no-store');
      if (pages[feature] !== undefined) { const page = await s.request('/membre' + pages[feature], auth); assert.equal(page.status, allowed ? 200 : 403, feature + ' page'); }
    }
  });
}
test('anonymous requests never reveal any member resource', async t => {
  const s = await setup(t);
  for (const feature of [...features, 'unknown']) { const r = await s.request('/api/member/v1/' + feature); assert.equal(r.status, 401, feature); assert.ok(!(await r.text()).includes('synthetic')); }
  const page = await s.request('/membre/duel'); assert.equal(page.status, 303); assert.equal(page.headers.get('location'), '/membre/connexion');
});
test('OAuth binds PKCE/state to a Strict HttpOnly cookie, one use, no fixation', async t => {
  const s = await setup(t); const flow = await s.start(), target = new URL(flow.response.headers.get('location'));
  assert.equal(target.origin, 'https://discord.com'); assert.equal(target.searchParams.get('scope'), 'identify guilds.members.read'); assert.equal(target.searchParams.get('code_challenge_method'), 'S256'); assert.match(flow.response.headers.get('set-cookie'), /HttpOnly; SameSite=Strict/);
  const missing = await s.request('/auth/discord/complete', { method: 'POST', data: { code: 'x', state: flow.state } }); assert.equal(missing.status, 401);
  const cross = await s.request('/auth/discord/complete', { method: 'POST', cookies: flow.cookie, origin: 'https://example.com', data: { code: 'x', state: flow.state } }); assert.equal(cross.status, 403);
  const ok = await s.request('/auth/discord/complete', { method: 'POST', cookies: flow.cookie, data: { code: 'x', state: flow.state } }); assert.equal(ok.status, 200);
  const replay = await s.request('/auth/discord/complete', { method: 'POST', cookies: flow.cookie, data: { code: 'x', state: flow.state } }); assert.equal(replay.status, 401);
  assert.equal(s.discord.calls.length, 1); const verifier = s.discord.calls[0].verifier; assert.equal(Buffer.from(hash(verifier), 'hex').toString('base64url'), target.searchParams.get('code_challenge'));
  const token = ok.headers.getSetCookie().find(c => c.startsWith('hyprl_member=')).split(';')[0].split('=')[1]; assert.ok(!s.app.member.sessions.items.has(token)); assert.ok(s.app.member.sessions.items.has(hash(token)));
  const callback = await s.request('/auth/discord/callback?code=private-test&state=private-state'); const html = await callback.text(); assert.ok(!html.includes('private-test')); assert.equal(callback.headers.get('referrer-policy'), 'no-referrer');
});
test('expired OAuth transaction is refused without contacting Discord', async t => {
  const s = await setup(t), f = await s.start(); s.advance(300001);
  assert.equal((await s.request('/auth/discord/complete', { method: 'POST', cookies: f.cookie, data: { code: 'x', state: f.state } })).status, 401); assert.equal(s.discord.calls.length, 0);
});
test('role downgrade and removal take effect on the next request; outage fails closed', async t => {
  const s = await setup(t), auth = await s.login(); s.discord.rolesByUser.set(IDS.user, [roleIds.early]);
  assert.equal((await s.request('/api/member/v1/duel', auth)).status, 403);
  assert.equal((await s.request('/api/member/v1/claims', auth)).status, 200);
  s.discord.roles = async () => { throw Object.assign(Error('discord_unavailable'), { status: 503 }); };
  assert.equal((await s.request('/api/member/v1/claims', auth)).status, 503);
  s.discord.roles = async () => [];
  assert.equal((await s.request('/api/member/v1/claims', auth)).status, 403);
  assert.equal((await s.request('/api/member/v1/session', auth)).status, 401);
});
test('idle and absolute session expiry; logout destroys its hash', async t => {
  const s = await setup(t), first = await s.login(); s.advance(30 * 60000);
  assert.equal((await s.request('/api/member/v1/session', first)).status, 401);
  const second = await s.login();
  for (let i = 0; i < 24; i++) { s.advance(29 * 60000); assert.equal((await s.request('/api/member/v1/session', second)).status, 200); }
  s.advance(24 * 60000); assert.equal((await s.request('/api/member/v1/session', second)).status, 401);
  const third = await s.login(); assert.equal((await s.request('/api/member/v1/logout', { ...third, method: 'POST', data: {} })).status, 200); assert.equal((await s.request('/api/member/v1/session', third)).status, 401);
});
test('CSRF and cross-member watchlist, radar, alert destinations and billing isolation', async t => {
  const s = await setup(t), a = await s.login(), b = await s.login('other');
  assert.equal((await s.request('/api/member/v1/watchlist', { cookies: a.cookies, method: 'PUT', data: { symbols: ['DEMO'] } })).status, 403);
  assert.equal((await s.request('/api/member/v1/watchlist', { ...a, method: 'PUT', origin: 'https://example.com', data: { symbols: ['DEMO'] } })).status, 403);
  assert.equal((await s.request('/api/member/v1/watchlist', { ...a, method: 'PUT', data: { symbols: ['DEMO'] } })).status, 200);
  assert.deepEqual((await (await s.request('/api/member/v1/watchlist', b)).json()).symbols, []);
  assert.equal((await (await s.request('/api/member/v1/radar', a)).json()).items.length, 2); assert.equal((await (await s.request('/api/member/v1/radar', b)).json()).items.length, 1);
  const add = await s.request('/api/member/v1/alerts', { ...a, method: 'POST', data: { channel: 'webhook', config: { url: 'https://example.com/destination', signing_key: 'x'.repeat(32) }, kinds: ['sec_filing'], watchlist: ['DEMO'] } }); assert.equal(add.status, 200); const dest = (await add.json()).id;
  assert.equal((await (await s.request('/api/member/v1/alerts', b)).json()).destinations.length, 0);
  assert.equal((await s.request('/api/member/v1/alerts', { ...b, method: 'DELETE', data: { destination: dest } })).status, 400);
  assert.equal((await s.request('/api/member/v1/alerts', { ...a, method: 'PATCH', data: { destination: dest, paused: true } })).status, 200);
  assert.equal(s.routerCalls.at(-1).customer, IDS.guild + '/' + IDS.user);
  s.app.member.state.billingOrigin = 'https://example.com';
  await s.app.member.state.applyBillingUpdate(memberKey(IDS.user, s.config.member), { renewal_at: '2027-01-01T00:00:00Z', manage_url: 'https://example.com/manage/private-a', cancel_url: 'https://example.com/cancel/private-a' });
  assert.equal((await (await s.request('/api/member/v1/account', b)).json()).billing, null); assert.ok((await s.request('/api/member/v1/account', a)).status === 200);
  assert.ok(!fs.readFileSync(s.config.member.stateFile, 'utf8').includes(IDS.user)); assert.equal(fs.statSync(s.config.member.stateFile).mode & 0o777, 0o600);
});
test('all write verbs retain tier guards; rejects extra fields and malformed symbols', async t => {
  const s = await setup(t); s.discord.rolesByUser.set(IDS.user, [roleIds.early]); const auth = await s.login();
  for (const [route, method] of [['watchlist', 'PUT'], ['alerts', 'POST'], ['alerts', 'PATCH'], ['alerts', 'DELETE']]) assert.equal((await s.request('/api/member/v1/' + route, { ...auth, method, data: {} })).status, 403);
  s.discord.rolesByUser.set(IDS.user, [roleIds.premium]);
  for (const symbols of [['bad<script>'], ['lowercase'], Array(51).fill('DEMO')]) assert.equal((await s.request('/api/member/v1/watchlist', { ...auth, method: 'PUT', data: { symbols } })).status, 400);
  assert.equal((await s.request('/api/member/v1/watchlist', { ...auth, method: 'PUT', data: { symbols: [], user: IDS.other } })).status, 400);
});
test('missing production snapshots return pending; schema rejects synthetic, symlinks and unrevealed content', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-snapshot-')); t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const reader = new Snapshots(dir); assert.throws(() => reader.read(), /snapshot_unavailable/);
  const file = path.join(dir, 'member-v1.json'), data = fixture();
  const write = () => { if (fs.existsSync(file)) fs.unlinkSync(file); fs.writeFileSync(file, JSON.stringify(data)); fs.chmodSync(file, 0o440); };
  write(); assert.throws(() => reader.read(), /snapshot_unavailable/);
  const testReader = new Snapshots(dir, { allowSynthetic: true }); assert.equal(testReader.read().forecasts.length, 1);
  data.provenance = 'observed'; write(); assert.equal(reader.read().forecasts.length, 1);
  data.forecasts[0].commitment.reveal_at = '2099-01-01T00:00:00Z'; write(); assert.throws(() => testReader.read(), /snapshot_unavailable/);
  fs.unlinkSync(file); fs.symlinkSync(path.join(dir, 'other.json'), file); assert.throws(() => testReader.read(), /snapshot_unavailable/);
});
test('strict CSP and secure Strict cookies in production config, no OAuth credential leaks', async t => {
  const s = await setup(t); const auth = await s.login(); const r = await s.request('/membre/duel', auth);
  assert.match(r.headers.get('content-security-policy'), /script-src 'self'/); assert.ok(!r.headers.get('content-security-policy').includes('unsafe-inline')); assert.ok(!(await r.text()).includes('synthetic-oauth-secret'));
  const c = loadConfig({ ...environment(s.dir, 'https://hyprlcore.com'), NODE_ENV: 'production' }); assert.equal(c.secureCookies, true);
  assert.throws(() => loadConfig({ ...environment(s.dir, 'https://hyprlcore.com'), MEMBER_STATE_FILE: path.join(c.root, 'private.json'), NODE_ENV: 'production' }));
});
test('OAuth endpoints are pinned, redirects fail, refresh preserves expected contract', async () => {
  const calls = [], d = new Discord({ clientId: 'x', clientSecret: 'synthetic', guildId: IDS.guild }, 'https://example.com', async (url, options) => { calls.push({ url, options }); return { ok: true, text: async () => JSON.stringify({ access_token: 'a', refresh_token: 'r', expires_in: 60, token_type: 'Bearer' }) }; });
  await d.exchange('code', 'verifier'); await d.refresh('r'); assert.equal(calls[0].url, 'https://discord.com/api/v10/oauth2/token'); assert.equal(calls[0].options.redirect, 'error'); assert.match(calls[0].options.body, /code_verifier=verifier/);
  d.request = async () => { throw Error('not printed'); }; await assert.rejects(d.roles('a'), /discord_unavailable/);
});
test('bounded request budget refuses exhaustion and recovers after expiry', () => { let now = 0; const b = new Budget(2, 1000, () => now); b.take('a'); b.take('a'); assert.throws(() => b.take('a'), /rate_limited/); now = 1000; b.take('a'); });
test('member preferences persist atomically, purge after 24 calendar months, rights deletion stays isolated', async t => {
  const { MemberState } = require('../server/member/state');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-member-state-')); t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  let now = Date.parse('2024-02-29T12:00:00Z'); const file = path.join(dir, 'state.json');
  const state = new MemberState(file, '', () => now), a = 'a'.repeat(64), b = 'b'.repeat(64);
  await Promise.all([state.setWatchlist(a, ['DEMO']), state.setWatchlist(b, ['OTHER'])]);
  assert.deepEqual(new MemberState(file, '', () => now).get(a).watchlist, ['DEMO']);
  assert.deepEqual(state.get(b).watchlist, ['OTHER']);
  await state.remove(a); assert.deepEqual(state.get(a).watchlist, []); assert.deepEqual(state.get(b).watchlist, ['OTHER']);
  await state.setWatchlist(a, ['DEMO']);
  now = Date.parse('2026-02-28T12:00:00Z'); state.purge();
  assert.deepEqual(state.get(a).watchlist, []); assert.deepEqual(state.get(b).watchlist, []);
  assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')).members, {});
});
test('member state refuses unsafe permissions, write failure cannot change another member', async t => {
  const { MemberState } = require('../server/member/state');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-member-atomic-')); t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'state.json'), state = new MemberState(file, ''), key = 'a'.repeat(64);
  await state.setWatchlist(key, ['DEMO']); const before = state.get(key);
  state.file = path.join(dir, 'missing-parent', 'state.json');
  await assert.rejects(state.setWatchlist(key, ['OTHER'])); assert.deepEqual(state.get(key), before);
  fs.chmodSync(file, 0o644); assert.throws(() => new MemberState(file, ''), /permissions/);
});
test('member bodies are bounded and malformed JSON media are rejected without leaking content', async t => {
  const s = await setup(t), auth = await s.login();
  const r = await s.request('/api/member/v1/watchlist', { ...auth, method: 'PUT', data: { symbols: ['DEMO'], extra: 'x'.repeat(17000) } });
  assert.equal(r.status, 413); assert.ok(!(await r.text()).includes('xxxxxxxx'));
});
