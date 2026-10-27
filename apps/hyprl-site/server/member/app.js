'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { Discord } = require('./discord');
const { MemberState, watchlist } = require('./state');
const { Snapshots } = require('./snapshots');
const { RouterClient } = require('./router-client');
const { hash, random, fail, TIERS, LABELS, FEATURES, tierFor, memberKey, cookie, cookieValue, Budget, Sessions, body, exact } = require('./security');
const ROUTES = { '': 'dashboard', brief: 'briefs', radar: 'radar', calendrier: 'calendar', duel: 'duel', chance: 'luck', screener: 'screener', preuves: 'proofs', 'chance-ou-pas': 'claims', alertes: 'alerts', compte: 'account' };
function createMemberApp(site, deps = {}) {
  const config = site.member || { enabled: false }, now = deps.now || Date.now;
  const origin = new URL(site.publicUrl).origin;
  function clientIp(req) {
    const remote = req.socket.remoteAddress || 'local';
    const real = req.headers['x-real-ip'];
    return site.trustProxy && ['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(remote) && require('node:net').isIP(real || '') ? real : remote;
  }
  const secure = site.secureCookies;
  const SESSION = secure ? '__Host-hyprl_member' : 'hyprl_member';
  const FLOW = secure ? '__Host-hyprl_oauth' : 'hyprl_oauth';
  const sessions = new Sessions(now, site.session), flows = new Map();
  const ipBudget = new Budget(240, 60000, now), userBudget = new Budget(120, 60000, now), oauthBudget = new Budget(10, 900000, now);
  const discord = config.enabled ? (deps.discord || new Discord(config, site.publicUrl)) : null;
  let state = null;
  try { state = config.enabled ? (deps.state || new MemberState(config.stateFile, config.billingOrigin, now)) : null; }
  catch { throw Error('Member state unavailable'); }
  const snapshots = config.enabled ? (deps.snapshots || new Snapshots(config.snapshotDir, { now })) : null;
  const router = config.enabled ? (deps.router || new RouterClient(config)) : null;
  const view = name => fs.readFileSync(path.join(__dirname, 'views', name + '.html'), 'utf8');
  const html = (res, status, content) => { res.writeHead(status, { 'Content-Type': 'text/html; charset=utf-8' }); res.end(content); };
  function json(res, status, value) {
    if (status >= 400) value = { type: 'about:blank', title: require('node:http').STATUS_CODES[status], status, detail: status === 503 ? 'En cours de constitution ou service indisponible. Réessayez plus tard.' : 'Accès ou requête refusé.', ...value };
    res.writeHead(status, { 'Content-Type': status >= 400 ? 'application/problem+json' : 'application/json', ...(status === 429 ? { 'Retry-After': '60' } : {}) }); res.end(JSON.stringify(value));
  }
  function setCookie(res, name, value, age) { res.setHeader('Set-Cookie', cookieValue(name, value, secure, age)); }
  function redirect(res, location) { res.writeHead(303, { Location: location }); res.end(); }
  function sameOrigin(req) { if (req.headers.origin !== origin) throw fail(403, 'origin_required'); }
  function sweep() {
    sessions.sweep();
    if (state) { try { state.purge(); } catch { /* reads/mutations retry and fail closed */ } }
    for (const [k, v] of flows) if (v.until <= now()) flows.delete(k);
  }
  const sweeper = setInterval(sweep, 60000); sweeper.unref();
  async function authenticate(req, feature) {
    const token = cookie(req, SESSION), s = sessions.get(token);
    userBudget.take(s.key);
    try {
      if (s.expiresAt <= now() + 30000) {
        if (!s.refreshing) s.refreshing = discord.refresh(s.refreshToken).then(t => {
          s.accessToken = t.access_token; s.refreshToken = t.refresh_token; s.expiresAt = now() + t.expires_in * 1000;
        }).finally(() => { delete s.refreshing; });
        await s.refreshing;
      }
      const roles = await discord.roles(s.accessToken);
      s.tier = tierFor(roles, config.roles);
      if (!s.tier) throw fail(403, 'membership_required');
    } catch (e) {
      if (e.status === 401 || e.status === 403) sessions.destroy(token);
      throw e;
    }
    if (!(feature in FEATURES) || TIERS[s.tier] < FEATURES[feature]) throw fail(403, 'tier_required');
    return s;
  }
  async function api(req, res, parts) {
    const feature = parts[0] === 'session' ? 'dashboard' : parts[0];
    // Unknown paths are authenticated too; no namespace or tenant discovery.
    const s = await authenticate(req, feature in FEATURES ? feature : 'dashboard');
    if (parts.length !== 1 && parts[0] !== 'proofs') throw fail(404, 'not_found');
    if (!(feature in FEATURES)) throw fail(404, 'not_found');
    if (req.method !== 'GET') {
      sameOrigin(req);
      if (req.headers['x-csrf-token'] !== s.csrf) throw fail(403, 'csrf_required');
    }
    const allowed = feature === 'watchlist' ? ['GET', 'PUT'] : feature === 'alerts' ? ['GET', 'POST', 'PATCH', 'DELETE'] : parts[0] === 'logout' ? ['POST'] : ['GET'];
    if (!allowed.includes(req.method)) throw fail(405, 'method_not_allowed');
    if (parts[0] === 'session') return json(res, 200, { tier: s.tier, tier_label: LABELS[s.tier], csrf: s.csrf, features: Object.keys(FEATURES).filter(f => TIERS[s.tier] >= FEATURES[f]) });
    if (feature === 'account') return json(res, 200, { tier: s.tier, tier_label: LABELS[s.tier], billing: state.get(s.key).billing });
    if (feature === 'watchlist') {
      if (parts.length !== 1) throw fail(404, 'not_found');
      if (req.method === 'GET') return json(res, 200, { symbols: state.get(s.key).watchlist });
      const input = await body(req); exact(input, ['symbols']);
      await state.setWatchlist(s.key, input.symbols);
      return json(res, 200, { symbols: state.get(s.key).watchlist });
    }
    if (feature === 'alerts') {
      const customer = `${config.guildId}/${s.user}`;
      if (parts.length !== 1) throw fail(404, 'not_found');
      if (req.method === 'GET') return json(res, 200, await router.call(customer, 'list'));
      const input = await body(req);
      if (req.method === 'POST') {
        exact(input, ['channel', 'config', 'kinds', 'watchlist']);
        if (!['discord', 'slack', 'telegram', 'teams', 'webhook'].includes(input.channel) || !Array.isArray(input.kinds) || !input.kinds.length || input.kinds.some(k => !['sec_filing', 'fed_release', 'cpi_release', 'nfp_release', 'earnings_date', 'earnings_result', 'radar_headline', 'duel_hashes', 'duel_results', 'weekly_luck'].includes(k))) throw fail(400, 'invalid_channel');
        input.watchlist = watchlist(input.watchlist);
        return json(res, 200, await router.call(customer, 'add', input));
      }
      exact(input, req.method === 'PATCH' ? ['destination', 'paused'] : ['destination']);
      if (!/^[a-f0-9]{32}$/.test(input.destination) || (req.method === 'PATCH' && typeof input.paused !== 'boolean')) throw fail(400, 'invalid_destination');
      return json(res, 200, await router.call(customer, req.method === 'PATCH' ? 'pause' : 'remove', input));
    }
    if (feature === 'dashboard') {
      let data = null;
      try { data = snapshots.read(); } catch {}
      return json(res, 200, { generated_at: data?.generated_at || null, provenance: data?.provenance || null, stale: data ? now() - Date.parse(data.generated_at) > 26 * 3600000 : true });
    }
    const data = snapshots.read();
    const meta = { schema_version: data.schema_version, generated_at: data.generated_at, provenance: data.provenance, stale: now() - Date.parse(data.generated_at) > 26 * 3600000 };
    const filtered = rows => { const symbols = new Set(state.get(s.key).watchlist); return rows.filter(r => ['fed', 'cpi', 'nfp'].includes(r.kind) || symbols.has(r.symbol)); };
    if (feature === 'briefs') return json(res, 200, { ...meta, items: data.briefs });
    if (feature === 'radar') return json(res, 200, { ...meta, items: filtered(data.radar) });
    if (feature === 'calendar') return json(res, 200, { ...meta, items: filtered(data.calendar) });
    if (feature === 'duel') return json(res, 200, { ...meta, baseline: data.baseline, forecasts: data.forecasts, failures: data.failures, reveals: data.reveals, series: data.series });
    if (feature === 'luck') return json(res, 200, { ...meta, series: [] });
    if (feature === 'screener') return json(res, 200, { ...meta, items: data.screener });
    if (feature === 'claims') return json(res, 200, { ...meta, items: data.claims });
    if (feature === 'proofs') {
      if (parts.length === 1) return json(res, 200, { ...meta, items: data.reveals });
      const row = parts.length === 2 && data.reveals.find(r => r.id === parts[1]);
      if (!row) throw fail(404, 'not_found');
      return json(res, 200, { ...meta, reveal: row });
    }
    throw fail(404, 'not_found');
  }
  async function handle(req, res, url) {
    const p = url.pathname;
    if (!(p === '/membre' || p.startsWith('/membre/') || p.startsWith('/auth/discord') || p.startsWith('/api/member/'))) return false;
    res.setHeader('Cache-Control', 'no-store'); res.setHeader('X-Robots-Tag', 'noindex, nofollow');
    // OAuth code/state must never leave this origin, including on error documents.
    res.setHeader('Referrer-Policy', 'no-referrer');
    try {
      ipBudget.take(clientIp(req));
      if (!config.enabled) { if (p.startsWith('/api/')) json(res, 503, { error: 'member_unavailable' }); else html(res, 503, view('login')); return true; }
      if (p === '/membre/connexion' && req.method === 'GET') { html(res, 200, view('login')); return true; }
      if (p === '/auth/discord/start' && req.method === 'GET') {
        oauthBudget.take(clientIp(req)); sweep();
        if (flows.size >= 10000) throw fail(429, 'rate_limited');
        const token = random(), stateToken = random(), verifier = random();
        flows.set(hash(token), { state: hash(stateToken), verifier, until: now() + 300000 });
        setCookie(res, FLOW, token, 300);
        const target = new URL('https://discord.com/oauth2/authorize');
        target.search = new URLSearchParams({ client_id: config.clientId, redirect_uri: site.publicUrl + '/auth/discord/callback', response_type: 'code', scope: 'identify guilds.members.read', state: stateToken, code_challenge: crypto.createHash('sha256').update(verifier).digest('base64url'), code_challenge_method: 'S256', prompt: 'consent' });
        redirect(res, target.href); return true;
      }
      if (p === '/auth/discord/callback' && req.method === 'GET') { html(res, 200, view('callback')); return true; }
      if (p === '/auth/discord/complete' && req.method === 'POST') {
        sameOrigin(req); oauthBudget.take(clientIp(req));
        const input = await body(req); exact(input, ['code', 'state']);
        const flowKey = hash(cookie(req, FLOW)); const flow = flows.get(flowKey);
        if (!flow || flow.until <= now() || typeof input.state !== 'string' || hash(input.state) !== flow.state || typeof input.code !== 'string' || !input.code.length || input.code.length > 2048) throw fail(401, 'oauth_state_invalid');
        flows.delete(flowKey); // Consume before network; replay and concurrent redemption fail closed.
        const t = await discord.exchange(input.code, flow.verifier);
        const user = await discord.user(t.access_token);
        if (!/^\d{17,20}$/.test(user.id || '')) throw fail(401, 'oauth_identity_invalid');
        const tier = tierFor(await discord.roles(t.access_token), config.roles);
        if (!tier) throw fail(403, 'membership_required');
        sessions.destroy(cookie(req, SESSION));
        const token = sessions.create({ user: user.id, key: memberKey(user.id, config), tier, accessToken: t.access_token, refreshToken: t.refresh_token, expiresAt: now() + t.expires_in * 1000 });
        res.setHeader('Set-Cookie', [cookieValue(SESSION, token, secure), cookieValue(FLOW, '', secure, 0)]);
        json(res, 200, { ok: true }); return true;
      }
      if (p === '/api/member/v2/logout' && req.method === 'POST') {
        const s = sessions.get(cookie(req, SESSION)); sameOrigin(req);
        if (req.headers['x-csrf-token'] !== s.csrf) throw fail(403, 'csrf_required');
        sessions.destroy(cookie(req, SESSION)); setCookie(res, SESSION, '', 0);
        json(res, 200, { ok: true }); return true;
      }
      if (p.startsWith('/api/member/v2/')) { await api(req, res, p.slice('/api/member/v2/'.length).split('/')); return true; }
      if (p === '/membre' || p.startsWith('/membre/')) {
        if (req.method !== 'GET') throw fail(405, 'method_not_allowed');
        const route = p === '/membre' ? '' : p.slice('/membre/'.length);
        const feature = ROUTES[route];
        if (!feature) throw fail(404, 'not_found');
        try { await authenticate(req, feature); }
        catch (e) {
          if (e.status === 401) { redirect(res, '/membre/connexion'); return true; }
          if (e.message !== 'tier_required') throw e;
          html(res, 403, view('member')); return true;
        }
        html(res, 200, view('member')); return true;
      }
      throw fail(404, 'not_found');
    } catch (e) {
      json(res, e.status || 503, { error: e.status ? e.message : 'member_unavailable' }); return true;
    }
  }
  return { handle, sessions, flows, state, close: () => clearInterval(sweeper) };
}
module.exports = { createMemberApp, ROUTES };
