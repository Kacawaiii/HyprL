'use strict';
const crypto = require('node:crypto');
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const random = () => crypto.randomBytes(32).toString('base64url');
const fail = (status, code) => Object.assign(new Error(code), { status });
const TIERS = { early: 0, essentiel: 1, premium: 2, fondateur: 3, equipe: 4 };
const LABELS = { early: 'Accès anticipé', essentiel: 'Essentiel', premium: 'Premium', fondateur: 'Fondateur', equipe: 'Equipe' };
const FEATURES = { dashboard: 0, briefs: 1, radar: 1, calendar: 1, duel: 2, luck: 2, screener: 2, claims: 0, alerts: 2, account: 0, watchlist: 1, proofs: 2 };
function tierFor(roles, mapping) {
  return Object.keys(TIERS).reverse().find(t => roles.includes(mapping[t])) || null;
}
function memberKey(user, config) {
  return crypto.createHmac('sha256', config.identityKey).update(`${config.guildId}/${user}`).digest('hex');
}
function cookie(req, name) {
  const values = (req.headers.cookie || '').split(';').map(x => x.trim()).filter(x => x.startsWith(name + '='));
  return values.length === 1 ? values[0].slice(name.length + 1) : '';
}
function cookieValue(name, value, secure, age) {
  return `${name}=${value}; Path=/; HttpOnly; SameSite=Strict${secure ? '; Secure' : ''}${age === undefined ? '' : '; Max-Age=' + age}`;
}
class Budget {
  constructor(max, windowMs, now = Date.now) { Object.assign(this, { max, windowMs, now }); this.keys = new Map(); }
  take(key) {
    const t = this.now();
    for (const [k, v] of this.keys) if (t >= v.until) this.keys.delete(k);
    let v = this.keys.get(key);
    if (!v) {
      if (this.keys.size >= 10000) throw fail(429, 'rate_limited');
      this.keys.set(key, v = { count: 0, until: t + this.windowMs });
    }
    if (++v.count > this.max) throw fail(429, 'rate_limited');
  }
}
class Sessions {
  constructor(now, limits) { Object.assign(this, { now, limits }); this.items = new Map(); }
  sweep() {
    for (const [k, s] of this.items) if (this.now() - s.seen >= this.limits.idleMs || this.now() - s.created >= this.limits.maxMs) this.items.delete(k);
  }
  create(identity) {
    this.sweep();
    if (this.items.size >= 10000) throw fail(503, 'sessions_full');
    const token = random();
    this.items.set(hash(token), { ...identity, created: this.now(), seen: this.now(), csrf: random() });
    return token;
  }
  get(token) {
    this.sweep();
    const s = token && this.items.get(hash(token));
    if (!s) throw fail(401, 'session_expired');
    s.seen = this.now(); return s;
  }
  destroy(token) { if (token) this.items.delete(hash(token)); }
}
async function body(req) {
  if (req.headers['content-type']?.split(';')[0] !== 'application/json') throw fail(415, 'json_required');
  const raw = await new Promise((resolve, reject) => {
    let size = 0, oversized = false; const chunks = [];
    req.on('data', chunk => {
      size += chunk.length;
      if (oversized) return;
      if (size > 16384) { oversized = true; chunks.length = 0; reject(fail(413, 'body_too_large')); return; }
      chunks.push(chunk);
    });
    req.on('end', () => { if (!oversized) resolve(Buffer.concat(chunks).toString('utf8')); });
    req.on('error', () => reject(fail(400, 'invalid_body')));
  });
  try {
    const result = JSON.parse(raw);
    if (!result || Array.isArray(result) || typeof result !== 'object') throw Error();
    return result;
  } catch { throw fail(400, 'invalid_json'); }
}
function exact(value, keys) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).sort().join(',') !== [...keys].sort().join(',')) throw fail(400, 'invalid_fields');
}
module.exports = { hash, random, fail, TIERS, LABELS, FEATURES, tierFor, memberKey, cookie, cookieValue, Budget, Sessions, body, exact };
