'use strict';
// Authentification admin : comparaison en temps constant, limitation des tentatives, sessions à expiration d'inactivité.

const crypto = require('node:crypto');

const sha256 = (s) => crypto.createHash('sha256').update(String(s), 'utf8').digest();

function passwordMatches(candidate, expected) {
  if (typeof candidate !== 'string' || candidate.length === 0 || candidate.length > 1024) return false;
  return crypto.timingSafeEqual(sha256(candidate), sha256(expected));
}

/** Fenêtre glissante : au plus `max` échecs par `windowMs` et par clé (adresse IP). */
class AttemptLimiter {
  constructor({ max, windowMs, now = Date.now }) {
    this.max = max;
    this.windowMs = windowMs;
    this.now = now;
    this.hits = new Map();
  }

  _recent(key) {
    const t = this.now();
    const list = (this.hits.get(key) || []).filter((ts) => t - ts < this.windowMs);
    if (list.length) this.hits.set(key, list); else this.hits.delete(key);
    return list;
  }

  /** null si autorisé, sinon nombre de secondes avant de pouvoir réessayer. */
  blockedFor(key) {
    const list = this._recent(key);
    if (list.length < this.max) return null;
    return Math.max(1, Math.ceil((list[0] + this.windowMs - this.now()) / 1000));
  }

  fail(key) {
    const list = this._recent(key);
    list.push(this.now());
    this.hits.set(key, list);
    return Math.max(0, this.max - list.length);
  }

  reset(key) {
    this.hits.delete(key);
  }

  sweep() {
    for (const key of [...this.hits.keys()]) this._recent(key);
  }
}

/** Sessions en mémoire. Seule l'empreinte du jeton est conservée côté serveur. */
class SessionStore {
  constructor({ idleMs, maxMs, now = Date.now }) {
    this.idleMs = idleMs;
    this.maxMs = maxMs;
    this.now = now;
    this.sessions = new Map();
  }

  create() {
    const token = crypto.randomBytes(32).toString('base64url');
    const t = this.now();
    this.sessions.set(sha256(token).toString('hex'), { created: t, lastSeen: t });
    return token;
  }

  /** 'ok' (et prolonge la session), 'expired' ou 'none'. */
  touch(token) {
    if (!token) return 'none';
    const key = sha256(token).toString('hex');
    const s = this.sessions.get(key);
    if (!s) return 'none';
    const t = this.now();
    if (t - s.lastSeen > this.idleMs || t - s.created > this.maxMs) {
      this.sessions.delete(key);
      return 'expired';
    }
    s.lastSeen = t;
    return 'ok';
  }

  destroy(token) {
    if (token) this.sessions.delete(sha256(token).toString('hex'));
  }

  sweep() {
    const t = this.now();
    for (const [k, s] of this.sessions) {
      if (t - s.lastSeen > this.idleMs || t - s.created > this.maxMs) this.sessions.delete(k);
    }
  }
}

module.exports = { passwordMatches, AttemptLimiter, SessionStore, sha256 };
