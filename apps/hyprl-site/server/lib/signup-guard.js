'use strict';
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const HOUR = 3600000, DAY = 24 * HOUR;
const LIMITS = Object.freeze({ minMs: 3000, ttlMs: 10 * 60000, bits: 16, ipHour: 5, globalHour: 30, mailDay: 30, challengeIpHour: 20, challengeGlobalHour: 300 });
const hash = v => crypto.createHash('sha256').update(v).digest('hex');

// One writer (the site process). Only keyed digests, timestamps and counters;
// never a request body, address, IP, bearer token or SMTP response.
class SignupGuard {
  constructor(file, key, now = Date.now) {
    this.file = file; this.key = key; this.now = now; this.queue = Promise.resolve();
    this.keyId = this.mac('key-binding');
    this.state = { version: 1, key_id: this.keyId, attempts: [], challenges: [], spent: {}, addresses: {}, digests: [] };
    fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
    if (fs.existsSync(file)) {
      const stat = fs.lstatSync(file);
      if (!stat.isFile() || stat.isSymbolicLink() || (stat.mode & 0o077)) throw Error('Private regular signup guard file required');
      this.state = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (this.state.version !== 1 || this.state.key_id !== this.keyId || !Array.isArray(this.state.attempts) || !Array.isArray(this.state.challenges) || !Array.isArray(this.state.digests) || !this.state.spent || Array.isArray(this.state.spent) || !this.state.addresses || Array.isArray(this.state.addresses)) throw Error('Invalid signup guard state/key binding');
    } else this.write(this.state);
  }
  mac(value) { return crypto.createHmac('sha256', this.key).update(value).digest('hex'); }
  write(state) {
    const tmp = this.file + '.' + process.pid + '.tmp';
    try {
      fs.writeFileSync(tmp, JSON.stringify(state), { mode: 0o600, flag: 'wx' });
      const fd = fs.openSync(tmp, 'r');
      try { fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
      fs.renameSync(tmp, this.file);
      const directory = fs.openSync(path.dirname(this.file), 'r');
      try { fs.fsyncSync(directory); } finally { fs.closeSync(directory); }
    } finally { fs.rmSync(tmp, { force: true }); }
  }
  mutate(change) {
    const operation = this.queue.then(() => {
      const s = structuredClone(this.state), t = this.now();
      s.attempts = s.attempts.filter(x => x.at > t - HOUR);
      s.challenges = s.challenges.filter(x => x.at > t - HOUR);
      for (const [k, expiry] of Object.entries(s.spent)) if (expiry <= t) delete s.spent[k];
      // Suppression digests last at most 24 calendar months, even after personal
      // pending records are erased. They prevent another confirmation email.
      for (const [k, value] of Object.entries(s.addresses)) if (value.expires <= t) delete s.addresses[k];
      s.digests = s.digests.filter(x => x > t - 2 * DAY);
      const result = change(s, t);
      this.write(s); this.state = s;
      return result;
    });
    this.queue = operation.catch(() => {});
    return operation;
  }
  async challenge(ip) {
    const ipHash = this.mac('ip:' + ip);
    const allowed = await this.mutate((s, at) => {
      if (s.challenges.length >= LIMITS.challengeGlobalHour || s.challenges.filter(x => x.ip === ipHash).length >= LIMITS.challengeIpHour) return false;
      s.challenges.push({ ip: ipHash, at }); return true;
    });
    if (!allowed) return null;
    const payload = Buffer.from(JSON.stringify({ v: 1, at: this.now(), ip: ipHash, nonce: crypto.randomBytes(16).toString('hex') })).toString('base64url');
    return { token: payload + '.' + this.mac('form:' + payload), min_ms: LIMITS.minMs, expires_ms: LIMITS.ttlMs, bits: LIMITS.bits };
  }
  verifyReason(ip, token, work) {
    if (!token) return { reason: 'no_token' };
    if (typeof token !== 'string' || token.length > 512 || !/^[A-Za-z0-9_-]+\.[a-f0-9]{64}$/.test(token)) return { reason: 'invalid_token' };
    const [payload, signature] = token.split('.');
    if (!crypto.timingSafeEqual(Buffer.from(signature, 'hex'), Buffer.from(this.mac('form:' + payload), 'hex'))) return { reason: 'invalid_token' };
    let p;
    try { p = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8')); } catch { return { reason: 'invalid_token' }; }
    const age = this.now() - p.at;
    if (p.v !== 1 || !Number.isSafeInteger(p.at) || age < 0) return { reason: 'invalid_token' };
    if (age < LIMITS.minMs) return { reason: 'too_fast' };
    if (age >= LIMITS.ttlMs) return { reason: 'expired_token' };
    if (p.ip !== this.mac('ip:' + ip)) return { reason: 'wrong_ip' };
    if (!/^\d{1,10}$/.test(work || '') || !hash(token + ':' + work).startsWith('0000')) return { reason: 'work' };
    return { proof: { id: hash(token), expires: p.at + LIMITS.ttlMs, ip: p.ip } };
  }
  verify(ip, token, work) { return this.verifyReason(ip, token, work).proof || null; }
  async consumeReason(proof) {
    if (!proof) return 'invalid_token';
    return this.mutate((s, at) => {
      if (proof.expires <= at) return 'expired_token';
      if (s.spent[proof.id]) return 'replay';
      if (s.attempts.length >= LIMITS.globalHour || s.attempts.filter(x => x.ip === proof.ip).length >= LIMITS.ipHour) return 'cap';
      s.spent[proof.id] = proof.expires; s.attempts.push({ ip: proof.ip, at }); return null;
    });
  }
  async consume(proof) { return !await this.consumeReason(proof); }
  reserveAddressReason(email) {
    const id = this.mac('address:' + email);
    return this.mutate((s, at) => {
      if (s.addresses[id]) return 'duplicate';
      if (Object.values(s.addresses).filter(x => x.at > at - DAY).length >= LIMITS.mailDay) return 'mail_cap';
      const { expiresAt } = require('./retention');
      s.addresses[id] = { at, expires: Date.parse(expiresAt(new Date(at).toISOString())) }; return null;
    });
  }
  async reserveAddress(email) { return !await this.reserveAddressReason(email); }
  reserveDigest() {
    return this.mutate((s, at) => {
      if (s.digests.some(x => x > at - DAY)) return false;
      s.digests.push(at); return true;
    });
  }
}
module.exports = { SignupGuard, LIMITS };
