'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { random, fail, exact } = require('./security');
const { expiresAt } = require('../lib/retention');
function watchlist(value) {
  if (!Array.isArray(value) || value.length > 50 || value.some(x => typeof x !== 'string' || !/^[A-Z][A-Z0-9.\-]{0,14}$/.test(x))) throw fail(400, 'invalid_watchlist');
  return [...new Set(value)].sort();
}
class MemberState {
  constructor(file, billingOrigin, now = Date.now) {
    Object.assign(this, { file, billingOrigin, now }); this.queue = Promise.resolve();
    fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
    if (fs.existsSync(file)) {
      const st = fs.lstatSync(file);
      if (!st.isFile() || st.isSymbolicLink() || st.uid !== process.getuid() || (st.mode & 0o077) || st.size > 16 * 1024 * 1024) throw Error('Unsafe member state permissions');
      this.data = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (this.data.version !== 1 || typeof this.data.members !== 'object' || !this.data.members || Array.isArray(this.data.members)) throw Error('Invalid member state');
      for (const [key, row] of Object.entries(this.data.members)) {
        if (!/^[a-f0-9]{64}$/.test(key)) throw Error('Invalid member state');
        exact(row, ['watchlist', 'billing', 'modified_at']); watchlist(row.watchlist); expiresAt(row.modified_at);
      }
    } else this.data = { version: 1, members: {} };
    this.purge();
  }
  persist(data) {
    const raw = JSON.stringify(data);
    if (Buffer.byteLength(raw) > 16 * 1024 * 1024) throw fail(503, 'state_full');
    const temp = this.file + '.' + random();
    try {
      const fd = fs.openSync(temp, 'wx', 0o600);
      try { fs.writeFileSync(fd, raw); fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
      fs.renameSync(temp, this.file);
      this.data = data;
    } finally { fs.rmSync(temp, { force: true }); }
  }
  purge() {
    const expired = Object.entries(this.data.members).filter(([, row]) => Date.parse(expiresAt(row.modified_at)) <= this.now()).map(([key]) => key);
    if (!expired.length) return;
    const data = structuredClone(this.data);
    for (const key of expired) delete data.members[key];
    this.persist(data);
  }
  get(key) { this.purge(); return structuredClone(this.data.members[key] || { watchlist: [], billing: null }); }
  mutate(key, fn) {
    if (!/^[a-f0-9]{64}$/.test(key)) throw fail(400, 'invalid_member');
    const op = this.queue.then(() => {
      this.purge();
      const data = structuredClone(this.data), row = data.members[key] || { watchlist: [], billing: null };
      fn(row); row.modified_at = new Date(this.now()).toISOString(); data.members[key] = row;
      if (Object.keys(data.members).length > 10000) throw fail(503, 'state_full');
      this.persist(data); return this.get(key);
    });
    this.queue = op.catch(() => {}); return op;
  }
  setWatchlist(key, symbols) { const values = watchlist(symbols); return this.mutate(key, row => { row.watchlist = values; }); }
  // Private rights hook; caller resolves the verified identity, never restores erased state.
  remove(key) {
    if (!/^[a-f0-9]{64}$/.test(key)) throw fail(400, 'invalid_member');
    const op = this.queue.then(() => { const data = structuredClone(this.data); delete data.members[key]; this.persist(data); });
    this.queue = op.catch(() => {}); return op;
  }
  // Private integration hook only. Never grants a tier: Discord roles remain authoritative.
  applyBillingUpdate(key, billing) {
    exact(billing, ['renewal_at', 'manage_url', 'cancel_url']);
    if (billing.renewal_at !== null && (!/^\d{4}-\d{2}-\d{2}T.*Z$/.test(billing.renewal_at) || !Number.isFinite(Date.parse(billing.renewal_at)))) throw fail(400, 'invalid_billing');
    for (const field of ['manage_url', 'cancel_url']) {
      let url; try { if (typeof billing[field] !== 'string' || billing[field].length > 2048) throw Error(); url = new URL(billing[field]); } catch { throw fail(400, 'invalid_billing'); }
      if (!this.billingOrigin || url.origin !== this.billingOrigin || url.username || url.password) throw fail(400, 'invalid_billing');
    }
    return this.mutate(key, row => { row.billing = structuredClone(billing); });
  }
}
module.exports = { MemberState, watchlist };
