'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { expiresAt } = require('./retention');

class SubscriberStore {
  constructor(file) {
    this.file = file;
    this.items = [];
    this.queue = Promise.resolve();
  }
  load() {
    fs.mkdirSync(path.dirname(this.file), { recursive: true, mode: 0o700 });
    if (fs.existsSync(this.file)) {
      const data = JSON.parse(fs.readFileSync(this.file, 'utf8'));
      if (!Array.isArray(data.subscribers)) throw new Error('Invalid subscriber store');
      for (const record of data.subscribers) expiresAt(record.created_at);
      this.items = data.subscribers;
    } else this._writeSync(this.items);
    return this;
  }
  _writeSync(items) {
    const tmp = this.file + '.' + process.pid + '.tmp';
    try {
      fs.writeFileSync(tmp, JSON.stringify({ subscribers: items }, null, 2), { mode: 0o600 });
      fs.chmodSync(tmp, 0o600);
      fs.renameSync(tmp, this.file);
    } finally {
      fs.rmSync(tmp, { force: true });
    }
  }
  _mutate(change) {
    const operation = this.queue.then(() => {
      const { items, result } = change(this.items);
      if (items !== this.items) {
        this._writeSync(items);
        this.items = items;
      }
      return result;
    });
    this.queue = operation.catch(() => {});
    return operation;
  }
  all() { return this.items.slice(); }
  findByEmail(email) { return this.items.find((s) => s.email === email) || null; }
  findById(id) { return this.items.find((s) => s.id === id) || null; }
  findByTokenHash(hash) { return this.items.find((s) => s.unsubscribe_token_hash === hash) || null; }
  add(record) {
    return this._mutate((items) => items.some((s) => s.email === record.email)
      ? { items, result: null } : { items: [...items, record], result: record });
  }
  update(id, change) {
    return this._mutate((items) => {
      const current = items.find(s => s.id === id);
      if (!current) return { items, result: null };
      const next = change(structuredClone(current));
      if (!next) return { items, result: null };
      return { items: items.map(s => s.id === id ? next : s), result: next };
    });
  }
  remove(id) {
    return this._mutate((items) => {
      const kept = items.filter((s) => s.id !== id);
      return { items: kept.length === items.length ? items : kept, result: kept.length !== items.length };
    });
  }
  purgeExpiredSync(now) {
    const kept = this.items.filter((s) => Date.parse(expiresAt(s.created_at)) > now);
    const removed = this.items.length - kept.length;
    if (removed) { this._writeSync(kept); this.items = kept; }
    return removed;
  }
  purgeExpired(now) {
    return this._mutate((items) => {
      const kept = items.filter((s) => Date.parse(expiresAt(s.created_at)) > now);
      return { items: kept.length === items.length ? items : kept, result: items.length - kept.length };
    });
  }
}
module.exports = { SubscriberStore };
