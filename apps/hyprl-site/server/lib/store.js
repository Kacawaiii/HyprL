'use strict';
// Stockage des inscriptions dans un fichier JSON (écriture atomique, droits 600).
// Suffisant pour une liste d'accès anticipé ; à remplacer par une base de données au-delà de quelques milliers d'inscrits.

const fs = require('node:fs');
const path = require('node:path');

class SubscriberStore {
  constructor(file) {
    this.file = file;
    this.items = [];
    this.queue = Promise.resolve();
  }

  load() {
    fs.mkdirSync(path.dirname(this.file), { recursive: true, mode: 0o700 });
    if (fs.existsSync(this.file)) {
      const raw = fs.readFileSync(this.file, 'utf8');
      const data = raw.trim() ? JSON.parse(raw) : { subscribers: [] };
      this.items = Array.isArray(data.subscribers) ? data.subscribers : [];
    } else {
      this.items = [];
      this._writeSync();
    }
    return this;
  }

  _writeSync() {
    const tmp = this.file + '.' + process.pid + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify({ subscribers: this.items }, null, 2), { mode: 0o600 });
    fs.renameSync(tmp, this.file);
  }

  /** Sérialise les écritures pour éviter qu'elles se chevauchent. */
  _persist() {
    this.queue = this.queue.then(() => this._writeSync());
    return this.queue;
  }

  all() {
    return this.items.slice();
  }

  findByEmail(email) {
    return this.items.find((s) => s.email === email) || null;
  }

  findByTokenHash(hash) {
    return this.items.find((s) => s.unsubscribe_token_hash === hash) || null;
  }

  async add(record) {
    this.items.push(record);
    await this._persist();
    return record;
  }

  /** Effacement définitif. Retourne true si l'inscrit existait. */
  async remove(id) {
    const before = this.items.length;
    this.items = this.items.filter((s) => s.id !== id);
    if (this.items.length === before) return false;
    await this._persist();
    return true;
  }
}

module.exports = { SubscriberStore };
