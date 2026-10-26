'use strict';
const fs = require('node:fs');
const path = require('node:path');
const Ajv = require('ajv');
const addFormats = require('ajv-formats');
const { hash, fail } = require('./security');
const schema = require('../../docs/member/snapshot-v1.schema.json');
const ajv = new Ajv({ strict: false }); addFormats(ajv);
const valid = ajv.compile(schema);
const canonical = obj => JSON.stringify(Object.fromEntries(Object.entries(obj).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)));
class Snapshots {
  constructor(directory, { now = Date.now, allowSynthetic = false } = {}) { Object.assign(this, { directory, now, allowSynthetic }); }
  read() {
    if (!this.directory) throw fail(503, 'snapshot_unavailable');
    let fd;
    try {
      if (fs.realpathSync(this.directory) !== path.resolve(this.directory)) throw Error();
      fd = fs.openSync(path.join(this.directory, 'member-v1.json'), fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
      const stat = fs.fstatSync(fd);
      if (!stat.isFile() || stat.size > 2 * 1024 * 1024 || (stat.mode & 0o022)) throw Error();
      const value = JSON.parse(fs.readFileSync(fd, 'utf8'));
      if (!valid(value) || (value.provenance !== 'observed' && !this.allowSynthetic) || Date.parse(value.generated_at) > this.now()) throw Error();
      for (const r of value.forecasts) {
        const c = r.commitment;
        if (Date.parse(c.reveal_at) > this.now() || Date.parse(c.committed_at) >= Date.parse(c.reveal_at) || Date.parse(c.reveal_at) > Date.parse(r.scored_at) || Date.parse(r.scored_at) > Date.parse(value.generated_at) || hash(canonical(c)) !== r.sha256) throw Error();
      }
      return value;
    } catch { throw fail(503, 'snapshot_unavailable'); }
    finally { if (fd !== undefined) fs.closeSync(fd); }
  }
}
module.exports = { Snapshots, canonical };
