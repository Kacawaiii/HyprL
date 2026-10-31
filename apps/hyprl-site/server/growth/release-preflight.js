'use strict';
// Read-only release check. Never constructs the app, subscriber store or mailer.
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const { loadConfig } = require('../config');
const { Snapshots } = require('../member/snapshots');
function check(config, liveRoot, now = Date.now()) {
  assert.equal(config.production, true);
  assert.equal(config.host, '127.0.0.1');
  assert.equal(config.port, 3000);
  assert.equal(config.publicUrl, 'https://hyprlcore.com');
  assert.equal(config.member.enabled, true);
  const roots = [fs.realpathSync(config.root), fs.realpathSync(liveRoot)];
  for (const value of [config.dataFile, config.member.stateFile, config.member.snapshotDir, config.publicSnapshotDir]) {
    assert.ok(path.isAbsolute(value), 'Absolute external state paths required');
    // Resolve existing ancestors too, preventing a parent symlink into a release.
    let ancestor = value;
    while (!fs.existsSync(ancestor)) ancestor = path.dirname(ancestor);
    const resolved = path.join(fs.realpathSync(ancestor), path.relative(ancestor, value));
    for (const root of roots) assert.ok(resolved !== root && !resolved.startsWith(root + path.sep), 'State must remain outside both code releases');
  }
  const v = new Snapshots(config.member.snapshotDir, { now: () => now }).read();
  const publicValue = new Snapshots(config.publicSnapshotDir, { now: () => now }).read();
  for (const value of [v, publicValue]) {
    assert.equal(value.schema_version, 2);
    assert.equal(value.provenance, 'observed');
    assert.ok(now - Date.parse(value.generated_at) >= 0 && now - Date.parse(value.generated_at) < 26 * 3600000, 'Fresh observed snapshot required');
  }
  assert.equal(fs.statSync(path.join(config.member.snapshotDir, 'member-v2.json')).mode & 0o777, 0o440);
  if (config.newsletter.enabled) {
    assert.equal(config.newsletter.trackingDisabled, true);
    assert.equal(config.smtp.euApproved, true);
    assert.ok(config.smtp.host && config.mailFrom);
  }
  return { schema: 2, provenance: 'observed', generated_at: v.generated_at, forecasts: v.forecasts.length, briefs: v.briefs.length, calendar: v.calendar.length, newsletter_enabled: config.newsletter.enabled };
}
if (require.main === module) console.log(JSON.stringify(check(loadConfig(), process.argv[2])));
module.exports = { check };
