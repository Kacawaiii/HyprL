'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const { mergePrivacy } = require('../server/growth/merge-privacy');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const { check } = require('../server/growth/release-preflight');
const { fixture } = require('./member-fixtures');
test('release privacy merge preserves all operator facts and replaces only growth sections', () => {
  const live = '<h1>Politique</h1><p>synthetic operator facts</p><h2>Espace membre et connexion Discord</h2><p>synthetic member facts</p><h2>Destinataires</h2><p>synthetic processor facts</p>';
  const proposed = '<h1>New</h1><p>Unreviewed placeholder</p><h2>Lettre hebdomadaire gratuite</h2><p>separate consent and DOI</p><h2>Parrainage Discord</h2><p>private ledger</p><h2>Espace membre et connexion Discord</h2>';
  const merged = mergePrivacy(live, proposed);
  for (const fact of ['synthetic operator facts', 'synthetic member facts', 'synthetic processor facts', 'separate consent and DOI', 'private ledger']) assert.ok(merged.includes(fact));
  assert.ok(!merged.includes('Unreviewed placeholder'));
  assert.equal(mergePrivacy(merged, proposed), merged);
  assert.throws(() => mergePrivacy('<h1>Different layout</h1>', proposed), /structure/);
  assert.throws(() => mergePrivacy(live, '<h1>No growth sections</h1>'), /Missing/);
});

test('release preflight refuses stale/synthetic exports and state within either release, including symlink parents', t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hyprl-release-check-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const root = path.join(directory, 'stage'), live = path.join(directory, 'live'), snapshots = path.join(directory, 'snapshots');
  for (const p of [root, live, snapshots]) fs.mkdirSync(p);
  const now = Date.parse('2026-10-31T12:00:00Z'), value = fixture(now);
  // Entire fixture remains synthetic; this label exercises the provenance gate.
  value.provenance = 'observed';
  const file = path.join(snapshots, 'member-v2.json');
  const write = () => { if (fs.existsSync(file)) fs.chmodSync(file, 0o600); fs.writeFileSync(file, JSON.stringify(value)); fs.chmodSync(file, 0o440); };
  write();
  const config = { root, production: true, host: '127.0.0.1', port: 3000, publicUrl: 'https://hyprlcore.com', dataFile: path.join(directory, 'data/subscribers.json'), publicSnapshotDir: snapshots, member: { enabled: true, stateFile: path.join(directory, 'data/member.json'), snapshotDir: snapshots }, newsletter: { enabled: false } };
  assert.equal(check(config, live, now).schema, 2);
  assert.throws(() => check(config, live, now + 27 * 3600000), /Fresh/);
  value.provenance = 'synthetic'; write();
  assert.throws(() => check(config, live, now), /snapshot_unavailable/);
  value.provenance = 'observed'; write();
  assert.throws(() => check({ ...config, dataFile: path.join(live, 'private.json') }, live, now), /outside/);
  fs.symlinkSync(root, path.join(directory, 'alias'));
  assert.throws(() => check({ ...config, dataFile: path.join(directory, 'alias/private.json') }, live, now), /outside/);
  fs.chmodSync(file, 0o640);
  assert.throws(() => check(config, live, now));
});
