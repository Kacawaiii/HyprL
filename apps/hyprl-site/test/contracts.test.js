'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const Ajv = require('ajv/dist/2020');
const addFormats = require('ajv-formats');
const parser = require('@apidevtools/swagger-parser');
const { signature, verify, createReceiver } = require('../server/lib/webhook');
const root = path.join(__dirname, '../public/contracts/v1');
const read = (name) => JSON.parse(fs.readFileSync(path.join(root, name), 'utf8'));
const ajv = new Ajv({ strict: true });
addFormats(ajv);
for (const name of ['alert', 'webhook', 'problem']) ajv.addSchema(read(name + '.schema.json'));
const validate = ajv.getSchema('https://hyprlcore.com/contracts/v1/webhook.schema.json');
const example = () => ({
  schema_version: '1.0.0', event_id: '00000000-0000-4000-8000-000000000001', type: 'market.alert', created_at: '2026-10-26T12:00:00.000Z',
  alert: { id: '00000000-0000-4000-8000-000000000002', created_at: '2026-10-26T12:00:00Z', symbol: 'SYNTH', asset_class: 'equity', title: 'Synthetic notice', body: 'Synthetic offline information.', source_url: 'https://example.test/notice', language: 'fr', price: { amount: '19.00', currency: 'EUR' } }
});

test('OpenAPI 3.1 resolves entirely offline and explicitly describes a future API', async () => {
  // JSON Schema $id rebases relative refs to the public canonical URL. Resolve
  // those exact IDs from the same local files, with HTTP still forbidden.
  const localSchemas = new Map(['alert', 'webhook', 'problem'].map(name => [read(name + '.schema.json').$id, name + '.schema.json']));
  const spec = await parser.validate(path.join(root, 'openapi.json'), { resolve: {
    http: false,
    canonicalLocal: { order: 1, canRead: file => localSchemas.has(file.url), read: file => fs.readFileSync(path.join(root, localSchemas.get(file.url)), 'utf8') },
  } });
  assert.equal(spec.openapi, '3.1.0'); assert.equal(spec.info.version, '1.0.0');
  assert.match(spec.info.description, /not enabled/);
  assert.ok(spec.webhooks.marketAlert.post.requestBody);
});

test('v1 webhook schemas require UTC, ISO currency, ISO language and reject extra fields', () => {
  assert.equal(validate(example()), true, JSON.stringify(validate.errors));
  const changes = [
    (e) => { e.created_at = '2026-10-26T12:00:00+01:00'; },
    (e) => { e.alert.created_at = '2026-02-30T00:00:00Z'; },
    (e) => { e.alert.price.currency = '€'; },
    (e) => { e.alert.price.currency = 'BTC'; },
    (e) => { e.alert.price.amount = 19; },
    (e) => { e.alert.language = 'french'; },
    (e) => { e.extra = 'unexpected'; },
    (e) => { delete e.event_id; },
    (e) => { e.alert.source_url = 'http://example.test/notice'; },
    (e) => { e.schema_version = '2.0.0'; }
  ];
  for (const change of changes) { const e = example(); change(e); assert.equal(validate(e), false, JSON.stringify(e)); }
});

test('HMAC uses exact raw bytes and constant-sized signature; stale/future/tampered rejected', () => {
  const secret = 'synthetic-test-secret-with-at-least-32-bytes';
  const timestamp = '1793016000';
  const rawBody = Buffer.from(JSON.stringify(example()));
  const supplied = signature(secret, timestamp, rawBody);
  const expected = 'v1=' + crypto.createHmac('sha256', secret).update(Buffer.concat([Buffer.from(timestamp + '.'), rawBody])).digest('hex');
  assert.equal(supplied, expected);
  const args = { secret, timestamp, rawBody, supplied, now: Number(timestamp) * 1000 };
  assert.equal(verify(args), true);
  assert.equal(verify({ ...args, rawBody: Buffer.from(JSON.stringify(example(), null, 2)) }), false);
  assert.equal(verify({ ...args, supplied: 'v1=' + '0'.repeat(64) }), false);
  assert.equal(verify({ ...args, supplied: 'invalid' }), false);
  assert.equal(verify({ ...args, timestamp: '-1' }), false);
  assert.equal(verify({ ...args, now: args.now + 301000 }), false);
  assert.equal(verify({ ...args, now: args.now - 301000 }), false);
  assert.equal(verify({ ...args, now: args.now + 300000 }), true);
  assert.throws(() => signature('short', timestamp, rawBody));
});

test('receiver rejects repeated signed event IDs through the replay window, including future clock skew', () => {
  const secret = 'synthetic-test-secret-with-at-least-32-bytes';
  let clock = 1793016000000;
  const receive = createReceiver({ secret, now: () => clock, validateEvent: validate });
  const timestamp = String(clock / 1000 + 300);
  const rawBody = Buffer.from(JSON.stringify(example()));
  const message = { timestamp, rawBody, supplied: signature(secret, timestamp, rawBody) };
  assert.equal(receive(message).event_id, example().event_id);
  assert.equal(receive(message), null);
  clock += 300000;
  assert.equal(receive(message), null);
  clock += 300000;
  assert.equal(receive(message), null, 'still a replay at the signature boundary');
  clock += 1000;
  assert.equal(receive(message), null, 'signature expired');
});

test('receiver validates signed payloads before reserving IDs and rejects oversized bodies', () => {
  const secret = 'synthetic-test-secret-with-at-least-32-bytes';
  const timestamp = '1793016000';
  const receive = createReceiver({ secret, now: () => Number(timestamp) * 1000, validateEvent: validate });
  const deliver = (value) => {
    const rawBody = Buffer.from(JSON.stringify(value));
    return receive({ timestamp, rawBody, supplied: signature(secret, timestamp, rawBody) });
  };
  assert.equal(deliver(null), null);
  const invalid = example(); delete invalid.alert;
  assert.equal(deliver(invalid), null);
  assert.equal(deliver(example()).event_id, example().event_id);
  const rawBody = Buffer.alloc(64 * 1024 + 1);
  assert.equal(verify({ secret, timestamp, rawBody, supplied: signature(secret, timestamp, rawBody), now: Number(timestamp) * 1000 }), false);
});
