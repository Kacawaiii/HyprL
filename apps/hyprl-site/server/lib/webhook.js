'use strict';
// Future v1 protocol reference. It is not connected to a live delivery endpoint.
const crypto = require('node:crypto');
const REPLAY_WINDOW_SECONDS = 300;
const sha256 = (s) => crypto.createHash('sha256').update(s).digest('hex');

function signature(secret, timestamp, rawBody) {
  if (typeof secret !== 'string' || Buffer.byteLength(secret) < 32) throw new Error('Webhook secret must have at least 32 bytes');
  if (!/^\d{10}$/.test(timestamp) || !Buffer.isBuffer(rawBody)) throw new Error('Invalid signing input');
  return 'v1=' + crypto.createHmac('sha256', secret).update(timestamp + '.', 'utf8').update(rawBody).digest('hex');
}

function verify({ secret, timestamp, supplied, rawBody, now = Date.now() }) {
  if (!/^\d{10}$/.test(timestamp) || !/^v1=[a-f0-9]{64}$/.test(supplied) || !Number.isFinite(now) ||
      !Buffer.isBuffer(rawBody) || rawBody.length > 64 * 1024 ||
      Math.abs(Math.floor(now / 1000) - Number(timestamp)) > REPLAY_WINDOW_SECONDS) return false;
  try {
    return crypto.timingSafeEqual(Buffer.from(supplied), Buffer.from(signature(secret, timestamp, rawBody)));
  } catch { return false; }
}

// Receiver reference: reserve the signed event ID atomically BEFORE processing.
// Repeated delivery IDs are rejected within the window; entries expire to bound memory.
// Single-process helper only: use a shared, atomic TTL store across receiver instances.
function createReceiver({ secret, now = Date.now, validateEvent }) {
  if (typeof validateEvent !== 'function') throw new Error('A full v1 schema validator is required');
  const seen = new Map();
  return function receive({ timestamp, supplied, rawBody }) {
    const clock = now();
    if (!verify({ secret, timestamp, supplied, rawBody, now: clock })) return null;
    let event;
    try { event = JSON.parse(rawBody.toString('utf8')); } catch { return null; }
    if (!event || event.schema_version !== '1.0.0' || !validateEvent(event)) return null;
    for (const [id, until] of seen) if (until <= clock) seen.delete(id);
    const key = sha256(event.event_id);
    if (seen.has(key)) return null;
    // Include future clock skew: signature remains usable until timestamp + window.
    seen.set(key, (Number(timestamp) + REPLAY_WINDOW_SECONDS + 1) * 1000);
    return event;
  };
}
module.exports = { signature, verify, createReceiver, REPLAY_WINDOW_SECONDS };
