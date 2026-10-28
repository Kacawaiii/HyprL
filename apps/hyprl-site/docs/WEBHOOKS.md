# Future alerts protocol v1

The current site implements no alerts API, subscriptions or webhook delivery. The
OpenAPI 3.1 and JSON Schema 2020-12 contracts are versioned under
`public/contracts/v1/`. Preserve this contract; breaking changes require v2.

JSON dates are ISO 8601 UTC with `Z`, prices are decimal strings with a supported
ISO 4217 currency code, and content languages use ISO 639-1 (`fr`, `en`). Header
`X-HyprL-Timestamp` is Unix seconds, the deliberate exception for signing.

Sender and receiver share a unique, random secret of at least 32 bytes per
endpoint, outside Git. Sign `ASCII(timestamp) + '.' + rawBody` using HMAC-SHA256.
`X-HyprL-Signature` is `v1=` followed by 64 lowercase hex characters. Verify the
exact received bytes before parsing JSON. Use constant-time comparison. Reject
clock skew greater than 300 seconds in either direction, missing/duplicate headers,
malformed signatures, oversized bodies, non-HTTPS transport and invalid schemas.

A timestamp alone does not stop replay within the window. After signature and
schema validation, atomically reserve the signed `event_id` in a shared TTL store
before processing. Retain it through `timestamp + 301 seconds` (including future
clock skew); reject a second reservation. Use a durable business idempotency key
for the same `event_id` across later signed retries too. `server/lib/webhook.js`
provides a tested single-process reference, not production distributed storage.
Receivers must also validate the full schema before business processing.

Each retry carries the same event ID and body, a fresh timestamp and signature.
A completed, idempotent retry can return 204; an active duplicate reservation can
return 409. Sender retry/backoff, secret rotation, revocation, subscriber
entitlements and durable queues remain prerequisites for the future service.
The current server deliberately returns RFC 9457 404 for `/api/v1/alerts`.

The reference requires a `validateEvent` callback implementing the full v1 schema
and rejects bodies larger than 64 KiB. The HTTP server uses RFC 9457 `application/problem+json` for API errors; legacy
`error` and rate-limit fields remain extension members for the admin UI.
