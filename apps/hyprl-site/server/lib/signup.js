'use strict';
const { sha256 } = require('./auth');
const { confirmationMessage, digestMessage } = require('./mail');
const { recordExpiresAt } = require('./retention');
const tokenHash = token => sha256(token).toString('hex');

function validateActions(raw) {
  if (!raw.trim()) return '';
  if (raw.length > 300 || !/^[\p{L}\p{N}. ,\-\r\n]+$/u.test(raw)) return null;
  const entries = raw.split(',').map(s => s.replace(/\s+/g, ' ').trim());
  if (entries.length > 10 || entries.some(s => !s || s.length > 60)) return null;
  // Company labels/tickers, not prose. Connectors in names remain allowed.
  const requests = /\b(?:i|you|your|wanted|want|know|please|je|vous|votre|voudrais|souhaite|connaître|savoir|quiero|quería|saber|su|wollte|wissen|ihren)\b|хотел|узнать|цену|我想|价格/iu;
  for (const name of entries) {
    const words = name.split(' ');
    if (!/[\p{L}\p{N}]/u.test(name) || words.length > 5 || requests.test(name)) return null;
    if (words.length > 1 && words.some(w => !/^(?:of|the|and|de|du|des|et|&)$/.test(w) && !/^[\p{Lu}\p{Lt}\p{N}]/u.test(w))) return null;
  }
  return entries.join(', ');
}

function createSignup({ config, store, guard, mailer, now = Date.now, logger = console }) {
  let digestRunning = false;
  function pending(token) {
    if (!/^[A-Za-z0-9_-]{43}$/.test(token || '')) return null;
    return store.all().find(s => s.status === 'pending' && s.confirmation_hash === tokenHash(token) && Date.parse(s.confirmation_expires_at) > now() && Date.parse(recordExpiresAt(s)) > now()) || null;
  }
  async function confirm(token) {
    const sub = pending(token);
    if (!sub) return false;
    return Boolean(await store.update(sub.id, s => {
      if (s.status !== 'pending' || s.confirmation_hash !== tokenHash(token) || Date.parse(s.confirmation_expires_at) <= now() || Date.parse(recordExpiresAt(s)) <= now()) return null;
      s.status = 'confirmed'; s.confirmed_at = new Date(now()).toISOString();
      delete s.confirmation_hash;
      // One explicit confirmation covers only the separately requested purposes.
      if (s.newsletter?.status === 'pending' && Date.parse(s.newsletter.confirmation_expires_at) > now()) {
        s.newsletter.status = 'confirmed'; s.newsletter.confirmed_at = s.confirmed_at;
        delete s.newsletter.confirmation_hash;
      }
      return s;
    }));
  }
  async function sendConfirmation(sub, token, unsubscribeToken) {
    // The durable address reservation and record always precede SMTP. Never retry
    // disabled, failed or ambiguous sends, including after process restart.
    let delivery = 'uncertain';
    try {
      const sent = await mailer.send(confirmationMessage(sub, { from: config.mailFrom,
        confirmationUrl: config.publicUrl + '/confirmation?token=' + token,
        unsubscribeUrl: config.publicUrl + '/desinscription?token=' + unsubscribeToken }));
      delivery = sent === true ? 'sent' : 'uncertain';
      logger.info('[signup] ' + JSON.stringify({ confirmation_sent: sent === true ? 1 : 0, confirmation_uncertain: sent === true ? 0 : 1 }));
    } catch { logger.info('[signup] ' + JSON.stringify({ confirmation_uncertain: 1 })); }
    await store.update(sub.id, s => { s.confirmation_delivery = delivery; s.confirmation_attempted_at = new Date(now()).toISOString(); return s; });
  }
  async function digest() {
    if (digestRunning || !config.notifyEmail || !config.smtp.euApproved || !config.smtp.host) return { sent: 0 };
    digestRunning = true;
    try {
      await store.purgeExpired(now());
      const midnight = Date.parse(new Date(now()).toISOString().slice(0, 10));
      const candidates = store.all().filter(s => s.status === 'confirmed' && s.confirmed_at && Date.parse(s.confirmed_at) < midnight && !s.digest_reserved_at);
      if (!candidates.length || !await guard.reserveDigest()) return { sent: 0 };
      const ids = [];
      for (const candidate of candidates) {
        const sub = await store.update(candidate.id, s => {
          if (s.status !== 'confirmed' || s.digest_reserved_at) return null;
          s.digest_reserved_at = new Date(now()).toISOString(); return s;
        });
        if (sub) ids.push(sub.id);
      }
      const confirmed = ids.map(id => store.findById(id)).filter(s => s?.status === 'confirmed');
      if (!confirmed.length) return { sent: 0 };
      let sent = false;
      try { sent = await mailer.send(digestMessage(confirmed, { to: config.notifyEmail, from: config.mailFrom })); } catch { /* reservation is final */ }
      logger.info('[signup] ' + JSON.stringify({ digest_sent: sent === true ? 1 : 0, confirmed_count: confirmed.length }));
      return { sent: sent === true ? 1 : 0 };
    } finally { digestRunning = false; }
  }
  const timer = setInterval(() => digest().catch(() => logger.info('[signup] ' + JSON.stringify({ digest_error: 1 }))), 3600000);
  timer.unref();
  return { pending, confirm, sendConfirmation, digest, close: () => clearInterval(timer) };
}
module.exports = { createSignup, validateActions };
