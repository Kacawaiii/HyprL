'use strict';
const fs = require('node:fs');
const crypto = require('node:crypto');
const { SubscriberStore } = require('./store');
const { validateActions } = require('./signup');
const { CONSENT_TEXTS } = require('./consent');
const { sha256 } = require('./auth');
const DAY = 86400000;
const EMAIL_RE = /^[^\s@<>()",;:\\\[\]]+@[^\s@<>()",;:\\\[\]]+\.[^\s@<>()",;:\\\[\]]{2,}$/;
const REASONS = Object.freeze({
  origin: 'Origine du formulaire absente ou incorrecte.',
  honeypot: 'Champ réservé aux robots rempli.',
  no_token: 'Aucun jeton de formulaire.',
  invalid_token: 'Jeton de formulaire invalide.',
  too_fast: 'Formulaire envoyé trop vite (moins de 3 secondes).',
  expired_token: 'Jeton de formulaire expiré.',
  wrong_ip: 'Jeton utilisé depuis une autre adresse réseau.',
  work: 'Preuve de travail absente ou incorrecte.',
  replay: 'Jeton déjà utilisé.',
  cap: 'Plafond de demandes atteint.',
  mail_cap: 'Plafond quotidien des confirmations atteint.',
  duplicate: 'Une confirmation a déjà été réservée pour cette adresse.',
  consent: 'Consentement absent ou formulaire périmé.',
  email: 'Adresse e-mail invalide.',
  actions: 'Phrase ou contenu non conforme dans le champ actions.',
  legacy: 'Ancienne inscription sans preuve de confirmation par e-mail.',
});

class QuarantineStore extends SubscriberStore {
  load() {
    if (fs.existsSync(this.file)) {
      const st = fs.lstatSync(this.file);
      if (!st.isFile() || st.isSymbolicLink() || st.nlink !== 1 || (st.mode & 0o077)) throw Error('Private regular quarantine file required');
    }
    super.load();
    if (this.items.some(s => s.status !== 'suspect' || !REASONS[s.reason_code] || !Number.isFinite(Date.parse(s.expires_at)))) throw Error('Invalid quarantine record');
    return this;
  }
  purgeExpiredSync(now) {
    const kept = this.items.filter(s => Date.parse(s.expires_at) > now);
    const count = this.items.length - kept.length;
    if (count) { this._writeSync(kept); this.items = kept; }
    return count;
  }
  purgeExpired(now) {
    return this._mutate(items => {
      const kept = items.filter(s => Date.parse(s.expires_at) > now);
      return { items: kept.length === items.length ? items : kept, result: items.length - kept.length };
    });
  }
}

function createTriage({ config, store, guard, signup, now = Date.now, logger = console }) {
  const quarantine = new QuarantineStore(config.quarantineFile).load();
  let queue = Promise.resolve(), timer, closed = false;
  const serial = change => {
    const operation = queue.then(change);
    queue = operation.catch(() => {});
    return operation;
  };
  const safeRecord = (record, reason, imported = false) => {
    const at = new Date(now()).toISOString();
    const email = String(record.email || '').trim().toLowerCase();
    // No captured body, IP, token, newsletter membership or arbitrary object is retained.
    return { id: record.id || crypto.randomUUID(), email: email.slice(0, 254), email_valid: email.length <= 254 && EMAIL_RE.test(email),
      channels: [...new Set(Array.isArray(record.channels) ? record.channels : [])].filter(c => ['discord', 'slack', 'telegram', 'teams', 'email', 'webhook', 'api'].includes(c)),
      actions: String(record.actions || '').slice(0, 1000), created_at: record.created_at || at,
      quarantined_at: at, expires_at: new Date(now() + 7 * DAY).toISOString(), status: 'suspect',
      reason_code: reason, reason: REASONS[reason], imported,
      consent: record.consent?.given === true && typeof CONSENT_TEXTS[record.consent.version] === 'string' && CONSENT_TEXTS[record.consent.version] === record.consent.text && Number.isFinite(Date.parse(record.consent.checked_at))
        ? { given: true, version: record.consent.version, text: record.consent.text, checked_at: record.consent.checked_at } : null };
  };
  // Import before subscriber retention. Copy first, then remove: interruption can
  // leave the same ID in both files, and this startup procedure safely finishes it.
  const kept = [];
  for (const record of store.all()) {
    const reason = !record.triage_reviewed_at && validateActions(record.actions || '') === null ? 'actions'
      : !['pending', 'confirmed'].includes(record.status) ? 'legacy' : null;
    if (!reason) { kept.push(record); continue; }
    if (!quarantine.findById(record.id)) {
      const next = [...quarantine.items, safeRecord(record, reason, true)];
      quarantine._writeSync(next); quarantine.items = next;
    }
  }
  if (kept.length !== store.items.length) { store._writeSync(kept); store.items = kept; }
  // Finish an interrupted promotion without ever resending an uncertain email.
  const remaining = quarantine.items.filter(q => !store.all().some(s => s.triage_source_id === q.id));
  if (remaining.length !== quarantine.items.length) { quarantine._writeSync(remaining); quarantine.items = remaining; }
  quarantine.purgeExpiredSync(now());

  function schedulePurge() {
    clearTimeout(timer);
    if (closed || !quarantine.items.length) return;
    const next = quarantine.items.reduce((earliest, s) => Math.min(earliest, Date.parse(s.expires_at)), Infinity);
    timer = setTimeout(() => run().catch(() => {
      logger.error('[triage] {"purge_error":1}');
      if (!closed) { timer = setTimeout(schedulePurge, 1000); timer.unref(); }
    }), Math.min(2147483647, Math.max(1, next - now())));
    timer.unref();
  }
  async function run() { await quarantine.purgeExpired(now()); schedulePurge(); }
  schedulePurge();
  async function capture(record, reason) {
    await run();
    const added = await quarantine.add(safeRecord(record, reason));
    schedulePurge();
    logger.info('[triage] ' + JSON.stringify({ quarantined: added ? 1 : 0 }));
  }
  function promote(id) {
    return serial(async () => {
      await run();
      const q = quarantine.findById(id);
      if (!q) return { status: 404, error: 'Demande introuvable ou expirée.' };
      if (!q.email_valid || !EMAIL_RE.test(q.email) || !q.consent) return { status: 409, error: 'Adresse ou consentement manquant : invitez la personne à remplir le formulaire.' };
      if (store.findByEmail(q.email)) return { status: 409, error: 'Cette adresse possède déjà une inscription.' };
      if (!config.smtp.euApproved || !config.smtp.host || !config.mailFrom) return { status: 503, error: 'L’envoi des confirmations est suspendu.' };
      if (!await guard.reserveAddress(q.email)) return { status: 409, error: 'Plafond atteint ou confirmation déjà réservée : aucun nouvel e-mail envoyé.' };
      if (!quarantine.findById(q.id) || Date.parse(q.expires_at) <= now()) return { status: 404, error: 'Demande introuvable ou expirée.' };
      const at = new Date(now()).toISOString();
      const token = crypto.randomBytes(32).toString('base64url'), unsubscribe = crypto.randomBytes(32).toString('base64url');
      const record = { id: crypto.randomUUID(), email: q.email, channels: q.channels, actions: q.actions,
        created_at: at, status: 'pending', confirmed_at: null, consent: q.consent,
        confirmation_hash: sha256(token).toString('hex'), confirmation_expires_at: new Date(now() + 2 * DAY).toISOString(),
        unsubscribe_token_hash: sha256(unsubscribe).toString('hex'), triage_source_id: q.id,
        triage_reviewed_at: at, triage_reason: q.reason, triage_original_created_at: q.created_at, confirmation_reserved_at: at };
      // Review authorizes early-access confirmation only, never marketing consent.
      const added = await store.add(record);
      if (!added) return { status: 409, error: 'Cette adresse possède déjà une inscription.' };
      await quarantine.remove(q.id);
      schedulePurge();
      logger.info('[triage] {"promotion_confirmation_reserved":1}');
      if (store.findById(record.id)) await signup.sendConfirmation(record, token, unsubscribe);
      return { status: 200 };
    });
  }
  const remove = id => serial(() => quarantine.remove(id));
  const eraseEmail = email => serial(async () => {
    for (const q of quarantine.all().filter(s => s.email === email)) await quarantine.remove(q.id);
    schedulePurge();
  });
  return { quarantine, capture, promote, remove, eraseEmail, run, close() { closed = true; clearTimeout(timer); } };
}
module.exports = { createTriage, QuarantineStore, REASONS };
