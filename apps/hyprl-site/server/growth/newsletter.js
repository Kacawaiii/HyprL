'use strict';
const crypto = require('node:crypto');
const VERSION = '2026-10-newsletter-v1';
const TEXT = 'Je souhaite recevoir la lettre hebdomadaire gratuite de HyprL par e-mail : brief, calendrier et « Chance ou pas ? ». Mon inscription sera active après confirmation par e-mail. Je peux retirer cet accord à tout moment via le lien de désabonnement de chaque lettre.';
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const DAY = 86400000;
function assertNewsletterPage(html) {
  const shown = html.match(/<span id="newsletter-text">([\s\S]*?)<\/span>/)?.[1];
  if (shown?.replace(/\s+/g, ' ').trim() !== TEXT || !html.includes('name="newsletter_version" value="' + VERSION + '"')) throw Error('Newsletter consent text/version mismatch');
  const input = html.match(/<input[^>]*id="newsletter"[^>]*>/)?.[0] || '';
  if (!input || /\s(?:checked|required)(?:\s|=|>)/i.test(input)) throw Error('Newsletter must be optional and unticked');
}
function publicNewsletter(n) {
  if (!n) return { status: 'not_requested' };
  const { confirmation_hash, unsubscribe_hash, deliveries, ...safe } = n;
  return safe;
}
function createNewsletter({ config, store, mailer, content, now = Date.now }) {
  let sending = false;
  const unsubscribeToken = (sub) => crypto.createHmac('sha256', config.newsletter.tokenKey).update(sub.id + ':' + sub.newsletter.requested_at).digest('base64url');
  const unsubscribeUrl = sub => config.publicUrl + '/lettre/desabonnement?token=' + unsubscribeToken(sub);
  function prepare(id, token, at) {
    const sub = { id, newsletter: { status: 'pending', version: VERSION, text: TEXT, requested_at: at,
      confirmation_hash: sha(token), confirmation_expires_at: new Date(now() + 2 * DAY).toISOString(),
      confirmed_at: null, unsubscribed_at: null, deliveries: {} } };
    sub.newsletter.unsubscribe_hash = sha(unsubscribeToken(sub));
    return sub.newsletter;
  }
  function find(token, field) {
    if (!/^[A-Za-z0-9_-]{43}$/.test(token || '')) return null;
    const hash = sha(token);
    return store.all().find(s => s.newsletter?.[field] === hash) || null;
  }
  function pending(token) {
    const s = find(token, 'confirmation_hash');
    return s?.newsletter.status === 'pending' && Date.parse(s.newsletter.confirmation_expires_at) > now() ? s : null;
  }
  async function unsubscribe(token) {
    const sub = find(token, 'unsubscribe_hash');
    if (!sub) return false;
    await store.update(sub.id, s => {
      s.newsletter = { status: 'unsubscribed', version: s.newsletter.version, text: s.newsletter.text, requested_at: s.newsletter.requested_at, confirmed_at: s.newsletter.confirmed_at, unsubscribed_at: new Date(now()).toISOString(), unsubscribe_hash: s.newsletter.unsubscribe_hash };
      return s;
    });
    return true;
  }
  async function sendWeekly() {
    if (sending) throw Object.assign(Error('newsletter_busy'), { status: 409 });
    if (!config.newsletter.enabled || !config.newsletter.trackingDisabled || !config.smtp.host || !config.smtp.euApproved) throw Object.assign(Error('newsletter_mail_disabled'), { status: 503 });
    sending = true;
    const counts = { sent: 0, uncertain: 0, skipped: 0 };
    try {
      const edition = content.edition(); // Refuses stale, missing or empty real content.
      for (const candidate of store.all()) {
        const sub = await store.update(candidate.id, s => {
          const n = s.newsletter;
          if (s.status !== 'confirmed' || n?.status !== 'confirmed' || !n.confirmed_at || n.version !== VERSION || n.deliveries?.[edition.id]) return null;
          n.deliveries ||= {};
          // Persist BEFORE SMTP: a crash or ambiguous SMTP error never triggers an automatic duplicate.
          n.deliveries[edition.id] = { status: 'reserved', at: new Date(now()).toISOString(), sha256: sha(edition.text) };
          return s;
        });
        if (!sub) { counts.skipped++; continue; }
        // Withdrawal/deletion during the await above cancels this recipient.
        if (store.findById(sub.id)?.newsletter?.status !== 'confirmed') { counts.skipped++; continue; }
        let status = 'uncertain';
        try {
          const url = unsubscribeUrl(sub);
          const sent = await mailer.send({ from: config.mailFrom, to: sub.email, subject: edition.subject, text: edition.text + '\n\nVous recevez cette lettre après votre confirmation.\nSe désabonner de la lettre : ' + url + '\nInformation générale, pas de conseil en investissement.\nHyprL', list: { unsubscribe: { url, comment: 'Se désabonner de la lettre' } }, headers: { 'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click', 'X-Mailin-Track-Clicks': '0', 'X-Mailin-Track-Opens': '0' } });
          if (sent === true) status = 'sent';
        } catch { /* Ambiguous delivery stays reserved against duplicate sends. */ }
        await store.update(sub.id, s => {
          if (s.newsletter?.deliveries?.[edition.id]) s.newsletter.deliveries[edition.id].status = status;
          return s;
        });
        counts[status]++;
      }
      return { edition: edition.id, ...counts };
    } finally { sending = false; }
  }
  return { prepare, pending, findUnsubscribe: token => find(token, 'unsubscribe_hash'), unsubscribe, sendWeekly, unsubscribeUrl };
}
module.exports = { VERSION, TEXT, assertNewsletterPage, publicNewsletter, createNewsletter };
