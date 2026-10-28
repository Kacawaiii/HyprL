'use strict';
// Application HTTP HyprL : fichiers statiques, inscription, désinscription, administration.
// Aucune dépendance hors Node.js, sauf nodemailer pour l'envoi d'e-mails.

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const { CONSENT_TEXTS, CURRENT_VERSION, assertPageMatches } = require('./lib/consent');
const { SubscriberStore } = require('./lib/store');
const { expiresAt, startRetentionJob } = require('./lib/retention');
const { passwordMatches, AttemptLimiter, SessionStore, sha256 } = require('./lib/auth');
const { createMailer, notificationMessage, confirmationMessage } = require('./lib/mail');
const { createPages, maskEmail, esc } = require('./lib/pages');
const { VERSION: NEWSLETTER_VERSION, assertNewsletterPage, publicNewsletter, createNewsletter } = require('./growth/newsletter');
const { PublicContent } = require('./growth/content');

const ALLOWED_CHANNELS = ['discord', 'slack', 'telegram', 'teams', 'email', 'webhook', 'api'];
const EMAIL_RE = /^[^\s@<>()",;:\\[\]]+@[^\s@<>()",;:\\[\]]+\.[^\s@<>()",;:\\[\]]{2,}$/;
const MAX_BODY = 16 * 1024;
const COOKIE = 'hyprl_admin';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.ico': 'image/x-icon',
  '.json': 'application/json; charset=utf-8', '.woff2': 'font/woff2', '.txt': 'text/plain; charset=utf-8', '.xml': 'application/xml', '.webmanifest': 'application/manifest+json',
};

const httpTitle = (status) => require('node:http').STATUS_CODES[status] || 'Error';

const CSP = [
  "default-src 'self'", "script-src 'self'", "style-src 'self'", "font-src 'self'", "img-src 'self' data:",
  "connect-src 'self'", "form-action 'self'", "frame-ancestors 'none'", "base-uri 'none'", "object-src 'none'",
].join('; ');

function createApp(config, deps = {}) {
  const now = deps.now || Date.now;
  const logger = deps.logger || console;

  const indexHtml = fs.readFileSync(path.join(config.publicDir, 'index.html'), 'utf8');
  assertPageMatches(indexHtml);
  assertNewsletterPage(indexHtml);

  const store = deps.store || new SubscriberStore(config.dataFile).load();
  const retention = startRetentionJob(store, { now, logger, ...(deps.retention || {}) });
  const mailer = deps.mailer || createMailer(config, logger);
  const pages = createPages(config.publicDir);
  const loginLimiter = new AttemptLimiter({ max: config.login.maxAttempts, windowMs: config.login.windowMs, now });
  const subscribeLimiter = new AttemptLimiter({ max: config.subscribeLimit.max, windowMs: config.subscribeLimit.windowMs, now });
  const sessions = new SessionStore({ idleMs: config.session.idleMs, maxMs: config.session.maxMs, now });
  const member = require('./member/app').createMemberApp(config, { now, ...(deps.member || {}) });
  const publicOrigin = new URL(config.publicUrl).origin;
  const content = deps.content || new PublicContent(config, { now, ...(deps.publicContent || {}) });
  const newsletter = createNewsletter({ config, store, mailer, content, now });

  const sweeper = setInterval(() => { loginLimiter.sweep(); subscribeLimiter.sweep(); sessions.sweep(); }, 60 * 1000);
  sweeper.unref();

  // ---------- utilitaires HTTP ----------
  function baseHeaders(res) {
    res.setHeader('Content-Security-Policy', CSP);
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    // 'same-origin' : aucun référent envoyé à l'extérieur, mais les navigateurs envoient un en-tête Origin exploitable
    // ('no-referrer' leur fait envoyer « Origin: null » sur les formulaires, ce qui casserait le contrôle d'origine).
    res.setHeader('Referrer-Policy', 'same-origin');
    res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=(), interest-cohort=(), browsing-topics=()');
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    if (config.secureCookies) res.setHeader('Strict-Transport-Security', 'max-age=31536000');
  }

  function send(res, status, body, type = 'text/html; charset=utf-8', extra = {}) {
    res.writeHead(status, { 'Content-Type': type, 'Cache-Control': 'no-store', ...extra });
    res.end(body);
  }
  const json = (res, status, obj, extra) => {
    if (status >= 400) {
      const { error, message, ...extensions } = obj || {};
      return send(res, status, JSON.stringify({
        type: 'about:blank', title: httpTitle(status), status,
        detail: message || httpTitle(status), ...extensions,
        ...(error ? { error } : {}),
      }), 'application/problem+json; charset=utf-8', extra);
    }
    return send(res, status, obj === undefined ? '' : JSON.stringify(obj), 'application/json; charset=utf-8', extra);
  };
  const noContent = (res, extra = {}) => { res.writeHead(204, { 'Cache-Control': 'no-store', ...extra }); res.end(); };
  const redirect = (res, location) => { res.writeHead(303, { Location: location, 'Cache-Control': 'no-store' }); res.end(); };

  function clientIp(req) {
    if (config.trustProxy) {
      const real = req.headers['x-real-ip'];
      if (typeof real === 'string' && real) return real.trim();
      const xff = req.headers['x-forwarded-for'];
      if (typeof xff === 'string' && xff) return xff.split(',').pop().trim();
    }
    return req.socket.remoteAddress || 'inconnu';
  }

  function readBody(req) {
    return new Promise((resolve, reject) => {
      let size = 0;
      const chunks = [];
      req.on('data', (c) => {
        size += c.length;
        if (size > MAX_BODY) { reject(Object.assign(new Error('too_large'), { status: 413 })); return; }
        chunks.push(c);
      });
      req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
      req.on('error', reject);
    });
  }

  async function readForm(req) {
    const type = String(req.headers['content-type'] || '');
    if (!type.startsWith('application/json') && !type.startsWith('application/x-www-form-urlencoded')) {
      throw Object.assign(new Error('unsupported_media_type'), { status: 415 });
    }
    const raw = await readBody(req);
    if (type.startsWith('application/json')) {
      let obj;
      try { obj = JSON.parse(raw || '{}'); if (!obj || typeof obj !== 'object' || Array.isArray(obj)) throw new Error(); } catch { throw Object.assign(new Error('bad_json'), { status: 400 }); }
      return { get: (k) => (obj[k] == null ? null : String(obj[k])), getAll: (k) => [].concat(obj[k] ?? []).map(String), json: true };
    }
    const p = new URLSearchParams(raw);
    return { get: (k) => p.get(k), getAll: (k) => p.getAll(k), json: false };
  }

  function cookie(req, name) {
    const header = req.headers.cookie || '';
    for (const part of header.split(';')) {
      const i = part.indexOf('=');
      if (i > 0 && part.slice(0, i).trim() === name) {
        try { return decodeURIComponent(part.slice(i + 1).trim()); } catch { return null; }
      }
    }
    return null;
  }

  function sessionCookie(value, maxAge) {
    return [
      COOKIE + '=' + value, 'Path=/', 'HttpOnly', 'SameSite=Strict',
      config.secureCookies ? 'Secure' : null, maxAge !== undefined ? 'Max-Age=' + maxAge : null,
    ].filter(Boolean).join('; ');
  }

  /** Refuse les requêtes d'écriture venant d'une autre origine (protection CSRF en plus de SameSite=Strict). */
  function sameOrigin(req) {
    const origin = req.headers.origin;
    if (!origin) return true;
    const hostOrigin = (config.secureCookies ? 'https://' : 'http://') + req.headers.host;
    return origin === publicOrigin || (!config.production && origin === hostOrigin);
  }

  function newToken() {
    const token = crypto.randomBytes(32).toString('base64url');
    return { token, hash: sha256(token).toString('hex') };
  }

  const unsubscribeUrl = (token) => config.publicUrl + '/desinscription?token=' + encodeURIComponent(token);

  function deliver(msg, label) {
    Promise.resolve()
      .then(() => mailer.send(msg))
      .catch(() => logger.error('[mail] échec de l’envoi (' + label + ')'));
  }

  // ---------- inscription ----------
  async function subscribe(req, res, apiErrors = false) {
    if (!sameOrigin(req)) return json(res, 403, { error: 'forbidden' });
    const form = await readForm(req);
    const wantsJson = form.json || String(req.headers.accept || '').includes('application/json');
    const fail = (status, title, message) => apiErrors || wantsJson
      ? json(res, status, { error: title, message })
      : send(res, status, pages.error(title, message));

    const ip = clientIp(req);
    if (subscribeLimiter.blockedFor(ip)) {
      return fail(429, 'Trop de demandes', 'Trop d’inscriptions depuis cette connexion. Réessayez dans une heure.');
    }
    subscribeLimiter.fail(ip); // compte chaque tentative d'inscription

    // Pot de miel : un robot remplit ce champ invisible. On répond comme si de rien n'était, sans rien enregistrer.
    if ((form.get('site_web') || '').trim() !== '') {
      return wantsJson ? json(res, 200, { ok: true }) : redirect(res, '/merci');
    }

    // Consentement obligatoire : sans case cochée, rien n'est enregistré.
    if (form.get('consent') !== 'oui') {
      return fail(400, 'Consentement requis', 'La case de consentement doit être cochée. Aucune donnée n’a été enregistrée.');
    }
    const version = form.get('consent_version') || '';
    if (version !== CURRENT_VERSION || !CONSENT_TEXTS[version]) {
      return fail(400, 'Formulaire périmé', 'Le formulaire a changé depuis l’ouverture de la page. Rechargez-la puis réessayez. Aucune donnée n’a été enregistrée.');
    }
    const wantsNewsletter = form.get('newsletter') === 'oui';
    if (wantsNewsletter && form.get('newsletter_version') !== NEWSLETTER_VERSION) return fail(400, 'Formulaire périmé', 'Rechargez la page pour lire le texte actuel de la lettre.');
    if (wantsNewsletter && !config.newsletter.enabled) return fail(503, 'Lettre temporairement indisponible', 'La lettre ne peut pas accepter de nouvelles demandes. Réessayez ultérieurement ou décochez cette option pour l’accès anticipé seul.');

    const email = (form.get('email') || '').trim().toLowerCase();
    if (email.length > 254 || !EMAIL_RE.test(email)) {
      return fail(400, 'Adresse invalide', 'L’adresse e-mail saisie n’est pas valide. Aucune donnée n’a été enregistrée.');
    }

    const channels = [...new Set(form.getAll('channels'))].filter((c) => ALLOWED_CHANNELS.includes(c));
    const actions = (form.get('actions') || '')
      .replace(/[\u0000-\u001F\u007F]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 500);

    // Adresse déjà inscrite : même réponse, aucune donnée modifiée, aucun e-mail renvoyé (pas de fuite d'information).
    const existing = store.findByEmail(email);
    if (existing) {
      if (wantsNewsletter) await newsletter.enroll(existing.id);
      return wantsJson ? json(res, 200, { ok: true }) : redirect(res, '/merci');
    }

    const at = new Date(now()).toISOString();
    const { token, hash } = newToken();
    const record = {
      id: crypto.randomUUID(),
      email,
      channels,
      actions,
      created_at: at,
      consent: { given: true, version, text: CONSENT_TEXTS[version], checked_at: at },
      unsubscribe_token_hash: hash,
    };
    const added = await store.add(record);
    if (!added) {
      if (wantsNewsletter) await newsletter.enroll(store.findByEmail(email).id);
      return wantsJson ? json(res, 200, { ok: true }) : redirect(res, '/merci');
    }
    if (wantsNewsletter) await newsletter.enroll(record.id);

    if (config.notifyEmail) deliver(notificationMessage(record, { to: config.notifyEmail, from: config.mailFrom }), 'notification');
    deliver(confirmationMessage(record, { from: config.mailFrom, unsubscribeUrl: unsubscribeUrl(token) }), 'confirmation');

    return wantsJson ? json(res, 201, { ok: true }) : redirect(res, '/merci');
  }

  // ---------- désinscription (lien de l'e-mail) ----------
  async function unsubscribeGet(req, res, url) {
    const token = url.searchParams.get('token') || '';
    const sub = token && store.findByTokenHash(sha256(token).toString('hex'));
    if (!sub) return send(res, 404, pages.unsubscribeInvalid(), undefined, { 'X-Robots-Tag': 'noindex' });
    // Pas d'effacement sur un simple GET : les antivirus de messagerie ouvrent les liens automatiquement.
    return send(res, 200, pages.unsubscribeConfirm(token, maskEmail(sub.email)), undefined, { 'X-Robots-Tag': 'noindex' });
  }

  async function unsubscribePost(req, res, url) {
    const form = await readForm(req);
    // Jeton dans le formulaire, ou dans l'URL pour le désabonnement en un clic (RFC 8058).
    const token = form.get('token') || url.searchParams.get('token') || '';
    const sub = token && store.findByTokenHash(sha256(token).toString('hex'));
    if (!sub) return send(res, 404, pages.unsubscribeInvalid());
    await store.remove(sub.id);
    return send(res, 200, pages.unsubscribeDone());
  }

  // ---------- administration ----------
  function requireAdmin(req, res) {
    const state = sessions.touch(cookie(req, COOKIE));
    if (state === 'ok') return true;
    json(res, 401, { error: state === 'expired' ? 'session_expired' : 'unauthorized' }, { 'Set-Cookie': sessionCookie('', 0) });
    return false;
  }

  async function login(req, res) {
    if (!sameOrigin(req)) return json(res, 403, { error: 'forbidden' });
    if (!String(req.headers['content-type'] || '').startsWith('application/json')) return json(res, 415, { error: 'json_required' });
    const ip = clientIp(req);
    const wait = loginLimiter.blockedFor(ip);
    if (wait) return json(res, 429, { error: 'too_many_attempts', retry_after: wait }, { 'Retry-After': String(wait) });

    const form = await readForm(req);
    if (!passwordMatches(form.get('password'), config.adminPassword)) {
      const remaining = loginLimiter.fail(ip);
      logger.warn('[admin] échec de connexion (' + remaining + ' tentative(s) restante(s) pour cette adresse IP)');
      return json(res, 401, { error: 'invalid_password', remaining });
    }
    loginLimiter.reset(ip);
    const token = sessions.create();
    return json(res, 200, { ok: true, idle_timeout_minutes: config.session.idleMs / 60000 }, { 'Set-Cookie': sessionCookie(token) });
  }

  function logout(req, res) {
    if (!sameOrigin(req)) return json(res, 403, { error: 'forbidden' });
    sessions.destroy(cookie(req, COOKIE));
    return noContent(res, { 'Set-Cookie': sessionCookie('', 0) });
  }

  function listSubscribers(req, res) {
    if (!requireAdmin(req, res)) return;
    const subscribers = store.all()
      .sort((a, b) => b.created_at.localeCompare(a.created_at))
      .map(({ unsubscribe_token_hash, newsletter: n, ...rest }) => ({ ...rest, newsletter: publicNewsletter(n) }));
    return json(res, 200, { subscribers });
  }

  function exportSubscriber(req, res, id) {
    if (!requireAdmin(req, res)) return;
    const record = store.findById(id);
    if (!record) return json(res, 404, { error: 'not_found' });
    const { id: personId, email, channels, actions, created_at, consent } = record;
    return json(res, 200, {
      schema_version: '1.0.0', exported_at: new Date(now()).toISOString(),
      retention_expires_at: expiresAt(created_at),
      subscriber: { id: personId, email, channels, actions, created_at, consent, newsletter: publicNewsletter(record.newsletter) },
    }, { 'Content-Disposition': 'attachment; filename="hyprl-personal-data.json"' });
  }

  async function deleteSubscriber(req, res, id) {
    if (!sameOrigin(req)) return json(res, 403, { error: 'forbidden' });
    if (!requireAdmin(req, res)) return;
    const removed = await store.remove(id);
    if (!removed) return json(res, 404, { error: 'not_found' });
    logger.info('[admin] inscrit supprimé (droit à l’effacement)');
    return noContent(res);
  }

  // ---------- fichiers statiques ----------
  const publicRoot = path.resolve(config.publicDir) + path.sep;

  function serveStatic(req, res, pathname) {
    let rel;
    try { rel = decodeURIComponent(pathname); } catch { return notFound(req, res); }
    if (rel.includes('\0')) return notFound(req, res);
    if (rel === '/') rel = '/index.html';
    else if (rel === '/admin') rel = '/admin.html';
    else if (!path.extname(rel)) rel += '.html';

    const file = path.resolve(config.publicDir, '.' + rel);
    if (!file.startsWith(publicRoot)) return notFound(req, res);
    if (path.basename(file).startsWith('.')) return notFound(req, res);

    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) return notFound(req, res);
      const ext = path.extname(file).toLowerCase();
      const etag = 'W/"' + st.size.toString(16) + '-' + Math.floor(st.mtimeMs).toString(16) + '"';
      const headers = {
        'Content-Type': MIME[ext] || 'application/octet-stream',
        ETag: etag,
        'Cache-Control': ext === '.woff2' ? 'public, max-age=31536000, immutable'
          : ext === '.html' ? 'no-cache' : 'public, max-age=3600',
      };
      if (rel === '/admin.html' || rel === '/merci.html') {
        headers['Cache-Control'] = 'no-store';
        headers['X-Robots-Tag'] = 'noindex, nofollow';
      }
      if (req.headers['if-none-match'] === etag) { res.writeHead(304, headers); return res.end(); }
      headers['Content-Length'] = st.size;
      res.writeHead(200, headers);
      if (req.method === 'HEAD') return res.end();
      fs.createReadStream(file).pipe(res);
    });
  }

  function notFound(req, res) {
    fs.readFile(path.join(config.publicDir, '404.html'), (err, buf) => {
      send(res, 404, err ? 'Page introuvable' : buf);
    });
  }

  // ---------- routeur ----------
  async function handle(req, res) {
    baseHeaders(res);
    try {
      let url;
      try { url = new URL(req.url, 'http://local'); }
      catch { return json(res, 400, { error: 'invalid_url' }); }
      const p = url.pathname;
      const m = req.method;
      if (await member.handle(req, res, url)) return;
      if (p.startsWith('/api/') || p.startsWith('/lettre/') || p === '/desinscription' || p === '/inscription') await retention.run();
      if (p === '/api/internal/newsletter/send') {
        if (m !== 'POST') return json(res, 405, { error: 'method_not_allowed' });
        const provided = String(req.headers.authorization || ''), expected = 'Bearer ' + config.newsletter.workerKey;
        const loopback = ['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress);
        if (!loopback || !config.newsletter.enabled || Buffer.byteLength(provided) !== Buffer.byteLength(expected) || !crypto.timingSafeEqual(Buffer.from(provided), Buffer.from(expected))) return json(res, 403, { error: 'forbidden' });
        return json(res, 200, await newsletter.sendWeekly());
      }
      if (p === '/api/admin/newsletter/export') {
        if (m !== 'GET') return json(res, 405, { error: 'method_not_allowed' });
        if (!requireAdmin(req, res)) return;
        return json(res, 200, { schema_version: 1, exported_at: new Date(now()).toISOString(), subscribers: store.all().map(s => ({ id: s.id, email: s.email, newsletter: publicNewsletter(s.newsletter) })) }, { 'Content-Disposition': 'attachment; filename="hyprl-newsletter-consents.json"' });
      }
      if (p === '/lettre/confirmation' || p === '/lettre/desabonnement') {
        if (!['GET', 'HEAD', 'POST'].includes(m)) return json(res, 405, { error: 'method_not_allowed' });
        const token = m === 'POST' ? (await readForm(req)).get('token') || url.searchParams.get('token') : url.searchParams.get('token');
        const confirmation = p === '/lettre/confirmation';
        const person = confirmation ? newsletter.pending(token) : newsletter.findUnsubscribe(token);
        res.setHeader('Referrer-Policy', 'no-referrer');
        res.setHeader('X-Robots-Tag', 'noindex, nofollow');
        if (!person) return send(res, 404, pages.error('Lien invalide ou expiré', 'Vous pouvez faire une nouvelle demande depuis le formulaire d’accès anticipé.'));
        if (m !== 'POST') {
          const action = confirmation ? 'Confirmer mon inscription à la lettre' : 'Me désabonner de la lettre';
          return send(res, 200, pages.error(action, confirmation ? 'Votre choix sera activé après validation ci-dessous.' : 'L’accès anticipé est conservé. Vous ne recevrez plus la lettre.').replace('<p><a class="btn', '<form method="post" action="' + p + '"><input type="hidden" name="token" value="' + esc(token) + '"><button class="btn btn-primary" type="submit">' + action + '</button></form><p><a class="btn'));
        }
        // The no-referrer token page can submit Origin:null. Its unguessable, expiring,
        // single-use bearer token still binds the action; explicit foreign origins fail.
        if (confirmation && req.headers.origin !== 'null' && !sameOrigin(req)) return json(res, 403, { error: 'forbidden' });
        const ok = confirmation ? await newsletter.confirm(token) : await newsletter.unsubscribe(token);
        return send(res, ok ? 200 : 404, pages.error(ok ? (confirmation ? 'Votre inscription à la lettre est confirmée.' : 'Vous êtes désabonné de la lettre.') : 'Lien invalide ou expiré', confirmation ? 'Vous recevrez la prochaine lettre hebdomadaire gratuite.' : 'Votre choix a été enregistré.'));
      }
      if (p === '/api/subscribe') return m === 'POST' ? await subscribe(req, res, true) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'POST' });
      if (p === '/inscription') return m === 'POST' ? await subscribe(req, res) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'POST' });
      if (p === '/desinscription') {
        if (m === 'GET' || m === 'HEAD') return await unsubscribeGet(req, res, url);
        if (m === 'POST') return await unsubscribePost(req, res, url);
        return json(res, 405, { error: 'method_not_allowed' }, { Allow: 'GET, POST' });
      }
      if (p === '/api/admin/login') return m === 'POST' ? await login(req, res) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'POST' });
      if (p === '/api/admin/logout') return m === 'POST' ? logout(req, res) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'POST' });
      if (p === '/api/admin/subscribers') return m === 'GET' ? listSubscribers(req, res) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'GET' });
      const exp = p.match(/^\/api\/admin\/subscribers\/([0-9a-f-]{36})\/export$/i);
      if (exp) return m === 'GET' ? exportSubscriber(req, res, exp[1].toLowerCase()) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'GET' });
      const del = p.match(/^\/api\/admin\/subscribers\/([0-9a-f-]{36})$/i);
      if (del) return m === 'DELETE' ? await deleteSubscriber(req, res, del[1].toLowerCase()) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'DELETE' });
      if (p.startsWith('/api/')) return json(res, 404, { error: 'not_found' });

      if (m !== 'GET' && m !== 'HEAD') return send(res, 405, 'Méthode non autorisée', 'text/plain; charset=utf-8', { Allow: 'GET, HEAD' });
      if (p === '/contact') {
        const contact = config.notifyEmail
          ? '<p><a href="mailto:' + esc(config.notifyEmail) + '">' + esc(config.notifyEmail) + '</a></p>'
          : '<p>Contact à compléter par l’éditeur dans sa configuration privée. Consultez les mentions légales.</p>';
        return send(res, 200, pages.error('Contact et exercice de vos droits',
          'Précisez votre demande (accès, export JSON, rectification, effacement, retrait du consentement ou sécurité) et l’adresse concernée. Réponse sous un mois. N’envoyez pas de justificatif d’identité spontanément.').replace('</h1>', '</h1>' + contact));
      }
      if (p === '/.well-known/security.txt') {
        const body = 'Contact: ' + config.publicUrl + '/contact\nExpires: ' +
          new Date(now() + 180 * 86400000).toISOString() + '\nPreferred-Languages: fr, en\nCanonical: ' +
          config.publicUrl + '/.well-known/security.txt\nPolicy: ' + config.publicUrl + '/securite\n';
        return send(res, 200, m === 'HEAD' ? '' : body, 'text/plain; charset=utf-8');
      }
      if (p === '/sitemap.xml') {
        const locations = ['/', '/mentions-legales', '/confidentialite', '/cgv', '/accessibilite', '/contact', '/securite', '/resiliation', ...content.paths()];
        const body = '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' +
          locations.map((loc) => '<url><loc>' + esc(config.publicUrl + loc) + '</loc></url>').join('') + '</urlset>';
        return send(res, 200, m === 'HEAD' ? '' : body, 'application/xml; charset=utf-8');
      }
      if (p === '/robots.txt') return send(res, 200, m === 'HEAD' ? '' :
        'User-agent: *\nAllow: /\nDisallow: /admin\nDisallow: /api/\nDisallow: /membre\nDisallow: /auth/\nDisallow: /lettre/\nDisallow: /desinscription\nDisallow: /merci\nSitemap: ' + config.publicUrl + '/sitemap.xml\n', 'text/plain; charset=utf-8');
      const article = content.page(p);
      if (article) {
        const { title, description, body, status = 200 } = article;
        const schema = article.schema || content.schema(p, title, description);
        const rendered = pages.content(title, description, config.publicUrl + p, body, schema);
        return send(res, status, m === 'HEAD' ? '' : rendered.html, undefined, { 'Content-Security-Policy': CSP.replace("script-src 'self'", "script-src 'self' 'sha256-" + crypto.createHash('sha256').update(rendered.ld).digest('base64') + "'"), ...(status === 503 ? { 'Retry-After': '1800' } : {}) });
      }
      return serveStatic(req, res, p);
    } catch (err) {
      if (err && err.status) return json(res, err.status, { error: err.message });
      logger.error('[http] erreur interne');
      if (!res.headersSent) json(res, 500, { error: 'internal_error' });
      else res.end();
    }
  }

  handle.store = store;
  handle.sessions = sessions;
  handle.member = member;
  handle.newsletter = newsletter;
  handle.close = () => { clearInterval(sweeper); retention.close(); member.close(); };
  return handle;
}

module.exports = { createApp, ALLOWED_CHANNELS };
