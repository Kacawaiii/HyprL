'use strict';
// Application HTTP HyprL : fichiers statiques, inscription, désinscription, administration.
// Aucune dépendance hors Node.js, sauf nodemailer pour l'envoi d'e-mails.

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const { CONSENT_TEXTS, CURRENT_VERSION, assertPageMatches } = require('./lib/consent');
const { SubscriberStore } = require('./lib/store');
const { passwordMatches, AttemptLimiter, SessionStore, sha256 } = require('./lib/auth');
const { createMailer, notificationMessage, confirmationMessage } = require('./lib/mail');
const { createPages, maskEmail } = require('./lib/pages');

const ALLOWED_CHANNELS = ['discord', 'slack', 'telegram', 'teams', 'email', 'webhook', 'api'];
const EMAIL_RE = /^[^\s@<>()",;:\\[\]]+@[^\s@<>()",;:\\[\]]+\.[^\s@<>()",;:\\[\]]{2,}$/;
const MAX_BODY = 16 * 1024;
const COOKIE = 'hyprl_admin';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.ico': 'image/x-icon',
  '.woff2': 'font/woff2', '.txt': 'text/plain; charset=utf-8', '.xml': 'application/xml', '.webmanifest': 'application/manifest+json',
};

const CSP = [
  "default-src 'self'", "script-src 'self'", "style-src 'self'", "font-src 'self'", "img-src 'self' data:",
  "connect-src 'self'", "form-action 'self'", "frame-ancestors 'none'", "base-uri 'none'", "object-src 'none'",
].join('; ');

function createApp(config, deps = {}) {
  const now = deps.now || Date.now;
  const logger = deps.logger || console;

  const indexHtml = fs.readFileSync(path.join(config.publicDir, 'index.html'), 'utf8');
  assertPageMatches(indexHtml);

  const store = deps.store || new SubscriberStore(config.dataFile).load();
  const mailer = deps.mailer || createMailer(config, logger);
  const pages = createPages(config.publicDir);
  const loginLimiter = new AttemptLimiter({ max: config.login.maxAttempts, windowMs: config.login.windowMs, now });
  const subscribeLimiter = new AttemptLimiter({ max: config.subscribeLimit.max, windowMs: config.subscribeLimit.windowMs, now });
  const sessions = new SessionStore({ idleMs: config.session.idleMs, maxMs: config.session.maxMs, now });
  const publicOrigin = new URL(config.publicUrl).origin;

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
  const json = (res, status, obj, extra) => send(res, status, obj === undefined ? '' : JSON.stringify(obj), 'application/json; charset=utf-8', extra);
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
        if (size > MAX_BODY) { reject(Object.assign(new Error('too_large'), { status: 413 })); req.destroy(); return; }
        chunks.push(c);
      });
      req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
      req.on('error', reject);
    });
  }

  async function readForm(req) {
    const type = String(req.headers['content-type'] || '');
    const raw = await readBody(req);
    if (type.startsWith('application/json')) {
      let obj;
      try { obj = JSON.parse(raw || '{}'); } catch { throw Object.assign(new Error('bad_json'), { status: 400 }); }
      return { get: (k) => (obj[k] == null ? null : String(obj[k])), getAll: (k) => [].concat(obj[k] ?? []).map(String), json: true };
    }
    const p = new URLSearchParams(raw);
    return { get: (k) => p.get(k), getAll: (k) => p.getAll(k), json: false };
  }

  function cookie(req, name) {
    const header = req.headers.cookie || '';
    for (const part of header.split(';')) {
      const i = part.indexOf('=');
      if (i > 0 && part.slice(0, i).trim() === name) return decodeURIComponent(part.slice(i + 1).trim());
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
    return origin === publicOrigin || origin === hostOrigin;
  }

  function newToken() {
    const token = crypto.randomBytes(32).toString('base64url');
    return { token, hash: sha256(token).toString('hex') };
  }

  const unsubscribeUrl = (token) => config.publicUrl + '/desinscription?token=' + encodeURIComponent(token);

  function deliver(msg, label) {
    Promise.resolve()
      .then(() => mailer.send(msg))
      .catch((err) => logger.error('[mail] échec de l’envoi (' + label + ') : ' + (err && err.message)));
  }

  // ---------- inscription ----------
  async function subscribe(req, res) {
    if (!sameOrigin(req)) return send(res, 403, pages.error('Requête refusée', 'Le formulaire doit être envoyé depuis le site HyprL. Aucune donnée n’a été enregistrée.'));
    const form = await readForm(req);
    const wantsJson = form.json || String(req.headers.accept || '').includes('application/json');
    const fail = (status, title, message) => wantsJson
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
    if (store.findByEmail(email)) {
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
    await store.add(record);

    deliver(notificationMessage(record, { to: config.notifyEmail, from: config.mailFrom }), 'notification');
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
      .map(({ unsubscribe_token_hash, ...rest }) => rest);
    return json(res, 200, { subscribers });
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
    const url = new URL(req.url, 'http://local');
    const p = url.pathname;
    const m = req.method;

    try {
      if (p === '/api/subscribe') return m === 'POST' ? await subscribe(req, res) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'POST' });
      if (p === '/desinscription') {
        if (m === 'GET' || m === 'HEAD') return await unsubscribeGet(req, res, url);
        if (m === 'POST') return await unsubscribePost(req, res, url);
        return json(res, 405, { error: 'method_not_allowed' }, { Allow: 'GET, POST' });
      }
      if (p === '/api/admin/login') return m === 'POST' ? await login(req, res) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'POST' });
      if (p === '/api/admin/logout') return m === 'POST' ? logout(req, res) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'POST' });
      if (p === '/api/admin/subscribers') return m === 'GET' ? listSubscribers(req, res) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'GET' });
      const del = p.match(/^\/api\/admin\/subscribers\/([0-9a-f-]{36})$/i);
      if (del) return m === 'DELETE' ? await deleteSubscriber(req, res, del[1].toLowerCase()) : json(res, 405, { error: 'method_not_allowed' }, { Allow: 'DELETE' });
      if (p.startsWith('/api/')) return json(res, 404, { error: 'not_found' });

      if (m !== 'GET' && m !== 'HEAD') return send(res, 405, 'Méthode non autorisée', 'text/plain; charset=utf-8', { Allow: 'GET, HEAD' });
      return serveStatic(req, res, p);
    } catch (err) {
      if (err && err.status) return json(res, err.status, { error: err.message });
      logger.error('[http] erreur interne : ' + (err && err.stack));
      if (!res.headersSent) send(res, 500, 'Erreur interne', 'text/plain; charset=utf-8');
      else res.end();
    }
  }

  handle.store = store;
  handle.sessions = sessions;
  handle.close = () => clearInterval(sweeper);
  return handle;
}

module.exports = { createApp, ALLOWED_CHANNELS };
