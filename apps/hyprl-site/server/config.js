'use strict';
// Configuration lue exclusivement depuis les variables d'environnement.

const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');

function bool(v, def = false) {
  if (v === undefined || v === '') return def;
  return /^(1|true|yes|oui)$/i.test(v);
}

function loadConfig(env = process.env) {
  const port = Number(env.PORT || 3000);
  const config = {
    root: ROOT,
    publicDir: path.join(ROOT, 'public'),
    dataFile: path.resolve(ROOT, env.DATA_FILE || 'data/subscribers.json'),
    host: env.HOST || '127.0.0.1',
    port,
    publicUrl: (env.PUBLIC_URL || 'http://localhost:' + port).replace(/\/+$/, ''),
    production: env.NODE_ENV === 'production',
    trustProxy: bool(env.TRUST_PROXY),
    adminPassword: env.ADMIN_PASSWORD || '',
    notifyEmail: env.NOTIFY_EMAIL || '',
    mailFrom: env.MAIL_FROM || '',
    smtp: {
      host: env.SMTP_HOST || '',
      port: Number(env.SMTP_PORT || 587),
      secure: bool(env.SMTP_SECURE),
      user: env.SMTP_USER || '',
      pass: env.SMTP_PASS || '',
    },
    login: { maxAttempts: 5, windowMs: 15 * 60 * 1000 },
    session: { idleMs: 30 * 60 * 1000, maxMs: 12 * 60 * 60 * 1000 },
    subscribeLimit: { max: 10, windowMs: 60 * 60 * 1000 },
  };

  const errors = [];
  if (!config.adminPassword) errors.push('ADMIN_PASSWORD est obligatoire (variable d’environnement).');
  else if (config.adminPassword.length < 20) errors.push('ADMIN_PASSWORD doit contenir au moins 20 caractères.');
  if (config.smtp.host && !config.mailFrom) errors.push('MAIL_FROM est obligatoire quand SMTP_HOST est défini.');
  if (config.production && !/^https:\/\//.test(config.publicUrl)) errors.push('PUBLIC_URL doit être en https:// en production.');
  if (errors.length) {
    const err = new Error('Configuration invalide :\n  - ' + errors.join('\n  - '));
    err.code = 'CONFIG';
    throw err;
  }
  config.secureCookies = /^https:\/\//.test(config.publicUrl);
  return config;
}

module.exports = { loadConfig };
