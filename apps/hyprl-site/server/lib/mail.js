'use strict';
// Envoi des e-mails : notification interne + confirmation à l'inscrit (avec lien de désinscription).

const CHANNEL_LABELS = {
  discord: 'Discord', slack: 'Slack', telegram: 'Telegram', teams: 'Microsoft Teams',
  email: 'E-mail', webhook: 'Webhook signé', api: 'API (bientôt)',
};

const parisFormat = new Intl.DateTimeFormat('fr-FR', {
  timeZone: 'Europe/Paris', year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', second: '2-digit',
});

function formatParis(iso) {
  return parisFormat.format(new Date(iso)) + ' (heure de Paris)';
}

function channelsText(list) {
  return list.length ? list.map((c) => CHANNEL_LABELS[c] || c).join(', ') : 'aucun';
}

/** Notification interne : uniquement date, adresse, canaux, actions suivies. */
function notificationMessage(sub, { to, from }) {
  return {
    from, to,
    subject: 'HyprL — nouvelle inscription',
    text: [
      'Date : ' + formatParis(sub.created_at),
      'Adresse : ' + sub.email,
      'Canaux : ' + channelsText(sub.channels),
      'Actions suivies : ' + (sub.actions || 'aucune'),
    ].join('\n') + '\n',
  };
}

function confirmationMessage(sub, { from, unsubscribeUrl }) {
  return {
    from,
    to: sub.email,
    subject: 'Votre inscription à l’accès anticipé HyprL',
    text: [
      'Bonjour,',
      '',
      'Votre inscription à l’accès anticipé HyprL est enregistrée.',
      '',
      'Canaux choisis : ' + channelsText(sub.channels),
      'Actions suivies : ' + (sub.actions || 'aucune'),
      '',
      'Pour vous désinscrire et effacer vos données :',
      unsubscribeUrl,
      '',
      'Information, pas un conseil en investissement.',
      '',
      'HyprL',
    ].join('\n') + '\n',
    list: { unsubscribe: { url: unsubscribeUrl, comment: 'Se désinscrire' } },
    headers: { 'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click' },
  };
}

function createMailer(config, logger = console) {
  if (!config.smtp.host) {
    logger.warn('[mail] SMTP non configuré : aucun e-mail ne sera envoyé.');
    return { send: async (msg) => { logger.warn('[mail] e-mail non envoyé (SMTP absent) : ' + msg.subject); return false; } };
  }
  const nodemailer = require('nodemailer');
  const transport = nodemailer.createTransport({
    host: config.smtp.host,
    port: config.smtp.port,
    secure: config.smtp.secure,
    auth: config.smtp.user ? { user: config.smtp.user, pass: config.smtp.pass } : undefined,
  });
  return {
    send: async (msg) => {
      await transport.sendMail(msg);
      return true;
    },
  };
}

module.exports = { createMailer, notificationMessage, confirmationMessage, formatParis, CHANNEL_LABELS };
