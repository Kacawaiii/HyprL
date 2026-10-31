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

function digestMessage(subscribers, { to, from }) {
  return { from, to, subject: 'HyprL — bilan quotidien des inscriptions confirmées',
    text: 'Inscriptions confirmées : ' + subscribers.length + '\n\n' + subscribers.map(sub => [
      'Confirmation : ' + formatParis(sub.confirmed_at), 'Adresse : ' + sub.email,
      'Canaux : ' + channelsText(sub.channels), 'Actions suivies : ' + (sub.actions || 'aucune'),
    ].join('\n')).join('\n\n') + '\n' };
}
function confirmationMessage(sub, { from, confirmationUrl, unsubscribeUrl }) {
  return { from, to: sub.email, subject: 'Confirmez votre inscription à HyprL',
    text: [
      'Bonjour,', '', 'Une demande d’accès anticipé HyprL a été faite avec cette adresse.',
      sub.newsletter ? 'La lettre hebdomadaire gratuite a aussi été demandée séparément.' : '',
      'Pour confirmer les choix demandés, ouvrez ce lien puis validez le formulaire :', confirmationUrl, '',
      'Le lien expire dans 48 heures. Sans confirmation, aucun autre e-mail ne sera envoyé et la demande sera effacée après 7 jours.',
      'Si vous n’avez pas fait cette demande, ignorez cet unique e-mail.', '',
      'Pour annuler la demande et effacer vos données :', unsubscribeUrl, '',
      'Information, pas un conseil en investissement.', 'HyprL',
    ].filter(v => v !== undefined).join('\n') + '\n',
    list: { unsubscribe: { url: unsubscribeUrl, comment: 'Se désinscrire' } },
    headers: { 'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click', 'X-Mailin-Track-Clicks': '0', 'X-Mailin-Track-Opens': '0' },
  };
}

function createMailer(config, logger = console) {
  if (!config.smtp.host || !config.smtp.euApproved) {
    logger.warn('[mail] Envoi désactivé : SMTP absent ou validation UE/DPA manquante.');
    return { send: async (msg) => { logger.info('[mail] ' + JSON.stringify({ disabled: 1 })); return false; } };
  }
  const nodemailer = require('nodemailer');
  const transport = nodemailer.createTransport({
    host: config.smtp.host,
    port: config.smtp.port,
    secure: config.smtp.secure,
    requireTLS: !config.smtp.secure,
    connectionTimeout: 10000,
    greetingTimeout: 10000,
    socketTimeout: 30000,
    auth: config.smtp.user ? { user: config.smtp.user, pass: config.smtp.pass } : undefined,
  });
  return {
    send: async (msg) => {
      await transport.sendMail(msg);
      return true;
    },
  };
}

module.exports = { createMailer, digestMessage, confirmationMessage, formatParis, CHANNEL_LABELS };
