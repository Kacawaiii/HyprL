'use strict';
// Pages générées par le serveur (désinscription, erreurs), dans la même mise en page que le site.

const fs = require('node:fs');
const path = require('node:path');

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function createPages(publicDir) {
  const shell = fs.readFileSync(path.join(publicDir, '404.html'), 'utf8');
  const open = '<main tabindex="-1" id="contenu" class="page">\n    <div class="container">\n';
  const close = '    </div>\n  </main>';
  const start = shell.indexOf(open);
  const end = shell.indexOf(close, start);
  if (start < 0 || end < 0) throw new Error('Gabarit 404.html inattendu');
  const before = shell.slice(0, start + open.length).replace(/\s*<link rel="alternate"[^>]+>/g, '');
  const after = shell.slice(end);

  function render(title, bodyHtml) {
    return before.replace(/<title>[^<]*<\/title>/, '<title>' + esc(title) + ' — HyprL</title>') +
      '      <div class="content">\n' + bodyHtml + '\n      </div>\n' + after;
  }

  const back = '<p><a class="btn btn-secondary" href="/">Retour à l’accueil</a></p>';

  return {
    content(title, description, canonical, bodyHtml, schema) {
      const ld = JSON.stringify(schema).replace(/</g, '\\u003c');
      const html = render(title, bodyHtml).replace('<meta name="robots" content="noindex, nofollow">',
        '<meta name="description" content="' + esc(description) + '"><link rel="canonical" href="' + esc(canonical) + '">' +
        '<meta property="og:title" content="' + esc(title) + '"><meta property="og:description" content="' + esc(description) + '"><meta property="og:url" content="' + esc(canonical) + '"><meta property="og:locale" content="fr_FR">')
        .replace('</head>', '<script type="application/ld+json">' + ld + '</script></head>');
      return { html, ld };
    },
    error(title, message) {
      return render(title, '<h1>' + esc(title) + '</h1>\n<p>' + esc(message) + '</p>\n' +
        '<p><a class="btn btn-secondary" href="/#acces">Revenir au formulaire</a></p>');
    },
    unsubscribeConfirm(token, maskedEmail) {
      return render('Désinscription',
        '<h1>Se désinscrire de HyprL</h1>\n' +
        '<p>Adresse concernée : <span class="mono">' + esc(maskedEmail) + '</span></p>\n' +
        '<p>La désinscription efface définitivement votre adresse, vos canaux, vos actions suivies et la trace de votre consentement.</p>\n' +
        '<form method="post" action="/desinscription">\n' +
        '  <input type="hidden" name="token" value="' + esc(token) + '">\n' +
        '  <button class="btn btn-primary" type="submit">Confirmer la désinscription</button>\n' +
        '</form>');
    },
    unsubscribeDone() {
      return render('Désinscription effectuée',
        '<h1>Vous êtes désinscrit.</h1>\n<p>Vos données ont été effacées. Vous ne recevrez plus d’e-mail de HyprL.</p>\n' + back);
    },
    unsubscribeInvalid() {
      return render('Lien invalide',
        '<h1>Ce lien n’est plus valide.</h1>\n<p>L’inscription a peut-être déjà été supprimée. Si vous recevez encore des e-mails, contactez-nous via les coordonnées des mentions légales.</p>\n' + back);
    },
  };
}

function maskEmail(email) {
  const [local, domain] = email.split('@');
  const keep = local.length <= 2 ? local.slice(0, 1) : local.slice(0, 2);
  return keep + '•••@' + domain;
}

module.exports = { createPages, maskEmail, esc };
