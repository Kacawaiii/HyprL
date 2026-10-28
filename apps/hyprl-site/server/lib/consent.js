'use strict';
// Texte EXACT des cases de consentement, par version.
// Le serveur vérifie au démarrage que public/index.html affiche mot pour mot le texte de CURRENT_VERSION.
// Pour modifier le texte : créer une NOUVELLE version (ne jamais réécrire une version existante,
// les inscriptions passées doivent garder le texte qu'elles ont réellement accepté).

const CONSENT_TEXTS = Object.freeze({
  '2026-10-v1':
    'J’accepte que HyprL enregistre mon adresse e-mail, les canaux choisis et les actions suivies ' +
    'afin de me contacter au sujet de l’accès anticipé. Je peux me désinscrire et faire effacer ' +
    'ces données à tout moment.',
});

const CURRENT_VERSION = '2026-10-v1';

function normalizeSpaces(s) {
  return s.replace(/\s+/g, ' ').trim();
}

/** Vérifie que la page publique affiche exactement le texte de la version courante. */
function assertPageMatches(html) {
  const m = html.match(/<span id="consent-text">([\s\S]*?)<\/span>/);
  if (!m) throw new Error('Consentement : <span id="consent-text"> introuvable dans index.html');
  const shown = normalizeSpaces(m[1]);
  const expected = normalizeSpaces(CONSENT_TEXTS[CURRENT_VERSION]);
  if (shown !== expected) {
    throw new Error(
      'Consentement : le texte affiché dans index.html diffère de server/lib/consent.js (' + CURRENT_VERSION + ').\n' +
      '  page    : ' + shown + '\n  attendu : ' + expected
    );
  }
  const v = html.match(/name="consent_version" value="([^"]+)"/);
  if (!v || v[1] !== CURRENT_VERSION) {
    throw new Error('Consentement : le champ caché consent_version de index.html doit valoir ' + CURRENT_VERSION);
  }
}

module.exports = { CONSENT_TEXTS, CURRENT_VERSION, assertPageMatches, normalizeSpaces };
