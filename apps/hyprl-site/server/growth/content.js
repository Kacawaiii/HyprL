'use strict';
const crypto = require('node:crypto');
const { Snapshots } = require('../member/snapshots');
const { esc } = require('../lib/pages');
const btc = require('./btc-study');
const digest = value => crypto.createHash('sha256').update(value).digest('hex');
const claimId = c => digest(c.claim).slice(0, 24);
const verdicts = { verified: 'Étayée par la source', not_supported: 'Non étayée par la source', inconclusive: 'La source ne permet pas de trancher' };
function weekAt(now) {
  const day = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(now);
  const start = new Date(day + 'T12:00:00Z');
  start.setUTCDate(start.getUTCDate() - (start.getUTCDay() + 6) % 7);
  const end = new Date(start); end.setUTCDate(end.getUTCDate() + 6);
  return { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
}
const sourceHtml = s => '<a href="' + esc(s.url) + '" rel="noopener">Consulter la source</a> · disponible le <time>' + esc(s.available_at) + '</time> · empreinte <code>' + esc(s.sha256) + '</code>';
class PublicContent {
  constructor(config, { now = Date.now, snapshots } = {}) {
    this.config = config; this.now = now;
    this.snapshots = snapshots || new Snapshots(config.publicSnapshotDir, { now });
  }
  read() {
    const v = this.snapshots.read();
    if (this.now() - Date.parse(v.generated_at) > 26 * 3600000) throw Object.assign(Error('public_content_stale'), { status: 503 });
    return v;
  }
  projection() {
    const v = this.read(), week = weekAt(this.now());
    return { ...week, generated_at: v.generated_at, provenance: v.provenance,
      calendar: v.calendar.filter(r => r.kind === 'earnings' && r.symbol && r.date >= week.start && r.date <= week.end).sort((a, b) => a.date.localeCompare(b.date) || a.symbol.localeCompare(b.symbol)),
      claims: [...new Map(v.claims.filter(c => verdicts[c.verdict] && Date.parse(c.verified_at) <= Date.parse(v.generated_at) && Date.parse(c.source.available_at) <= Date.parse(c.verified_at)).sort((a, b) => a.verified_at.localeCompare(b.verified_at)).map(c => [claimId(c), c])).values()],
      briefs: v.briefs.filter(b => b.date >= week.start && b.date <= week.end),
    };
  }
  paths() {
    let claims = []; try { claims = this.projection().claims; } catch {}
    return ['/calendrier-resultats-us', '/chance-ou-pas', btc.pathname, '/comprendre-depot-sec', '/parrainage', ...new Set(claims.map(c => '/chance-ou-pas/' + claimId(c)))];
  }
  schema(pathname, title, description, type = 'WebPage', extra = {}) {
    return { '@context': 'https://schema.org', '@type': type, name: title, description, url: this.config.publicUrl + pathname, inLanguage: 'fr-FR', publisher: { '@type': 'Organization', name: 'HyprL', url: this.config.publicUrl }, ...extra };
  }
  page(pathname) {
    if (pathname === btc.pathname) return { ...btc.page(), schema: this.schema(pathname, btc.title, btc.description, 'Article', { headline: btc.title, datePublished: '2026-10-29', dateModified: '2026-10-29', author: { '@type': 'Organization', name: 'HyprL' }, citation: [btc.reportUrl, btc.evidenceUrl] }) };
    if (pathname === '/comprendre-depot-sec') {
      const guide = require('./guide');
      return { ...guide, schema: this.schema(pathname, guide.title, guide.description, 'Article', { headline: guide.title, dateModified: '2026-10-28', author: { '@type': 'Organization', name: 'HyprL' } }) };
    }
    if (pathname === '/parrainage') return { title: 'Parrainage Discord HyprL', description: 'Invitez vos proches avec un lien personnel. Les règles de validation et les récompenses du parrainage HyprL.', body: '<p class="eyebrow">Communauté</p><h1>Partagez une information vérifiable.</h1><p>Dans le serveur HyprL, utilisez <code>/parrainage</code> pour obtenir votre invitation personnelle permanente et consulter votre compteur privé.</p><h2>Les récompenses</h2><ul><li>3 parrainages validés : un crédit d’un mois Premium, utilisable lorsque la facturation Stripe sera disponible.</li><li>10 parrainages validés : une place Fondateur, enregistrée dans votre compte de parrainage.</li></ul><p>Chaque palier est attribué une seule fois. Le crédit ne déclenche aucun paiement ni abonnement automatique.</p><h2>Quand un parrainage compte</h2><p>Le compte invité doit avoir au moins sept jours à son arrivée et rester sept jours complets dans le serveur. Les robots, auto-invitations et retours d’un même compte sont exclus. Une arrivée dont le lien ne peut pas être établi avec certitude reste sans attribution.</p><p>Un éventuel classement, activé séparément par l’équipe, utilise des pseudonymes de parrainage, sans nom ni identifiant Discord. Votre pseudonyme est visible dans la réponse privée de la commande.</p><p><a href="https://discord.gg/BCgmuk6kXE" rel="noopener">Rejoindre le Discord HyprL</a></p>' };
    if (pathname !== '/calendrier-resultats-us' && pathname !== '/chance-ou-pas' && !/^\/chance-ou-pas\/[a-f0-9]{24}$/.test(pathname)) return null;
    let p; try { p = this.projection(); } catch { p = null; }
    if (pathname === '/calendrier-resultats-us') {
      const title = 'Calendrier des résultats US', description = 'Les résultats d’entreprises américaines de la semaine en français : dates du calendrier HyprL, provenance et limites de confirmation.';
      let body = '<p class="eyebrow">Agenda de la semaine</p><h1>' + title + '</h1><p>Repérez les publications de résultats des sociétés américaines. Chaque date conserve la précision et la provenance de notre calendrier.</p>';
      if (!p) body += '<p role="status">Le calendrier vérifié est temporairement indisponible. Aucune date n’est affichée.</p>';
      else {
        body += '<h2>Semaine du ' + p.start + ' au ' + p.end + '</h2><p>Repères observés, susceptibles de changer. Une date sans source émetteur reste à confirmer ; elle ne permet pas de déduire une heure.</p>';
        body += p.calendar.length ? '<div class="table-wrap" tabindex="0" role="region" aria-label="Calendrier des résultats US"><table><caption>Résultats US de la semaine</caption><thead><tr><th scope="col">Date</th><th scope="col">Société (ticker)</th><th scope="col">Provenance</th></tr></thead><tbody>' + p.calendar.map(r => '<tr><td><time datetime="' + esc(r.date) + '">' + esc(r.date) + '</time></td><th scope="row">' + esc(r.symbol) + '</th><td>' + (r.source ? sourceHtml(r.source) : 'Repère observé, à confirmer auprès de l’émetteur. Reçu le ' + esc(r.evidence.received_at) + ' · empreinte <code>' + esc(r.evidence.sha256) + '</code>') + '</td></tr>').join('') + '</tbody></table></div>' : '<p>Aucune date de résultats admissible dans notre calendrier pour cette semaine. Cela ne signifie pas qu’aucune entreprise ne publie.</p>';
        body += '<p class="muted">Export du <time>' + esc(p.generated_at) + '</time>. Le calendrier couvre uniquement les éléments disponibles dans notre export.</p>';
      }
      body += '<h2>Lire le calendrier</h2><p>La date d’un résultat et la date d’un dépôt SEC peuvent différer. Vérifiez la rubrique investisseurs de la société pour connaître l’horaire de sa publication. Les fuseaux français et américain changent d’heure à des dates différentes.</p><p><a href="/comprendre-depot-sec">Comprendre un 8-K et un 10-Q</a></p>';
      return { title, description, body, status: p ? 200 : 503, schema: this.schema(pathname, title, description, 'CollectionPage') };
    }
    if (pathname === '/chance-ou-pas') return { title: 'Chance ou pas ?', description: 'Des affirmations de marché confrontées à leurs sources. Le verdict, la date de vérification et les limites de chaque examen.', body: '<p class="eyebrow">Vérification</p><h1>Chance ou pas ?</h1><h2><a href="' + btc.pathname + '">BTC le vendredi : le verdict de l’étude</a></h2><p>' + btc.description + '</p><p>Chaque article examine une affirmation avec sa source et sa date de vérification. Un verdict sur une affirmation ne mesure pas à lui seul le talent d’une méthode.</p>' + (p?.claims.length ? '<ul>' + [...new Map(p.claims.map(c => [claimId(c), c])).values()].map(c => '<li><a href="/chance-ou-pas/' + claimId(c) + '">' + esc(c.claim) + '</a> · ' + esc(verdicts[c.verdict]) + '</li>').join('') + '</ul>' : '<p>Les vérifications sourcées sont en cours de constitution. Aucun article n’est publié sans élément vérifié.</p>') + '<h2>Le duel et la part du hasard</h2><p>HypR-A1 et HypR-E1 publient leurs analyses avant de connaître le résultat. Les preuves sont consultables dans l’espace membre. Le compteur de chance reste en constitution tant que sa méthode et son échantillon ne permettent pas de conclure.</p>', schema: this.schema(pathname, 'Chance ou pas ?', btc.description, 'CollectionPage'), status: 200 };
    const c = p?.claims.find(c => pathname === '/chance-ou-pas/' + claimId(c));
    if (!c) return { status: p ? 404 : 503, title: 'Vérification indisponible', description: 'Cette vérification n’est pas disponible.', body: '<h1>Vérification indisponible</h1><p><a href="/chance-ou-pas">Consulter les vérifications disponibles</a></p>' };
    const title = 'Chance ou pas ? ' + c.claim;
    return { title, description: c.claim, schema: this.schema(pathname, title, c.claim, 'Article', { headline: title, dateModified: c.verified_at, author: { '@type': 'Organization', name: 'HyprL' }, citation: c.source.url }), body: '<article><p class="eyebrow">Chance ou pas ?</p><h1>' + esc(c.claim) + '</h1><h2>Verdict : ' + esc(verdicts[c.verdict]) + '</h2><p>Vérification enregistrée le <time>' + esc(c.verified_at) + '</time>.</p><h2>La source examinée</h2><p>' + sourceHtml(c.source) + '</p><h2>Portée de la vérification</h2><p>Ce verdict porte sur l’affirmation ci-dessus, à la date indiquée. Il ne démontre ni une relation de cause à effet, ni la capacité à prévoir les marchés. Consultez la publication liée pour son contexte complet et ses éventuelles corrections.</p><p><a href="/chance-ou-pas">Toutes les vérifications</a></p></article>' };
  }
  edition() {
    const p = this.projection();
    const lines = ['La lettre HyprL · semaine du ' + p.start, ''];
    for (const b of p.briefs) for (const f of b.facts) lines.push(f.summary, f.source.url, 'Source disponible : ' + f.source.available_at, '');
    if (p.calendar.length) lines.push('Calendrier des résultats US', ...p.calendar.map(r => r.date + ' · ' + r.symbol + (r.source ? ' · ' + r.source.url : ' · repère observé, date à confirmer auprès de l’émetteur')), this.config.publicUrl + '/calendrier-resultats-us', '');
    for (const c of p.claims) lines.push('Chance ou pas ? ' + c.claim, verdicts[c.verdict], c.source.url, this.config.publicUrl + '/chance-ou-pas/' + claimId(c), '');
    if (!p.briefs.some(b => b.facts.length) && !p.calendar.length && !p.claims.length) throw Object.assign(Error('newsletter_content_empty'), { status: 503 });
    return { id: p.start, subject: 'La lettre HyprL · semaine du ' + p.start, text: lines.join('\n') };
  }
}
module.exports = { PublicContent, weekAt, claimId };
