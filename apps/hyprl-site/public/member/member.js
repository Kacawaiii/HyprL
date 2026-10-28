'use strict';
(() => {
  const $ = id => document.getElementById(id), page = $('page'), notice = $('notice');
  const routes = { '': ['dashboard', 'Tableau de bord'], brief: ['briefs', 'Brief du jour'], radar: ['radar', 'Radar des événements'], calendrier: ['calendar', 'Calendrier'], duel: ['duel', 'HypR-A1 vs HypR-E1'], chance: ['luck', 'Compteur de chance'], screener: ['screener', 'Accélération du chiffre d’affaires'], 'chance-ou-pas': ['claims', 'Chance ou pas ?'], alertes: ['alerts', 'Alertes'], compte: ['account', 'Compte'], preuves: ['proofs', 'Preuves de séance'] };
  const route = location.pathname.replace(/^\/membre\/?/, ''), [feature, title] = routes[route] || routes[''];
  const kinds = { sec_8k: 'SEC · 8-K', sec_10q: 'SEC · 10-Q', sec_form4: 'SEC · Form 4', fed: 'Fed', cpi: 'CPI', nfp: 'NFP', earnings: 'Résultats d’entreprise' };
  const fmt = (value, digits = 2) => value == null ? 'Indéfini' : new Intl.NumberFormat('fr-FR', { maximumFractionDigits: digits }).format(value);
  const pct = value => value == null ? 'Indéfini' : fmt(value * 100, 1) + ' %';
  const date = value => value ? new Intl.DateTimeFormat('fr-FR', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' }).format(new Date(value)) + ' UTC' : 'En cours de constitution';
  let session, plots = [], chartObservers = [], redrawCharts;
  function el(tag, text, cls) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; }
  function append(parent, ...children) { children.forEach(x => parent.append(x)); return parent; }
  function link(label, href) {
    const a = el('a', label);
    try { const u = new URL(href, location.origin); if (!['https:', 'http:'].includes(u.protocol) || u.username || u.password || (u.protocol === 'http:' && u.origin !== location.origin)) return el('span', label); a.href = u.href; a.rel = 'noopener noreferrer'; } catch { return el('span', label); }
    return a;
  }
  function empty(parent = page, reason = 'Les données vérifiées seront publiées ici dès qu’elles seront disponibles.') { append(parent, append(el('section', undefined, 'empty-state'), el('h2', 'En cours de constitution'), el('p', reason))); }
  function intro(text) { page.append(el('p', text, 'member-intro')); }
  function message(text) { notice.textContent = text; }
  function errorMessage(error) {
    if (error.status === 401) { message('Votre session a expiré. Reconnectez-vous avec Discord.'); notice.append(' ', link('Se reconnecter', '/membre/connexion')); }
    else if (error.status === 403) message('Cet accès est indisponible pour votre rôle actuel. Consultez votre compte ou contactez l’équipe.');
    else if (error.status === 429) message('Trop de demandes. Attendez une minute avant de réessayer.');
    else message('Service indisponible ou données en cours de constitution. Réessayez plus tard.');
  }
  async function api(path, method = 'GET', data) {
    const response = await fetch('/api/member/v2/' + path, { method, credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.timeout(15000), headers: { ...(method !== 'GET' ? { 'Content-Type': 'application/json', 'X-CSRF-Token': session?.csrf || '' } : {}) }, ...(data ? { body: JSON.stringify(data) } : {}) });
    if (!response.ok) throw Object.assign(Error('request_failed'), { status: response.status });
    return response.json();
  }
  function metadata(data) {
    $('updated').textContent = data.generated_at ? 'Dernier export · ' + date(data.generated_at) : '';
    if (data.provenance === 'synthetic') message('Aperçu de test : données entièrement synthétiques. Aucun résultat réel.');
    else if (data.stale && data.generated_at) message('Cet export est ancien. La date affichée fait foi ; la mise à jour est en cours.');
  }
  function source(parent, value) {
    const p = el('div', undefined, 'source');
    append(p, link('Consulter la source', value.url), el('p', 'Publié : ' + date(value.published_at) + ' · Disponible : ' + date(value.available_at)));
    const details = append(el('details'), el('summary', 'Empreinte de la source'), el('code', value.sha256, 'digest'));
    append(parent, append(p, details));
  }
  function table(headers, rows, caption, parent = page) {
    const t = el('table'), head = el('tr'); headers.forEach(h => { const th = el('th', h); th.scope = 'col'; head.append(th); });
    const body = el('tbody');
    rows.forEach(values => { const row = el('tr'); values.forEach(v => { const cell = el('td'); if (v instanceof Node) cell.append(v); else cell.textContent = v; row.append(cell); }); body.append(row); });
    append(t, el('caption', caption), append(el('thead'), head), body);
    const wrap = el('div', undefined, 'table-wrap'); wrap.tabIndex = 0; wrap.setAttribute('role', 'region'); wrap.setAttribute('aria-label', caption); append(parent, append(wrap, t));
  }
  function dashboard(data) {
    metadata(data);
    append(page, append(el('section', undefined, 'welcome'), el('h2', 'Un point de repère pour votre journée.'), el('p', 'Retrouvez les faits, leurs sources et les éléments qui permettent de juger la qualité des prévisions. Chaque publication conserve sa date de disponibilité.')));
    if (!data.generated_at) empty();
    const grid = el('div', undefined, 'workspace-links');
    [['briefs', '/membre/brief', 'Lire le brief', 'Le contexte de la séance, puis les éditions précédentes.'], ['radar', '/membre/radar', 'Suivre les événements', 'Vos titres, les publications SEC et les rendez-vous macro.'], ['duel', '/membre/duel', 'Examiner le duel', 'Prévisions révélées, scores et preuves vérifiables.'], ['claims', '/membre/chance-ou-pas', 'Chance ou pas ?', 'Les affirmations examinées, avec leur source.']].forEach(([f, href, label, description]) => {
      const card = append(el('article'), append(el('h3'), link(label, href)), el('p', description));
      if (!session.features.includes(f)) card.append(el('span', f === 'duel' ? 'Premium requis' : 'Essentiel requis', 'access'));
      grid.append(card);
    });
    page.append(grid);
  }
  function facts(parent, rows) {
    if (!rows.length) return empty(parent);
    const list = el('ul', undefined, 'fact-list');
    rows.forEach(r => { const item = append(el('li'), el('p', kinds[r.kind] + (r.symbol ? ' · ' + r.symbol : ''), 'fact-kind'), el('p', r.summary)); source(item, r.source); list.append(item); }); parent.append(list);
  }
  function briefs(data) {
    intro('Le contexte du jour, sous forme de faits sourcés. Retrouvez chaque édition dans l’archive.'); metadata(data);
    if (!data.items.length) return empty();
    const editions = [...data.items].sort((a, b) => b.date.localeCompare(a.date));
    const label = el('label', 'Édition du brief'); label.htmlFor = 'edition'; const select = el('select'); select.id = 'edition';
    editions.forEach((r, i) => { const o = el('option', r.date); o.value = String(i); select.append(o); });
    const content = el('section');
    const render = () => { content.replaceChildren(el('h2', 'Brief du ' + editions[Number(select.value)].date), el('p', 'Publié : ' + date(editions[Number(select.value)].published_at), 'small muted')); facts(content, editions[Number(select.value)].facts); };
    select.addEventListener('change', render); append(page, label, select, content); render();
  }
  async function watchlistForm() {
    const data = await api('watchlist'), form = el('form');
    const label = el('label', 'Votre liste de suivi'); label.htmlFor = 'symbols';
    const input = el('input'); input.id = 'symbols'; input.value = data.symbols.join(', '); input.maxLength = 800; input.autocomplete = 'off'; input.setAttribute('aria-describedby', 'watchlist-help');
    const note = el('p', 'Symboles séparés par des virgules, 50 au maximum. Les événements Fed, CPI et NFP restent visibles pour tous les titres.', 'form-note'); note.id = 'watchlist-help';
    const button = el('button', 'Enregistrer la liste', 'btn btn-secondary'); button.type = 'submit';
    append(form, label, input, note, button);
    form.addEventListener('submit', async e => {
      e.preventDefault(); button.disabled = true;
      try { await api('watchlist', 'PUT', { symbols: input.value.split(',').map(x => x.trim().toUpperCase()).filter(Boolean) }); location.reload(); } catch (e) { errorMessage(e); } finally { button.disabled = false; }
    }); page.append(form);
  }
  async function radar(data, calendar) {
    intro(calendar ? 'Les rendez-vous macro et ceux de votre liste de suivi. Les repères de journée indiquent une date seule ; une heure n’est affichée que si elle est confirmée.' : 'Dépôts SEC 8-K, 10-Q et Form 4, résultats et publications macro. Les événements de société suivent votre liste privée.'); metadata(data);
    await watchlistForm();
    if (!data.items.length) return empty(page, 'Complétez votre liste de suivi. Les prochains faits vérifiés apparaîtront ici.');
    if (!calendar) return facts(page, data.items);
    table(['Date / horaire confirmé', 'Événement', 'Titre', 'Preuve'], [...data.items].sort((a, b) => a.date.localeCompare(b.date)).map(r => [r.precision === 'time' ? date(r.scheduled_at) : r.date + ' · heure inconnue', kinds[r.kind], r.symbol || 'Macro', r.source ? link('Source', r.source.url) : append(el('details'), el('summary', 'Repère de journée observé, date non confirmée par l’émetteur'), el('p', 'Reçu : ' + date(r.evidence.received_at)), el('code', r.evidence.sha256, 'digest'))]), 'Calendrier · précision de la date conservée');
  }
  function chart(parent, points, key, label, mode) {
    const panel = append(el('section', undefined, 'chart-panel'), el('h3', label)); const target = el('div', undefined, 'chart'); target.setAttribute('role', 'img'); target.setAttribute('aria-label', label + '. Valeurs accessibles dans le tableau qui suit.'); panel.append(target); parent.append(panel);
    const styles = getComputedStyle(document.documentElement), accent = styles.getPropertyValue('--accent').trim(), muted = styles.getPropertyValue('--muted').trim(), line = styles.getPropertyValue('--line').trim();
    const plot = new uPlot({ width: Math.max(180, target.clientWidth), height: 210, legend: { show: false }, cursor: { show: false }, select: { show: false }, scales: { x: { time: false }, y: mode === 'percentage' ? { range: [0, 1] } : {} }, axes: [{ label: 'Prévision observée', stroke: muted, grid: { stroke: line }, ticks: { stroke: line }, values: (_u, ticks) => ticks.map(x => Number.isInteger(x) ? String(x) : '') }, { stroke: muted, grid: { stroke: line }, ticks: { stroke: line }, values: (_u, ticks) => ticks.map(x => mode === 'percentage' ? pct(x) : fmt(x)) }], series: [{}, { label, stroke: accent, width: 2.5, points: { show: true, size: 5 } }] }, [points.map(p => p.n), points.map(p => p[key])], target);
    let previousWidth = Math.max(180, target.clientWidth);
    const observer = new ResizeObserver(() => { const width = Math.max(180, target.clientWidth); if (target.clientWidth && width !== previousWidth) { previousWidth = width; plot.setSize({ width, height: 210 }); } }); observer.observe(target); plots.push(plot); chartObservers.push(observer);
  }
  const modeLabel = mode => mode === 'INTRADAY' ? 'Intraday' : 'Régulier';
  const seriesLabel = r => r.agent + ' · ' + r.variant + ' · ' + r.horizon + ' · ' + modeLabel(r.mode);
  function metrics(data, luck) {
    metadata(data);
    if (luck) { intro('L’échantillon et les dépendances entre prévisions doivent être établis avant de publier une probabilité de chance.'); return empty(); }
    intro('Probabilité de surperformer le SPY à l’horizon annoncé. Chaque variante, horizon et mode garde son propre score. Seules les observations notées et reliées à une révélation publique alimentent les courbes.');
    if (!data.forecasts.length && !data.failures.length) return empty();
    if (data.baseline) append(page, el('p', 'Référence figée : ' + pct(data.baseline.probability) + ' · ' + data.baseline.successes + ' succès / ' + data.baseline.n + ' observations de développement, lissage de Laplace. Le même taux de référence est utilisé pour chaque horizon.'), link('Consulter le gel du taux de base', data.baseline.url));
    const label = el('label', 'Agent, variante, horizon et mode'); label.htmlFor = 'variant'; const select = el('select'); select.id = 'variant';
    data.series.forEach((r, i) => { const option = el('option', seriesLabel(r) + (r.n ? '' : ' · aucune observation')); option.value = String(i); select.append(option); });
    const first = data.series.findIndex(r => r.n > 0); select.value = String(first < 0 ? 0 : first);
    const content = el('div'); append(page, label, select, content);
    const render = () => {
      plots.forEach(p => p.destroy()); chartObservers.forEach(o => o.disconnect()); plots = []; chartObservers = []; content.replaceChildren();
      const s = data.series[Number(select.value)]; if (!s?.points.length) return empty(content, 'Aucune observation notée et révélée pour cet horizon. Aucun point n’est ajouté.');
      content.append(el('h2', seriesLabel(s)));
      chart(content, s.points, 'brier_skill', 'Brier skill cumulé vs taux de base figé');
      chart(content, s.points, 'rolling_hit_rate', 'Taux de réussite · 20 dernières prévisions', 'percentage');
      content.append(el('p', 'Brier = moyenne de (probabilité − observation)². Skill = 1 − Brier / Brier de référence ; négatif signifie moins bon que la référence. Réussite de prévision : seuil de probabilité ≥ 0,5. Fenêtre de 20 observations, plus courte au démarrage. Ordre : séance, heure de décision, symbole. Les observations d’une même séance sont dépendantes.', 'data-note'));
      table(['Observation', 'Séance / titre', 'Notation UTC', 'Réussites / n', 'Brier', 'Brier skill', 'Réussite glissante'], s.points.map(p => [String(p.n), p.session + ' / ' + p.symbol, date(p.at), p.hits + ' / ' + p.n, fmt(p.brier, 4), fmt(p.brier_skill, 4), pct(p.rolling_hit_rate)]), 'Valeurs observées · ' + seriesLabel(s), content);
    };
    select.addEventListener('change', render); redrawCharts = render; render();
    page.append(el('h2', 'Comparaison par horizon'));
    table(['Agent / variante / horizon / mode', 'n', 'Réussites de prévision', 'Brier', 'Brier skill', 'Réussite glissante', 'Trades réussis / notés', 'Échecs / MODEL_FAILURE'], data.series.map(s => [seriesLabel(s), String(s.n), String(s.hits), fmt(s.brier, 4), fmt(s.brier_skill, 4), pct(s.rolling_hit_rate), s.trade_hits + ' / ' + s.trade_n, s.failures + ' / ' + s.model_failures]), 'Scores séparés par horizon et mode');
    page.append(el('p', 'Les échecs sont des séances sans décision : le même compte est rappelé sur chaque horizon du mode concerné et ne s’additionne pas. Les réussites de trades sont celles du score amont, distinctes des réussites de prévision ; les trades news ont leur propre durée et restent hors de ce tableau.', 'data-note'));
    page.append(el('h2', 'Révélations par séance'));
    table(['Séance', 'Agent / mode', 'État', 'Preuve'], data.reveals.map(r => [r.session, r.agent + ' / ' + modeLabel(r.mode), r.status, link('Vérifier ' + r.sha256.slice(0, 12), '/membre/preuves?id=' + r.id)]), 'Records entiers publiés et engagements antérieurs');
    page.append(el('h2', 'Prévisions révélées et notées'));
    if (!data.forecasts.length) empty(); else table(['Séance', 'Agent / variante / horizon / mode', 'Événement', 'Probabilité', 'Observation', 'Record révélé'], data.forecasts.map(r => [r.session, seriesLabel(r), r.symbol + ' · surperformance vs SPY · ' + r.horizon, pct(r.probability), r.outcome ? 'Surperformance' : 'Pas de surperformance', link('Record de séance', '/membre/preuves?id=' + r.reveal_id)]), 'Prévisions effectivement notées, sans points simulés');
    if (data.failures.length) { page.append(el('h2', 'Échecs documentés')); table(['Séance', 'Agent / mode', 'Statut / motif', 'Preuve'], data.failures.map(f => [f.session, f.agent + ' / ' + modeLabel(f.mode), f.status + ' / ' + f.code, link('Vérifier l’échec', '/membre/preuves?id=' + f.reveal_id)]), 'Séances MODEL_FAILURE, comptées une fois par agent et mode'); }
  }
  function screener(data) {
    intro('Évolution factuelle du chiffre d’affaires déclaré à la SEC, sur trois trimestres consécutifs comparables. Tri alphabétique des symboles. Les variations peuvent refléter la saisonnalité, des acquisitions ou une révision comptable.'); metadata(data);
    if (!data.items.length) return empty();
    table(['Titre / CIK', 'Périodes / chiffre d’affaires déclaré', 'Croissance précédente', 'Croissance récente', 'Accélération'], data.items.map(r => {
      const periods = el('div'); r.periods.forEach(p => append(periods, el('p', p.start + ' → ' + p.end + ' · ' + p.revenue + ' ' + p.currency), link('Dépôt SEC ' + p.accession, p.source.url)));
      return [r.symbol + ' / ' + r.cik, periods, fmt(r.growth_previous) + ' %', fmt(r.growth_latest) + ' %', fmt(r.acceleration_pp) + ' points'];
    }), 'Croissances séquentielles, sans classement d’investissement');
  }
  function claims(data) {
    intro('Un registre des affirmations examinées. Chaque conclusion est datée et renvoie à sa source.'); metadata(data);
    if (!data.items.length) return empty();
    const list = el('ul', undefined, 'fact-list');
    data.items.forEach(r => { const item = append(el('li'), el('h2', r.claim), el('p', ({ verified: 'Vérifiée', not_supported: 'Non étayée', inconclusive: 'Conclusion indéterminée' })[r.verdict] + ' · ' + date(r.verified_at))); source(item, r.source); list.append(item); }); page.append(list);
  }
  async function proofs(data) {
    metadata(data); intro('La preuve porte sur le record entier de la séance. Le producteur vérifie les octets du fichier dans Git et son engagement antérieur ; la page ne recalcule aucune empreinte unitaire de prévision.');
    if (!data.reveal) {
      if (!data.items.length) return empty();
      return table(['Séance', 'Agent / mode', 'État', 'Preuve'], data.items.map(r => [r.session, r.agent + ' / ' + modeLabel(r.mode), r.status, link('Ouvrir la preuve', '/membre/preuves?id=' + r.id)]), 'Révélations de séance');
    }
    const r = data.reveal;
    append(page, el('h2', r.session + ' · ' + r.agent + ' · ' + modeLabel(r.mode)), el('p', 'État : ' + r.status), link('Ouvrir le fichier au commit de révélation', r.url), el('p', 'Commit : ' + r.ledger_commit), el('p', 'Fichier : ' + r.file), el('p', 'SHA-256 du fichier révélé :'), el('code', r.sha256, 'digest'), el('p', 'Publié : ' + date(r.published_at) + ' · Disponible : ' + date(r.available_at)), link('Ouvrir l’engagement antérieur dans le registre', r.commitment.url), el('p', 'Engagé : ' + date(r.commitment.committed_at) + ' · Publication confirmée : ' + date(r.commitment.published_at)), el('p', 'SHA-256 engagé :'), el('code', r.commitment.sha256, 'digest'));
    if (r.status === 'REVEALED') append(page, el('p', 'Après téléchargement brut du fichier, comparez son SHA-256 avec l’engagement du registre.'), el('pre', 'sha256sum ' + r.agent + '.json', 'proof-content'));
    else page.append(el('p', 'NO_DECISION : le fichier révélé est la notice README. Son hash est distinct de l’engagement d’échec ; aucun record de décision ni résultat nul n’est fabriqué.'));
  }
  function account(data) {
    intro('Vos accès suivent les rôles de votre compte Discord dans le serveur HyprL.');
    append(page, el('h2', 'Votre formule'), el('p', data.tier_label), el('h2', 'Renouvellement'), el('p', data.billing?.renewal_at ? date(data.billing.renewal_at) : 'En cours de constitution. Aucune échéance de paiement renseignée.'), el('h2', 'Gérer ou résilier'));
    if (data.billing) append(page, append(el('p'), link('Gérer mon abonnement', data.billing.manage_url)), append(el('p'), link('Résilier mon abonnement', data.billing.cancel_url)));
    else append(page, el('p', 'Aucun abonnement payant relié à cet espace. Le paiement n’est pas activé.'), link('Informations sur la résiliation', '/resiliation'));
    append(page, el('h2', 'Données personnelles'), link('Demander un export ou un effacement', '/contact'));
  }
  async function alerts(data) {
    intro('Choisissez vos canaux de réception. Les destinations et leurs secrets restent privés ; les adresses enregistrées ne sont jamais réaffichées.');
    const destinationArea = el('section'); page.append(destinationArea);
    const renderDestinations = async rows => {
      destinationArea.replaceChildren(el('h2', 'Vos destinations'));
      if (!rows.length) empty(destinationArea, 'Ajoutez votre premier canal pour recevoir les prochaines publications vérifiées.');
      else table(['Canal', 'État', 'Suivi', 'Actions'], rows.map(r => {
        const actions = el('div', undefined, 'table-actions');
        for (const [label, method, fields] of [[r.paused ? 'Reprendre' : 'Mettre en pause', 'PATCH', { destination: r.id, paused: !r.paused }], ['Supprimer', 'DELETE', { destination: r.id }]]) {
          const b = el('button', label); b.type = 'button'; b.setAttribute('aria-label', label + ' ' + r.channel + ' ' + r.id.slice(0, 8));
          b.addEventListener('click', async () => { b.disabled = true; try { await api('alerts', method, fields); await renderDestinations((await api('alerts')).destinations); message('Destination mise à jour.'); } catch (e) { errorMessage(e); b.disabled = false; } }); actions.append(b);
        }
        return [r.channel + ' · ' + r.id.slice(0, 8), r.paused ? 'En pause' : 'Active', r.watchlist.join(', ') || 'Tous les symboles', actions];
      }), 'Destinations de votre compte uniquement', destinationArea);
    };
    await renderDestinations(data.destinations);
    const form = el('form'); form.autocomplete = 'off';
    function field(labelText, id, type = 'text') { const label = el('label', labelText); label.htmlFor = id; const input = el('input'); input.id = id; input.type = type; input.maxLength = 2048; append(form, label, input); return input; }
    form.append(el('h2', 'Ajouter un canal'));
    const label = el('label', 'Canal'); label.htmlFor = 'channel'; const select = el('select'); select.id = 'channel'; ['discord', 'slack', 'telegram', 'teams', 'webhook'].forEach(v => { const o = el('option', v === 'webhook' ? 'Webhook personnalisé' : v[0].toUpperCase() + v.slice(1)); o.value = v; select.append(o); }); append(form, label, select);
    const endpoint = field('URL privée du webhook', 'endpoint', 'url'); endpoint.required = true; endpoint.setAttribute('aria-describedby', 'destination-help');
    const key = field('Clé HMAC du webhook personnalisé (32 à 128 caractères)', 'signing-key', 'password'); key.maxLength = 128; key.minLength = 32;
    const symbols = field('Symboles suivis pour ce canal (facultatif)', 'alert-symbols');
    const note = el('p', 'Les symboles sont séparés par des virgules. Telegram nécessite un identifiant de conversation préalablement autorisé par l’équipe. L’e-mail est en cours de constitution ; aucune adresse n’est collectée.', 'form-note'); note.id = 'destination-help'; form.append(note);
    const update = () => { endpoint.type = select.value === 'telegram' ? 'text' : 'url'; form.querySelector('label[for="endpoint"]').textContent = select.value === 'telegram' ? 'Identifiant de conversation Telegram' : 'URL privée du webhook'; key.disabled = select.value !== 'webhook'; key.required = select.value === 'webhook'; key.hidden = key.disabled; form.querySelector('label[for="signing-key"]').hidden = key.disabled; endpoint.value = ''; key.value = ''; }; select.addEventListener('change', update); update();
    const button = el('button', 'Ajouter la destination', 'btn btn-primary'); button.type = 'submit'; form.append(button); page.append(form);
    form.addEventListener('submit', async e => {
      e.preventDefault(); button.disabled = true;
      const config = select.value === 'telegram' ? { chat_id: endpoint.value.trim() } : { url: endpoint.value.trim(), ...(select.value === 'webhook' ? { signing_key: key.value } : {}) };
      const payload = { channel: select.value, config, kinds: ['sec_filing', 'fed_release', 'cpi_release', 'nfp_release', 'earnings_date', 'earnings_result', 'radar_headline', 'duel_hashes', 'duel_results', 'weekly_luck'], watchlist: symbols.value.split(',').map(x => x.trim().toUpperCase()).filter(Boolean) };
      endpoint.value = ''; key.value = '';
      try { await api('alerts', 'POST', payload); await renderDestinations((await api('alerts')).destinations); message('Destination ajoutée. Les prochaines publications suivront vos préférences.'); } catch (e) { message('Ajout refusé ou routeur indisponible. Vérifiez le canal et les autorisations de votre compte.'); } finally { button.disabled = false; }
    });
  }
  $('theme').addEventListener('click', () => { document.documentElement.dataset.theme = (document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark' ? 'light' : 'dark'; if (redrawCharts) redrawCharts(); });
  $('logout').addEventListener('click', async () => { try { await api('logout', 'POST', {}); location.replace('/membre/connexion'); } catch (e) { errorMessage(e); } });
  $('page-title').textContent = title; document.title = title + ' · HyprL';
  async function run() {
    try {
      session = await api('session'); $('tier').textContent = session.tier_label;
      document.querySelectorAll('[data-feature]').forEach(a => { a.dataset.entitled = String(session.features.includes(a.dataset.feature)); if (a.dataset.feature === (feature === 'proofs' ? 'duel' : feature)) a.setAttribute('aria-current', 'page'); });
      page.replaceChildren();
      if (!session.features.includes(feature)) { intro('Cet espace est compris dans une formule supérieure à votre accès actuel.'); page.append(link('Consulter mon compte', '/membre/compte')); return; }
      const proofId = new URLSearchParams(location.search).get('id');
      const path = feature === 'proofs' && proofId ? 'proofs/' + encodeURIComponent(proofId) : feature;
      let data;
      try { data = await api(path); }
      catch (e) { if (e.status === 503) { intro('Les publications vérifiées et les connexions se préparent.'); empty(); } throw e; }
      if (feature === 'dashboard') dashboard(data);
      else if (feature === 'briefs') briefs(data);
      else if (feature === 'radar' || feature === 'calendar') await radar(data, feature === 'calendar');
      else if (feature === 'duel' || feature === 'luck') metrics(data, feature === 'luck');
      else if (feature === 'screener') screener(data);
      else if (feature === 'claims') claims(data);
      else if (feature === 'proofs') await proofs(data);
      else if (feature === 'account') account(data);
      else if (feature === 'alerts') await alerts(data);
    } catch (e) { errorMessage(e); }
  }
  run();
})();
