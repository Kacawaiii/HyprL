'use strict';
// Price-free aggregates from the completed, already public study. No collection here.
const study = require('./btc-study-evidence');
const pathname = '/chance-ou-pas/btc-vendredi-plus-bas-asie';
const revision = '98c7189b207e857777e188318b7248e87616ddd1';
const repository = 'https://github.com/Kacawaiii/HyprL/blob/' + revision;
const reportUrl = repository + '/docs/BTC_ASIA_STUDY_20261029.md';
const evidenceUrl = repository + '/docs/artifacts/btc_asia_study_result_v1.json';
const pct = value => (100 * value).toFixed(1).replace('.', ',');
const title = 'BTC le vendredi : le plus bas de la séance asiatique est-il cassé dans 60 % des cas ?';
const verdict = 'Aucun avantage vendredi établi';
const description = 'Étude historique BTC-USD : ' + pct(study.friday.rate) + ' % des vendredis contre ' + pct(study.other_days.rate) + ' % les autres jours. Aucun avantage vendredi établi.';
function page() {
  return { title, description, body: '<article><p class="eyebrow">Chance ou pas ? · étude du 29 octobre 2026</p><h1>' + title + '</h1>' +
    '<h2>Verdict : ' + verdict + '</h2><p>Le seuil annoncé de 60 % n’est pas confirmé. La fréquence observée est de <strong>' + pct(study.friday.rate) + ' %</strong> (' + study.friday.k + '/' + study.friday.n + ' vendredis), contre <strong>' + pct(study.other_days.rate) + ' %</strong> les autres jours (' + study.other_days.k + '/' + study.other_days.n + ').</p>' +
    '<p>L’intervalle de confiance à 95 % de la fréquence du vendredi va de ' + pct(study.friday.ci95[0]) + ' à ' + pct(study.friday.ci95[1]) + ' %. Le test bilatéral contre exactement 60 % donne p = 0,075 : il ne rejette pas 60 % au seuil de 5 %, ce qui ne confirme pas cette affirmation.</p>' +
    '<h2>La règle mesurée</h2><p>Sur les bougies horaires BTC-USD de Coinbase Exchange, on cherche une cassure stricte du plus bas de 02 h à 11 h, heure de Paris, après 11 h et avant minuit du même jour. Un simple contact ne compte pas. Les changements d’heure suivent Europe/Paris.</p>' +
    '<h2>Échantillon et comparaison</h2><p>Du 1er janvier 2018 au 31 août 2026 inclus : ' + study.observed_candles.toLocaleString('fr-FR') + ' bougies observées, ' + study.random_day.n.toLocaleString('fr-FR') + ' journées complètes et ' + study.excluded_days + ' journées incomplètes exclues, sans remplissage. L’année 2026 est partielle. Le vendredi dépasse la fréquence des autres jours dans ' + study.years_friday_beats_other_days + ' années sur ' + study.years_comparable + ', une comparaison descriptive.</p>' +
    '<p>La comparaison principale rééchantillonne 450 semaines ISO appariées, 10 000 fois. L’intervalle à 95 % de la différence vendredi moins autres jours va de −7,1 à +2,9 points de pourcentage. Il inclut zéro. La fréquence sur l’ensemble des jours est de ' + pct(study.random_day.rate) + ' % : le repère empirique n’est pas un pile ou face à 50 %.</p>' +
    '<h2>Ce que cette étude permet de dire</h2><p>Cette vérification historique est exploratoire. Elle mesure une fréquence, sans démontrer rendement, rentabilité après coûts, exécution possible ou effet causal du vendredi. Les intervalles de Wilson et le test binomial supposent des observations indépendantes ; les blocs hebdomadaires n’excluent pas une dépendance plus longue. Il ne s’agit pas d’une preuve sur un échantillon hors étude intact. Le périmètre protégé de septembre à novembre 2026 a été exclu.</p>' +
    '<h2>Sources et méthode reproductible</h2><p><a href="' + reportUrl + '">Lire le rapport complet et ses limites</a> · <a href="' + evidenceUrl + '">Consulter les agrégats publics</a>. Révision <code>' + revision + '</code>. Empreinte SHA-256 des agrégats : <code>' + study.evidence_sha256 + '</code>.</p><p><a href="/chance-ou-pas">Toutes les vérifications</a></p></article>' };
}
module.exports = { pathname, title, verdict, description, reportUrl, evidenceUrl, study, page };
