# Crypto News Trader Intelligence — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Construire dans HyprL une IA de trading événementiel BTC/ETH, pilotée par Hermes, qui collecte et vérifie l’actualité, mesure la réaction du marché, produit des thèses structurées et un score de risque de sommet, puis demande une validation humaine sans jamais exécuter automatiquement en V1.

**Architecture:** Un collecteur déterministe et durable détecte les événements et conserve leur premier timestamp. Une couche de vérification relie chaque lead à une source primaire et déduplique les reprises. Hermes intervient seulement sur les événements filtrés pour raisonner sur la surprise, les conséquences de premier et second ordre, le caractère déjà pricé et les scénarios d’invalidation. Un moteur déterministe ajoute le contexte de marché, calcule les scores calibrables, journalise la recommandation et envoie une carte Telegram. Les résultats sont évalués prospectivement sur plusieurs horizons avant toute utilisation avec argent réel.

**Tech Stack:** Python 3.11+, dataclasses/Pydantic selon les conventions existantes, HTTP/RSS/APIs publiques autorisées, stockage JSONL ou SQLite append-only, données de marché multi-venue, pytest, Hermes cron pour l’orchestration périodique et Telegram pour la validation humaine. Aucun broker ni module d’exécution dans V1.

---

## 1. Positionnement et limites non négociables

- Le système ne promet pas de trouver exactement les sommets ni de « gagner énormément ».
- L’objectif mesurable est de détecter plus tôt les changements de régime et les événements à asymétrie favorable, avec une perte bornée lorsque la thèse est fausse.
- GPT-5.6 Sol Max est un moteur de recherche, vérification et raisonnement, pas un oracle de prix.
- Le score final ne doit jamais être la confiance auto-déclarée du LLM. Il doit être recalibré sur des résultats prospectifs.
- V1 couvre BTC et ETH uniquement.
- V1 ne contient aucun appel broker, aucune clé broker et aucune exécution automatique.
- Toute recommandation expire et doit comporter une invalidation objective.
- Les décisions `WAIT` et `NO TRADE` sont journalisées comme les décisions directionnelles.
- Aucune optimisation rétrospective des seuils sur les résultats prospectifs : toute modification crée une nouvelle version de protocole.

## 2. Contexte existant à réutiliser ou isoler

### À réutiliser conceptuellement

- `scripts/news_signal/protocol.py` : protocole figé, hash d’univers, champs invariants.
- `scripts/news_signal/shadow_daily.py` : observation immuable et écriture atomique.
- `scripts/news_signal/shadow_evaluate.py` : séparation collecte/évaluation, coûts, tests HAC et statut manual-review.
- `tests/news_signal/` : exemples de tests d’isolation empêchant une intégration accidentelle aux ordres.
- `src/hyprl/crypto/signals.py` et `policy.py` : calculs de contexte prix/volume à réutiliser seulement après audit causal.

### À ne pas connecter

- `src/hyprl/crypto/trader.py` et toute fonction `place_order`.
- Le portefeuille actions momentum/tendance et son Sharpe 0,65.
- Les anciens modèles XGBoost crypto comme arbitre final.
- FinBERT générique comme signal directionnel autonome.

### Constat de départ

- Le pipeline news actuel vise les actions et n’a encore aucune observation OOS mature.
- Le module crypto actuel est essentiellement technique et ne contient pas d’intelligence événementielle.
- La recherche actions a montré que le sentiment générique était faible ou de mauvais signe ; la nouvelle stratégie doit classifier des événements, leur surprise et leur propagation, pas simplement le ton positif/négatif.

## 3. Univers de sources et hiérarchie de preuve

### Tier 0 — sources primaires

- SEC, CFTC, Federal Reserve, Trésor, Maison-Blanche, tribunaux et régulateurs européens pertinents.
- Blogs/status pages officiels des exchanges.
- Blogs, gouvernance, dépôts GitHub et comptes vérifiés des protocoles.
- Émetteurs d’ETF et dépôts réglementaires.
- Sources officielles des stablecoins et custodians.

### Tier 1 — agences et médias financiers

- Reuters, AP, FT, Bloomberg, CNBC et médias équivalents lorsque l’accès est légalement disponible.

### Tier 2 — médias spécialisés crypto

- CoinDesk, The Block, Blockworks et sources comparables, utilisés comme leads à confirmer.

### Tier 3 — social et rumeurs

- X, Truth Social, Telegram, Discord et autres réseaux.
- Aucun trade sur un Tier 3 seul, sauf compte officiel vérifié et contenu authentifié.

### Règles de collecte

- Préférer API, RSS, webhooks et pages officielles ; respecter les conditions d’utilisation et ne contourner aucun paywall.
- Conserver `published_at`, `first_seen_at`, `retrieved_at`, URL canonique, auteur/source, hash du contenu utilisé et chaîne de corroboration.
- Dédupliquer les reprises d’une même dépêche.
- Identifier la source primaire derrière chaque article secondaire.
- Marquer `UNVERIFIED`, `CORROBORATED`, `PRIMARY_CONFIRMED` ou `RETRACTED`.

## 4. Taxonomie d’événements

Créer une taxonomie versionnée couvrant au minimum :

1. macro/liquidité : taux, inflation, emploi, dollar, obligations, crise géopolitique ;
2. réglementation/litige : ETF, SEC/CFTC, interdictions, fiscalité, décisions judiciaires ;
3. exchange/custody : solvabilité, retraits, panne, hack, listing/delisting ;
4. protocole : upgrade, bug, exploit, gouvernance, tokenomics ;
5. stablecoin : depeg, réserves, émission/destruction, réglementation ;
6. flux institutionnels : ETF, trésoreries d’entreprise, fonds souverains ;
7. dérivés/levier : funding, basis, open interest, liquidations, options ;
8. on-chain : transferts, dépôts exchange, mouvements de wallets identifiés ;
9. offre : unlocks, émission, burn, ventes forcées ;
10. social/politique : annonces de personnalités capables de déplacer le marché.

Chaque événement doit indiquer : actifs affectés, direction hypothétique, horizon, mécanisme causal, bénéficiaires/perdants de second ordre et conditions d’invalidation.

## 5. Scores ordinaux séparés

### Event Quality Score — qualité factuelle, 0 à 100

- fiabilité de la source : 0–20 ;
- confirmation primaire/corroboration : 0–15 ;
- nouveauté réelle : 0–15 ;
- surprise par rapport aux attentes : 0–15 ;
- pertinence directe pour BTC/ETH : 0–15 ;
- clarté du mécanisme causal : 0–10 ;
- précision temporelle et authenticité : 0–10.

Pénalités : rumeur/contradiction 0 à −25 ; reprise sans information nouvelle 0 à −15 ; source anonyme non confirmée 0 à −15.

### Trade Opportunity Score — qualité du trade, 0 à 100

- réaction encore incomplète : 0–20 ;
- confirmation prix/volume : 0–15 ;
- confirmation dérivés/funding/OI : 0–15 ;
- asymétrie rendement/risque : 0–20 ;
- liquidité/exécutabilité : 0–10 ;
- persistance probable : 0–10 ;
- cohérence macro/régime : 0–10.

Pénalités : événement déjà pricé 0 à −25 ; mouvement déjà supérieur au seuil ATR 0 à −20 ; marché contradictoire 0 à −20 ; invalidation impossible = score zéro.

Ajouter un `priced_in_score` ordinal de 0 à 100 avec ses composants et `priced_in_data_coverage`. Il résume uniquement les éléments observables à `frozen_at` : mouvement normalisé depuis `first_seen_at`, dérive avant l'événement, âge de l'information, diffusion déjà constatée, volume, funding, basis et OI disponibles. Il ne doit pas être nommé ou interprété comme une probabilité avant calibration OOS.

`priced_in_data_coverage` est un entier de 0 à 100 représentant le pourcentage des poids de composants préenregistrés alimentés par des données fraîches à `frozen_at` : 0 signifie aucune composante disponible, 100 toutes les composantes disponibles. Les poids manquants ne sont pas redistribués. Une donnée obligatoire absente peut forcer `UNKNOWN` ou `NO TRADE` même avec une couverture globale élevée.

Les pondérations sont une hypothèse V1 préenregistrée. Elles seront modifiées uniquement dans une nouvelle version après un échantillon prospectif suffisant. Aucune confiance auto-déclarée du LLM n'est une probabilité.

## 6. Score de risque de sommet

Le système ne prédit pas un prix exact de sommet. Il calcule un `Top Risk Score` de 0 à 100 avec :

1. euphorie informationnelle et densité de bonnes nouvelles ;
2. levier : funding, basis, open interest, liquidations et options ;
3. réponse décroissante aux bonnes nouvelles ;
4. distribution : échecs de breakout, volume de vente, divergences BTC/ETH ;
5. risque macro : Nasdaq, taux, dollar, liquidité, politique et réglementation.

États V1 : 0–39 normal ; 40–59 vigilance ; 60–74 distribution possible ; 75–100 risque élevé.

Un score élevé ne déclenche jamais automatiquement un short. Il exige une confirmation de structure et produit d’abord `REDUCE`, `WAIT` ou une surveillance renforcée.

Le score retourne `UNKNOWN` tant que les blocs « euphorie informationnelle » et « réponse décroissante aux bonnes nouvelles » ne disposent pas d'un historique prospectif suffisant. Un bloc obligatoire manquant n'est jamais repondéré silencieusement.

## 7. Playbooks V1

Limiter la V1 à quatre playbooks préenregistrés :

### A. Choc officiel sous-réagi

Événement primaire confirmé, surprise forte, mécanisme direct, réaction initiale incomplète, puis confirmation prix/volume. Horizon confirmatoire principal : 4 heures. Les horizons 5–15 minutes mesurent la latence, pas l'alpha principal.

### B. Sell-the-news

Événement positif anticipé, levier élevé, échec de continuation et rupture d’un niveau d’invalidation. Horizon confirmatoire principal : 24 heures. En V1, sortie limitée à `REDUCE` ou `WAIT`, sans short.

### C. Mauvaise nouvelle absorbée

Événement négatif confirmé mais absence de nouveaux plus bas, réduction du levier et reprise de structure. Horizon confirmatoire principal : 24 heures.

### D. Risque systémique

Hack, depeg, insolvabilité, retraits bloqués ou choc réglementaire. Protocole de sûreté distinct des stratégies alpha, avec diagnostics à 1 h, 4 h et 24 h. Priorité à `EXIT`, `REDUCE` ou `WAIT`; aucune tentative automatique d’acheter le point bas.

Chaque playbook définit entrée, non-trade, invalidation, horizon maximal, données obligatoires et métriques.

## 8. Sortie structurée de Hermes

```json
{
  "event_id": "sha256",
  "fact_status": "PRIMARY_CONFIRMED",
  "primary_source": "url",
  "corroborating_sources": ["url"],
  "published_at": "ISO-8601",
  "first_seen_at": "ISO-8601",
  "frozen_at": "ISO-8601",
  "event_type": "regulation",
  "affected_assets": ["BTC", "ETH"],
  "event_quality_score": 84,
  "direction": "bearish",
  "horizon": "4h-3d",
  "novelty": "high",
  "surprise": "UNKNOWN",
  "priced_in_score": 35,
  "priced_in_data_coverage": 80,
  "market_confirmation": "partial",
  "top_risk_score": null,
  "top_risk_status": "UNKNOWN",
  "playbook": "sell_the_news",
  "decision": "WAIT",
  "trade_opportunity_score": 58,
  "thesis": "mécanisme causal concis",
  "invalidation": "condition observable",
  "entry_zone": null,
  "stop_level": null,
  "target_zones": [],
  "max_holding_time": "24h",
  "risk_budget_bps": 25,
  "expires_at": "ISO-8601",
  "missing_evidence": ["funding multi-venue"]
}
```

Le validateur rejette toute analyse sans source primaire ou sans invalidation lorsque la décision est directionnelle. Un événement critique exige en plus un deuxième canal indépendant, une signature ou une confirmation technique : un canal officiel peut lui-même être compromis.

## 9. Gestion du risque V1

- shadow/manual uniquement ;
- budget de risque simulé maximal par thèse : `risk_budget_bps = 25` ;
- budget cumulé simulé : 50 points de base ;
- perte journalière simulée : 100 points de base ; perte hebdomadaire : 200 points de base ;
- deux thèses simultanées maximum ;
- aucun levier et aucun averaging down ;
- aucune entrée après expiration ;
- aucune décision si les données sont absentes ou stale ;
- toute rétractation génère une réévaluation urgente ;
- kill switch manuel permanent.

Ces limites ne garantissent pas la perte maximale en cas de gap ou panne.

## 10. Validation prospective

Le protocole principal commence prospectivement avec `first_seen_at` immuable, car les archives historiques de news comportent souvent des timestamps révisés et des données manquantes.

Horizons diagnostiques enregistrés : 5 min, 15 min, 1 h, 4 h, 24 h et 72 h. Chaque playbook possède toutefois un seul horizon confirmatoire principal figé avant collecte ; il est interdit de sélectionner après coup l'horizon gagnant.

Métriques : précision directionnelle, rendement net, expectancy en R, MFE/MAE, monotonie des bins de score, performance par type/source/playbook/score, faux positifs, drawdown, Sharpe/Sortino lorsque l’échantillon le permet, comparaison à buy-and-hold, momentum simple et décisions aléatoires aux mêmes timestamps. Brier et log-loss ne sont autorisés qu'après définition d'un label binaire et gel d'un calibrateur probabiliste.

Les 100 événements matures et 30 observations par playbook constituent uniquement un checkpoint opérationnel V0, jamais une preuve d'edge. Le seuil confirmatoire final sera fixé par power analysis après estimation prospective de la variance et de la dépendance entre clusters. Toute promotion exige coûts/latence inclus, correction du multiple testing, sous-périodes cohérentes et confirmation sur une nouvelle cohorte intacte.

Évaluer séparément la qualité de l'IA, ancrée à `frozen_at` plus un délai simulé figé, et la qualité du workflow humain, ancrée à `human_decided_at` plus son propre délai. `frozen_at` est l'unique nom du timestamp auquel l'analyse devient immuable ; le nom `t_freeze` est interdit dans le schéma et le code. Les deux performances ne doivent jamais être mélangées.

## 11. Orchestration Hermes

Hermes ne doit pas lancer une recherche LLM chaque minute :

1. collecteur léger durable sur RSS/APIs/webhooks ;
2. filtre déterministe et déduplication ;
3. Hermes invoqué seulement sur événements prioritaires ;
4. source primaire + corroborations + contexte marché ;
5. validateur déterministe du JSON ;
6. carte Telegram pour validation humaine ;
7. cron Hermes pour digest et évaluation des outcomes.

La documentation du runtime Hermes utilisé ici indique une limite dure d'environ trois minutes par run cron ; cette limite doit être revérifiée sur le profil/version déployé avant implémentation. Un daemon/webhook convient mieux à la détection continue ; cron convient aux digests, évaluations et files d’attente. `attach_to_session=true` est pertinent pour un briefing conversationnel.

## 12. Fichiers proposés

### Service isolé

- Create: `services/crypto_news/pyproject.toml` avec dépendances figées et sans Alpaca/broker.
- Create: `services/crypto_news/src/crypto_news/{__init__,models,source_registry,normalize,dedupe,verification,taxonomy,market_context,scoring,top_risk,playbooks,journal,telegram}.py`
- Interdire toute dépendance ou import depuis `hyprl.crypto`, dont `__init__.py` importe directement `CryptoTrader` et dont `signals.py` instancie ce trader.

### Scripts

- Create: `services/crypto_news/scripts/{collect,analyze_pending,evaluate_outcomes,daily_digest}.py`

### Recherche

- Create: `research/crypto_news/hypothesis.json`
- Create: `research/crypto_news/README.md`
- Create: `research/crypto_news/{events,analyses,outcomes}/`
- Create: `research/crypto_news/latest_report.md`

### Tests

- Create: `services/crypto_news/tests/test_models.py`
- Create: `services/crypto_news/tests/test_dedupe.py`
- Create: `services/crypto_news/tests/test_verification.py`
- Create: `services/crypto_news/tests/test_scoring.py`
- Create: `services/crypto_news/tests/test_top_risk.py`
- Create: `services/crypto_news/tests/test_playbooks.py`
- Create: `services/crypto_news/tests/test_journal_immutability.py`
- Create: `services/crypto_news/tests/test_no_execution_imports.py`
- Create: `services/crypto_news/tests/test_egress_allowlist.py`

## 13. Ordre d’implémentation

### Phase 1 — contrat et isolation

1. Créer un environnement séparé sans SDK ni secret broker.
2. Tests AST/dépendances interdisant les imports broker/trader.
3. Modèles `SourceReceipt`, `EventVersion`, `MarketSnapshot`, `Analysis`, `HumanDecision`, `Outcome` et `Retraction`.
4. Protocole V0 figé avec hash et validation stricte.
5. Tables insert-only sans `UPDATE/DELETE`, chaîne de hash, versions liées et export JSONL signé. Canonicaliser chaque payload selon RFC 8785 (JCS), puis calculer `record_hash = SHA-256(previous_record_hash || canonical_payload)`. Produire un manifeste quotidien contenant la période UTC, le nombre de lignes, le premier/dernier hash et le SHA-256 du fichier JSONL ; signer le manifeste avec Ed25519. Conserver seulement l'identifiant et la clé publique dans le dépôt. La clé privée reste hors dépôt et hors environnement des collecteurs/analyseurs ; un processus de signature dédié ne reçoit que le hash du manifeste. Toute rotation de clé crée un nouveau `key_id` et une entrée append-only.
6. Egress allowlist pour les domaines de sources et de données autorisés.

### Phase 2 — collecte et preuve

1. Registre Tier 0–3.
2. Collecteur RSS/API minimal avec fixtures.
3. Normalisation timestamps/URLs.
4. Déduplication.
5. Chaîne source primaire/corroborations.
6. Rejet des rumeurs non confirmées.

### Phase 3 — contexte marché

1. Prix/volume BTC et ETH multi-venue.
2. Rendements, ATR et réaction depuis les timestamps.
3. Funding/OI/liquidations après sélection d’une source stable.
4. Détection des données stale/contradictoires.
5. Persistance du snapshot exact utilisé.

### Phase 4 — raisonnement Hermes

1. Prompt spécialisé avec JSON obligatoire.
2. Sources et citations imposées.
3. Séparation faits/inférences/inconnues.
4. Meilleur argument opposé à la thèse.
5. Refus sans invalidation.
6. Validation avant journalisation.

### Phase 5 — scores et playbooks

1. Event Quality Score.
2. Trade Opportunity Score.
3. Top Risk Score.
4. Quatre playbooks sans exécution.
5. Tests des pénalités, données manquantes et expirations.

### Phase 6 — Telegram et journal

1. Carte compacte avec sources.
2. Feedback `APPROUVER`, `REFUSER`, `WAIT`, sans broker.
3. Décision humaine timestampée.
4. Digest quotidien continuable.

### Phase 7 — évaluation

1. Outcomes aux horizons figés.
2. Calibration, MFE/MAE, expectancy et baselines.
3. Rapport par source/type/playbook/score.
4. Versionnement du protocole.
5. Aucune promotion avant les seuils.

## 14. Validation prévue

```bash
cd /home/kyo/HyprL/services/crypto_news
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
cd /home/kyo/HyprL
.venv/bin/python -m pytest tests/news_signal/test_isolation.py -v
services/crypto_news/.venv/bin/python services/crypto_news/scripts/collect.py --fixture services/crypto_news/tests/fixtures
services/crypto_news/.venv/bin/python services/crypto_news/scripts/analyze_pending.py --dry-run
services/crypto_news/.venv/bin/python services/crypto_news/scripts/evaluate_outcomes.py --as-of YYYY-MM-DD
```

Vérifier aussi : aucun import broker/place_order ; événement immuable ; rejet sans source primaire ; rejet d’un trade sans invalidation ; prix strictement postérieurs à `first_seen_at` ; fixtures offline ; échec propre des sources indisponibles.

## 15. Risques et critères de réussite

Risques : Hermes n’est pas HFT ; données fiables parfois payantes ; hallucinations ; timestamps révisés ; overfitting ; coûts tokens ; régimes changeants. Les contrôles sont le filtrage déterministe, les sources primaires, le protocole immuable, les citations et la validation prospective.

Réussite V1 : collecte dédupliquée, preuves auditables, analyses BTC/ETH structurées, scores ordinaux explicables, Telegram humain, zéro chemin d’exécution, outcomes prospectifs immuables et capacité à produire souvent `NO TRADE`.

## 16. Décisions V0 retenues après revue contradictoire

1. **Sources et budget :** commencer gratuitement avec feeds officiels, Coinbase, Kraken et Deribit. Les news licenciées, le consensus macro historisé et l'on-chain premium restent hors V0 et sont marqués `UNKNOWN` lorsqu'ils manquent.
2. **Architecture :** daemon de collecte permanent, stockage insert-only/hash-chain, queue d'analyse séparée, Hermes sur les seuls candidats prioritaires, cron pour digests/outcomes.
3. **Hypothèses :** sous-réaction officielle à 4 h, mauvaise nouvelle absorbée à 24 h, sell-the-news à 24 h sans short ; risque systémique séparé de l'alpha.
4. **Sémantique :** `Event Quality`, `Trade Opportunity`, `priced_in` et `Top Risk` sont des indices ordinaux. Aucun champ `probability` en V0/V1.
5. **Sécurité :** service Python physiquement isolé, sans Alpaca, sans variable broker et avec egress allowlist. Telegram ne peut enregistrer qu'un feedback de recherche.
6. **Simulation par défaut à préenregistrer :** notional spot de 10 000 USD ; qualité IA mesurée à `frozen_at + 60 s` ; qualité humaine mesurée à `human_decided_at + 30 s` ; frais, spread et profondeur mesurés sur les venues de référence au lieu d'un coût fictif constant.
7. **Promotion :** V0 vérifie la plomberie et estime variance/latence. V1 est une cohorte intacte dont la taille est déterminée par power analysis. Sharpe OOS net supérieur à 1 reste un objectif de preuve, jamais un paramètre d'optimisation.

---

# Prompt de cadrage pour l’autre projet/session

```text
Tu travailles dans le projet HyprL situé à /home/kyo/HyprL.

Je souhaite réorienter une partie du projet vers une IA de trading crypto événementiel centrée sur BTC et ETH. Je ne veux pas un nouveau bot technique optimisé sur des bougies, ni une exécution automatique. Je veux une IA agissant comme un analyste-trader : elle collecte et vérifie l’actualité, comprend les mécanismes de premier et second ordre, mesure la réaction du marché, estime si l’information est déjà pricée, détecte les risques de distribution/sommet et propose une thèse avec invalidation. La décision finale reste humaine.

Avant de proposer quoi que ce soit, inspecte obligatoirement :
- BOT_AUDIT.md ;
- docs/PROJECT_SUMMARY.md ;
- docs/EQUITY_STRATEGY_RESEARCH.md ;
- scripts/news_signal/ ;
- research/news_shadow/ ;
- src/hyprl/crypto/ ;
- tests/news_signal/ ;
- .hermes/plans/2026-07-31_125601-crypto-news-trader.md.

Contraintes non négociables :
1. Aucun ordre, aucune connexion broker et aucune modification de compte.
2. BTC et ETH uniquement en V1.
3. Aucun levier en V1.
4. Sources primaires et timestamps de première détection obligatoires.
5. Séparer Event Quality Score, Trade Opportunity Score et Top Risk Score.
6. La confiance du LLM ne doit pas être utilisée comme probabilité non calibrée.
7. Toute thèse directionnelle doit contenir une invalidation objective, une expiration et les preuves manquantes.
8. Les décisions WAIT/NO TRADE doivent être conservées.
9. Validation prospective immuable ; pas de réécriture après observation du prix.
10. Réutiliser les bonnes idées du protocole news actions, mais ne pas mélanger les données actions et crypto.
11. Ne pas connecter src/hyprl/crypto/trader.py ni place_order.
12. Ne pas chercher à forcer un Sharpe >1 par optimisation ; démontrer l’edge hors échantillon.

Je veux d’abord une discussion critique, pas une implémentation immédiate. Produis une réponse en français qui :
- résume ce qui est réellement réutilisable dans le repo ;
- signale les composants obsolètes, dangereux ou incompatibles ;
- critique le plan si une hypothèse est irréaliste ;
- compare au moins deux architectures possibles pour la collecte et l’orchestration ;
- propose une hiérarchie concrète de sources officielles, médias et social ;
- définit comment mesurer nouveauté, surprise, information déjà pricée et confirmation marché ;
- définit un score de risque de sommet sans prétendre prédire le sommet exact ;
- propose les 3 ou 4 playbooks maximum à tester d’abord ;
- détaille le protocole prospectif, les métriques et les baselines ;
- estime les principales dépendances, coûts de données et limites de latence ;
- donne un verdict GO / NO-GO / GO SOUS CONDITIONS ;
- termine par les cinq décisions que nous devons prendre avant de coder.

Ne fabrique aucune performance et ne promets aucun gain. Appuie chaque constat sur les fichiers du projet ou sur une source externe vérifiable. Si une information manque, marque-la INCONNUE.
```
