# Market Data V1 — décision fournisseur

Statut vérifié le **2026-08-02**.

## Décision

**Coinbase Exchange REST est le candidat technique V1 pour la collecte research/shadow BTC/USD et ETH/USD.** Cette décision autorise uniquement l’adapter hors ligne, les fixtures locales et, dans une phase ultérieure, un collecteur interne sans capacité d’ordre.

Cette décision **n’autorise pas** la redistribution, l’affichage public ou l’exploitation commerciale des données Coinbase. La page légale n’était pas consultable automatiquement dans l’environnement de revue (protection Cloudflare). Une validation juridique écrite des conditions de stockage, dérivation, affichage et redistribution reste donc un gate bloquant avant toute commercialisation.

## Périmètre technique vérifié

- Endpoint documenté : `GET /products/{product_id}/candles`.
- Produits publics vérifiés le 2026-08-02 :
  - `BTC-USD` — statut `online` ;
  - `ETH-USD` — statut `online`.
- Granularités retenues :
  - `3600` secondes → `1h` ;
  - `86400` secondes → `1d`.
- Réponse fournisseur : lignes `[time, low, high, open, close, volume]`.
- Maximum officiel : 300 bougies par requête.
- Limite REST publique officielle : 10 requêtes/s par IP, burst jusqu’à 15.
- La documentation avertit que l’historique peut être incomplet et qu’aucune donnée n’est publiée pour les intervalles sans tick.
- La documentation déconseille l’interrogation fréquente de l’historique ; le temps réel devra utiliser un flux dédié dans une phase ultérieure.

## Sources primaires

- Bougies produit : <https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles>
- Limites REST : <https://docs.cdp.coinbase.com/exchange/rest-api/rate-limits>
- Métadonnées publiques vérifiées :
  - <https://api.exchange.coinbase.com/products/BTC-USD>
  - <https://api.exchange.coinbase.com/products/ETH-USD>
- Conditions de marché à faire valider juridiquement avant commercialisation : <https://www.coinbase.com/legal/market_data>

## Contrat d’intégration

1. Le collecteur futur conserve les octets exacts de chaque réponse avant parsing et calcule leur SHA-256.
2. L’adapter reçoit uniquement ces octets, des timestamps observés et le contexte produit/timeframe ; il n’effectue aucun accès réseau.
3. Seules les fenêtres entièrement closes sont demandées. Une bougie dont `bar_close_at > available_at` est refusée comme partielle.
4. Les réponses sont bornées à 300 lignes et à une taille maximale explicite avant décodage.
5. Les trous restent explicites : aucune bougie synthétique, aucun forward-fill et aucune interpolation dans la couche canonique.
6. Les révisions sont append-only. L’identité logique d’une barre reste stable ; son identité de version dépend des OHLCV normalisés, pas du découpage arbitraire des pages HTTP.
7. Les payloads malformés, valeurs non finies, doublons temporels, intervalles hors alignement ou actifs non autorisés sont rejetés en bloc.
8. Le collecteur futur devra respecter `429`, backoff borné, jitter, timeout et compteur de tentatives ; ces politiques ne font pas partie de l’adapter pur.
9. Aucun SDK broker, secret, credential, endpoint d’ordre ou import de `src.hyprl.crypto` n’est autorisé dans cette couche.

## Persistance MarketDataStore (Phase 1B)

Statut vérifié le **2026-08-03**. Implémentation : `scripts/trading_lab/market_data_store.py`.

`MarketDataStore.ingest_coinbase_response` reçoit un payload Coinbase brut déjà capturé (aucun accès réseau dans cette couche) et persiste, dans une seule transaction SQLite, les octets bruts et les `MarketBar` normalisées qui en découlent.

### Tables et rôle

- `raw_market_payloads` : un exemplaire des octets exacts d'une réponse fournisseur, indexé par `payload_sha256`. Un même payload brut n'est jamais stocké deux fois.
- `market_ingestions` : une entrée par appel d'ingestion, identifiée par un `ingestion_id` déterministe (hash des métadonnées canoniques : provider, produit, timeframe, `available_at`, `ingested_at`, hash du payload brut).
- `market_bar_receipts` : une `MarketBar` normalisée par barre et par ingestion, indexée par `content_sha256` (hash canonique de l'enregistrement complet).

### Append-only

Les trois tables portent des triggers SQL `BEFORE UPDATE`/`BEFORE DELETE` qui lèvent systématiquement `RAISE(ABORT, ...)`. L'immuabilité est appliquée au niveau base de données, pas seulement dans le code applicatif.

### Idempotence et replay

Un ré-ingestion du même payload brut, avec les mêmes métadonnées, ne réécrit rien : l'`ingestion_id` déterministe permet de détecter le cas et de retourner `exact_replays` sans insertion. Une ré-ingestion concurrente du même payload (deux threads) ne produit qu'un seul exemplaire stocké — la transaction `BEGIN IMMEDIATE` sérialise les écritures.

### Révisions MarketBar

Chaque barre a deux identités distinctes (héritées de `market_bar.py`) :

- `bar_id` : identité logique stable, dérivée de l'actif, la venue, le provider, le timeframe et les bornes temporelles de la barre. Elle ne change jamais pour une même barre.
- `bar_version_id` : identité d'une révision, dérivée de `bar_id` et des valeurs OHLCV normalisées — **pas** du hash du payload brut ni de `ingested_at`. Une ré-ingestion du même contenu à un instant `ingested_at` différent ne crée donc pas de nouvelle version ; une correction fournisseur qui change réellement l'OHLCV d'une barre crée une nouvelle version uniquement pour cette barre-là.

### Garanties atomiques et fail-closed

- Toute violation de contrat (payload malformé, record non conforme au schéma, hash `content_sha256` qui ne se recalcule pas à l'identique, `raw_payload_sha256` du record qui ne correspond pas au payload réellement stocké) fait échouer toute la transaction : aucune ligne partielle n'est écrite dans les 3 tables.
- Un conflit d'identité immuable (même hash de payload brut mais contenu différent, même `ingestion_id` mais métadonnées différentes, même `bar_id` occupé par un autre `content_sha256` dans la même ingestion) lève `MarketDataConflict` et annule la transaction.
- Un échec inattendu en cours de transaction (ex. contrainte SQL imprévue) déclenche un rollback complet ; une nouvelle tentative après correction réussit sans laisser de trace de la tentative avortée.

### Limites de taille et de cardinalité

- `MAX_RESPONSE_BYTES` (1 000 000 octets) est vérifié sur le payload brut **avant** tout appel à l'adapter Coinbase.
- `MAX_CANDLES_PER_RESPONSE` (300) est vérifié sur la sortie de l'adapter **avant** toute reconstruction de `MarketBar`, y compris si un adapter défaillant retournait davantage de lignes.

### Tests prouvant ces invariants

Tous dans `tests/crypto/test_market_data_store.py` (14 tests, exécutés le 2026-08-03, tous verts) :

- `test_market_data_store_persists_exact_raw_bytes_and_bar_receipts`
- `test_market_data_store_exact_replay_is_a_noop`
- `test_same_raw_payload_at_new_ingestion_time_creates_receipts_not_versions`
- `test_provider_revision_creates_new_version_only_for_changed_bar`
- `test_invalid_response_leaves_store_empty_atomically`
- `test_unexpected_post_begin_failure_rolls_back_and_retry_succeeds`
- `test_market_data_tables_are_insert_only`
- `test_concurrent_exact_ingestions_store_one_copy`
- `test_preexisting_raw_hash_collision_fails_closed`
- `test_store_rejects_adapter_output_not_bound_to_raw_payload`
- `test_store_rebuilds_and_rejects_fully_rehashed_tampered_bar`
- `test_store_bounds_raw_input_before_calling_adapter`
- `test_store_bounds_adapter_record_count_before_rebuilding`

### Hors périmètre actuel

- Le collecteur réseau réel (requêtes HTTP vers Coinbase, gestion `429`, backoff, jitter, timeout) n'existe pas encore. `MarketDataStore` ne fait que persister des octets déjà capturés ailleurs.
- La **sélection** des snapshots causaux historiques est implémentée (Phase 1C-A/1C-B, voir plus bas) ; leur **matérialisation** persistée (tables `market_snapshot_*`, APIs `create_snapshot` / `load_snapshot` / `list_snapshot_bars`) ne l'est pas encore.
- Le backfill automatique, le retry, les alertes et toute forme de comblement (forward-fill, bougie synthétique) restent explicitement hors périmètre : la détection de gap ne fait qu'observer et journaliser, jamais combler.

## Détection persistée des gaps MarketBar (Phase 1B)

Statut : implémenté en TDD. **Cinq** contre-revues indépendantes (CR1 à CR5) plus une micro-revue read-only sur le contrat temporel de `ingested_at` ont eu lieu.

- **CR1** a confirmé 3 findings (F1 CRITICAL — amplification non bornée ; F2 MEDIUM — causalité DETECTED potentiellement antérieure à une borne nécessaire ; F3 LOW — documentation de cardinalité incomplète).
- **CR2**, après correction, a confirmé F1/F2 mais révélé que le correctif F1 avait **déplacé** l'amplification (rescan complet de l'historique du domaine à chaque ingestion) et introduit un **état absorbant irréversible** (plafond sur le stock cumulé du domaine, bloquant définitivement tout backfill historique discontinu) — corrigé par le passage à une détection **strictement intra-payload**.
- **CR3** a confirmé ce correctif mais révélé CR3-F1 (MEDIUM) : une ouverture manquante dans le payload courant mais **déjà connue du store via un autre payload** produisait un `DETECTED` factuellement faux, irréparable (table append-only) et non résoluble par l'ingestion qui le créait ; plus 3 findings LOW documentaires/couverture (CR3-F2, CR3-F3, CR3-F4).
- **CR4** a confirmé le correctif CR3-F1 mais révélé 3 findings LOW documentaires/couverture (CR4-F1 : attribution erronée d'une couverture de test à 10 000 barres ; CR4-F2 : garde anti-scan SQL inopérante ; CR4-F3 : test « non monotone » à l'oracle insuffisamment discriminant).
- Une **micro-revue read-only** a ensuite tranché le contrat sémantique de `ingested_at` : décision manager d'adopter explicitement le **Contrat A** — `ingested_at` est un timestamp historique déclaré par l'appelant, jamais l'heure réelle de réception du store (voir section dédiée plus bas).
- **CR5** a validé techniquement la gate (Contrat A cohérent, CR4-F1/F2/F3 fermés) mais a bloqué la clôture pour deux raisons : les trois fichiers de la gate n'étaient pas suivis par Git (CR5-F1, résolu séparément par l'ajout à l'index), et 4 findings LOW résiduels (CR5-F2 : compteurs de tests obsolètes dans cette doc ; CR5-F3 : le test contractuel R1/R2 ne discriminait pas à lui seul une implémentation « dernier receipt physiquement inséré » ; CR5-F4 : ambiguïté sur la sélection de `available_at` par `_first_seen` ; CR5-F5 : précision sur la nature du commentaire SQL de `first_stored_at`). Les quatre sont corrigés ci-dessous.

`ready_for_independent_review=true` — une nouvelle contre-revue indépendante reste requise avant clôture définitive de la gate. Implémentation : `scripts/trading_lab/market_data_store.py`, table `market_data_gap_events`.

### Définition exacte d'un gap (détection strictement intra-payload, existence vérifiée dans le store)

Une ouverture de bougie `expected_bar_open_at` est un gap si, et seulement si :

1. elle est **strictement comprise entre deux `MarketBar` consécutives présentes ensemble dans le payload actuellement ingéré**, espacée d'un multiple de la durée du timeframe ;
2. **et** elle est également absente de `market_bar_receipts` pour exactement le même `(provider, product_id, timeframe)` — une ouverture déjà connue du store, quelle que soit l'ingestion qui l'a produite, n'est **jamais** un gap, même si elle est absente du payload courant (correctif CR3-F1).

Exemple en `1h` : un payload contenant `10:00` et `12:00` (mais pas `11:00`), et `11:00` absente du store → un gap prouvé à `11:00`. Si en revanche `11:00` a déjà été ingérée par un payload antérieur (fenêtres de collecte chevauchantes), aucun `DETECTED` n'est créé : l'ouverture est présente, point final — aucune dépendance à une re-collecte future.

**La détection ne compare jamais deux payloads distincts entre eux pour PROUVER un gap** (l'absence d'une barre dans un payload n'est jamais, à elle seule, une preuve d'absence chez le fournisseur — une page se terminant en 2024 suivie d'une page commençant en 2023 est une simple discontinuité entre deux requêtes indépendantes). Mais la détection **consulte** bien le store, via une vérification d'existence bornée aux candidats du payload courant (voir plus bas) — ce n'est pas une comparaison entre payloads, c'est une vérification ponctuelle par ouverture candidate.

La **résolution** reste transversale aux payloads : toute barre reçue, dans n'importe quel payload ultérieur, peut résoudre un `DETECTED` antérieur si son `bar_open_at` correspond exactement à une ouverture déjà journalisée comme manquante — voir plus bas.

### Bords sans plage causale

Le store ne connaît pas les bornes de la requête fournisseur (quelle plage a réellement été demandée). Il ne détecte donc **jamais** de gap avant la première `MarketBar` du payload ni après la dernière — seuls les intervalles intérieurs, entre deux barres consécutives du même payload, sont prouvables. Un payload à une seule barre (ou une barre sans voisine dans le même payload) ne produit aucun événement.

### Cycle DETECTED / RESOLVED, append-only

Un événement reste produit **par ouverture manquante** (un événement par ouverture, pas d'agrégation de plage), dans la limite de cardinalité décrite ci-dessous. Si une `MarketBar` correspondant à une ouverture précédemment `DETECTED` arrive dans un payload ultérieur (n'importe lequel, pas nécessairement adjacent), un événement `RESOLVED` est ajouté via une recherche ponctuelle par `event_id` déterministe — l'événement `DETECTED` d'origine n'est **jamais** modifié ni supprimé. La table `market_data_gap_events` porte les mêmes triggers `BEFORE UPDATE`/`BEFORE DELETE` que les trois tables existantes.

La cause d'un gap reste explicitement non interprétée (`cause = "unknown"`) : le système ne prétend jamais savoir pourquoi une barre manque.

### Algorithme de détection (correctif CR3-F1)

Pour chaque ingestion :

1. trier les `bar_open_at` du payload courant ;
2. calculer **arithmétiquement** (pas par énumération) le nombre d'ouvertures candidates entre chaque paire consécutive ;
3. vérifier la cardinalité totale de candidats **avant** toute matérialisation, requête d'existence ou insertion (voir limite ci-dessous) ;
4. charger **en batch/chunké** (pas une requête par candidat, pas de `SELECT DISTINCT` global) quels candidats ont déjà un receipt dans `market_bar_receipts` pour exactement le même `(provider, product_id, timeframe)` ;
5. soustraire les candidats déjà connus ;
6. créer un `DETECTED` uniquement pour les candidats réellement absents ;
7. résoudre, sans changement, les `DETECTED` antérieurs dont l'ouverture apparaît dans le payload courant (recherche ponctuelle par `event_id`).

Une barre déjà connue sous un **autre** `provider`, `product_id` ou `timeframe` ne bloque jamais la création d'un `DETECTED` dans le domaine courant : le filtre d'existence porte sur les quatre coordonnées exactes, jamais sur une seule. Plusieurs révisions d'une même ouverture dans le même domaine comptent comme une seule ouverture présente.

Un replay exact d'une ingestion antérieure ne relance pas nécessairement cette réconciliation : l'idempotence de `ingest_coinbase_response` court-circuite avant même d'atteindre la détection lorsque l'`ingestion_id` (octets + métadonnées) est strictement identique à une ingestion déjà enregistrée.

### Limite de cardinalité, sur les candidats examinés, par ingestion et non par domaine (correctif F1, révisé deux fois après contre-revue)

`MAX_GAP_CANDIDATES_PER_INGESTION = 10_000` : nombre maximal d'ouvertures **candidates** (pas seulement le nombre final de `DETECTED`) que les paires intérieures **du payload actuellement ingéré** peuvent produire, tous intervalles de ce payload confondus (le total est vérifié globalement sur le payload, pas séparément par paire). La limite borne le travail à examiner (candidats à matérialiser puis à vérifier en base), pas seulement le résultat final après soustraction des ouvertures déjà connues.

Cette limite porte uniquement sur les candidats de **cette ingestion**, jamais sur le stock déjà persisté du domaine. Une version antérieure du correctif comptait le total cumulé de tout l'historique du domaine à chaque ingestion, ce qui saturait définitivement un domaine dès que son stock de gaps dépassait 10 000 — rendant tout backfill historique descendant (pages non contiguës) impossible passé ce seuil, sans aucun moyen de réparation (tables append-only). Ce n'est plus le cas : chaque ingestion est jugée indépendamment sur son propre payload (≤ 300 bougies), donc un domaine ne peut jamais se retrouver bloqué par l'historique déjà accumulé, quel que soit le nombre d'événements déjà persistés (prouvé avec un stock préexistant de 8 000 événements dans `test_discontinuous_historical_backfill_never_gets_rejected_or_saturates`).

Le total de candidats est calculé **arithmétiquement** (`delta_secondes // durée_timeframe - 1`, avec validation stricte de l'alignement), jamais par énumération. Il est vérifié **avant** toute matérialisation de candidat, avant toute requête d'existence, et avant toute insertion d'événement. Si le total dépasse la limite, l'ingestion entière échoue avec une erreur explicite et **rien n'est persisté** : ni raw payload, ni ingestion, ni receipt, ni événement de gap — aucune troncature silencieuse, aucune agrégation en plage, aucun événement partiel.

Cette limite protège le CPU, la RAM, le disque et la durée de tenue du verrou d'écriture SQLite (`BEGIN IMMEDIATE`) contre un payload légal minuscule (quelques bougies, largement en-deçà de `MAX_CANDLES_PER_RESPONSE`) mais dont les bougies sont espacées d'une durée arbitrairement grande — l'adapter ne contraint pas la proximité temporelle entre bougies d'un même payload.

### Coût structurel borné par le payload — précision sur `_first_seen` (correctifs F1/CR3-F3)

La réconciliation ne relit plus jamais l'ensemble des `bar_open_at` déjà persistés pour le domaine. Le **nombre de requêtes SQL structurelles** (détection, vérification d'existence, résolution) ne dépend que de la taille du payload courant et du nombre de candidats qu'il implique (lui-même plafonné), jamais de la taille de l'historique déjà accumulé pour ce domaine.

Preuve à deux niveaux, pour rester honnête sur ce que chacune couvre exactement :
- **test persistant automatisé** `test_ingestion_cost_is_bounded_by_payload_not_by_domain_history` (`tests/crypto/test_market_data_store.py`) : comptage exact des requêtes SQL via `sqlite3.set_trace_callback`, avec un payload contenant un vrai gap, identique entre un domaine vide et un domaine à **900 barres** d'historique (3 pages de 300 bougies contiguës).
- **probe indépendante non persistante (contre-revue CR4)** : la même propriété (nombre de requêtes structurelles inchangé) a été observée en poussant l'historique à **10 000 barres**, et séparément à **20 000 événements de gap déjà persistés** dans le domaine. Cette mesure a été exécutée par un reviewer indépendant, hors dépôt, et n'est pas rejouée automatiquement par la suite de tests.

Précision honnête : ceci borne le **nombre de requêtes**, pas nécessairement le **volume de lignes parcourues par chacune**. La recherche de première observation (`_first_seen`, utilisée pour le calcul causal des bornes) peut consulter davantage de lignes si l'ouverture consultée porte de nombreuses révisions ou provient de nombreuses ingestions distinctes — son coût dépend du nombre de receipts associés aux ouvertures précises consultées (bornes du payload courant), pas de la taille totale du domaine. Les accès restent indexés (`market_bar_receipts_open_lookup`, confirmé sans balayage complet par `EXPLAIN QUERY PLAN` — voir `test_existence_check_and_first_seen_use_indexed_access`, qui capture le SQL réellement émis par `_load_known_bar_opens`/`_first_seen` via `sqlite3.set_trace_callback` puis vérifie son plan avec une garde robuste aux formats de sortie SQLite, elle-même testée par `test_indexed_search_guard_rejects_full_table_scan_plans`) et bornés aux ouvertures du payload et à ses candidats — jamais un scan complet du domaine.

### Sémantique causale de DETECTED / RESOLVED (correctif F2)

Le timestamp causal (`available_at`/`ingested_at`) d'un événement ne provient **jamais** de l'horloge système ni aveuglément de l'ingestion qui déclenche la réconciliation. Il dépend des deux `MarketBar` qui rendent le gap démontrable :

- `DETECTED` : `max(première observation de la borne basse, première observation de la borne haute)` — le gap n'est démontrable qu'à partir du moment où les **deux** bornes encadrantes sont connues, même si les ingestions arrivent dans un ordre non monotone (`ingested_at` rétrogradant entre deux appels).
- `RESOLVED` : `max(timestamp causal du DETECTED, première observation de la barre tardive)` — une résolution ne peut jamais précéder causalement sa propre détection, même si la barre tardive porte elle-même un `ingested_at` antérieur au `DETECTED` existant.

« Première observation » = timestamps `(available_at, ingested_at)` du receipt portant le plus petit `ingested_at` **déclaré** pour cette ouverture dans ce domaine (voir le contrat explicite ci-dessous — ce n'est ni la première réception physique par le store, ni l'ordre des `INSERT`). `observed_by_ingestion_id` continue de référencer l'ingestion qui a déclenché la détection/résolution (inchangé) ; seuls les champs `available_at`/`ingested_at` dépendent désormais des bornes plutôt que de cette ingestion.

En flux monotone normal (le cas immense majorité), ce calcul redonne exactement les mêmes valeurs qu'avant le correctif — aucune régression fonctionnelle.

### Contrat de `ingested_at` (Contrat A, adopté pour MarketDataStore V1)

Décision explicite du manager après micro-revue read-only du contrat causal : **`ingested_at` représente un timestamp historique déclaré par l'appelant**, pas l'heure réelle de réception par le store.

Ce que `ingested_at` **n'est pas** :
- l'heure réelle de réception par le `MarketDataStore` ;
- l'heure murale locale du processus qui exécute `ingest_coinbase_response` ;
- un ordre d'arrivée local immuable.

**Frontière de confiance** : l'appelant de `ingest_coinbase_response` est considéré comme situé à l'intérieur d'une frontière de confiance — collecteur interne, importeur historique contrôlé, ou fixture de test. Le store ne vérifie que la causalité locale par barre (`ingested_at >= available_at >= bar_close_at`, dans `build_market_bar`). Il ne peut ni ne cherche à garantir que l'appelant n'a pas fourni une valeur rétrodatée, erronée ou mensongère. Des entrées non fiables (un consommateur externe non contrôlé) ne doivent pas pouvoir invoquer cette API directement avec un `ingested_at` de leur choix.

**Sémantique de `first_seen`** (`_first_seen`) : *earliest declared historical ingestion timestamp* — le minimum des `ingested_at` **déclarés** parmi les receipts connus pour l'ouverture. Jamais la première réception physique par le store, jamais l'ordre réel des `INSERT`, jamais l'heure locale à laquelle HyprL a appris la donnée. Une ré-ingestion ultérieure portant un `ingested_at` historique inférieur déplace donc légitimement `first_seen` vers le passé — voulu pour les imports historiques, les backfills et la reconstruction causale à partir de timestamps historiques faisant autorité. Ce mécanisme ne constitue à lui seul aucune preuve de connaissance *live* sans lookahead.

**Précision sur `available_at` (correctif CR5-F4)** : `_first_seen` ne calcule **pas** séparément `MIN(available_at)` et `MIN(ingested_at)`. Il sélectionne le **receipt unique** portant le plus petit `ingested_at` historique déclaré (`ORDER BY ingested_at ASC`, avec `content_sha256` comme départage déterministe), puis renvoie le couple `(available_at, ingested_at)` **tel qu'enregistré sur ce receipt**. Si un autre receipt de la même ouverture portait un `available_at` plus petit mais un `ingested_at` plus grand, ce `available_at` plus petit ne serait **pas** retenu — les deux champs voyagent ensemble comme une seule observation cohérente, jamais minimisés indépendamment l'un de l'autre.

**`raw_market_payloads.first_stored_at`** : malgré son nom, cette colonne est alimentée par le timestamp historique déclaré (`canonical_ingested_at`), pas par une mesure réelle du moment de persistance par le store. Le nom est **historique/legacy** ; il n'est ni renommé ni migré dans cette mission.

**Précision sur le commentaire SQL de `first_stored_at` (correctif CR5-F5)** : ce nom historique/legacy est expliqué par un commentaire `--` placé directement dans le DDL `CREATE TABLE raw_market_payloads` (texte passé à `executescript`). Ce commentaire est stocké tel quel dans `sqlite_master.sql` pour toute base créée avec ce schéma — visible en interrogeant `SELECT sql FROM sqlite_master WHERE name='raw_market_payloads'`. Il reste **sémantiquement inerte** : aucune contrainte, colonne, index, trigger, requête ou logique métier n'en dépend ni n'en est affecté ; ce n'est pas une modification de comportement, seulement un texte informatif capturé dans le schéma stocké.

**Hors périmètre de V1, laissé explicitement pour une gate future** :
- `store_received_at` — une horloge propre au store, capturée au moment réel de l'`INSERT` ;
- un ordre local immuable type `receipt_sequence` ;
- les règles précisant quelle horloge (déclarée vs. locale) gouverne les snapshots et les décisions en régime live.

**Preuve** (`tests/crypto/test_market_data_store.py`) : `test_first_seen_returns_earliest_declared_ingested_at_not_insert_order` — trois receipts de la même ouverture, insérés physiquement dans cet ordre : R1 (`ingested_at`=12:06), R2 (`ingested_at`=10:06), R3 (`ingested_at`=14:06, à la fois dernier inséré et valeur maximale déclarée). `_first_seen` retient 10:06 (R2) — ni le premier `INSERT` (12:06), ni le dernier `INSERT` (14:06), ni le maximum déclaré (14:06). R2 n'étant ni le premier ni le dernier receipt physiquement écrit, ce scénario élimine en un seul test les stratégies « premier inséré », « dernier inséré », « MAX », « rowid ASC » et « rowid DESC ».

### Identité déterministe et idempotence

Chaque événement a un `event_id` déterministe (hash de `provider` + `product_id` + `timeframe` + `expected_bar_open_at` + `event_type`), indépendant du timestamp d'ingestion. Un replay exact d'un payload — qu'il s'agisse de la détection initiale ou de la résolution ultérieure — ne crée donc aucun doublon, et les timestamps causaux recalculés restent strictement identiques après replay.

### Garanties

- Isolation stricte par `(provider, product_id, timeframe)` : un gap BTC-USD/1h ne contamine ni ETH-USD ni BTC-USD/1d. L'isolation est portée par l'identité même (`event_id` dérivé du triplet), pas seulement par une clause `WHERE`.
- Transactionnel et fail-closed : la réconciliation des gaps (préflight de cardinalité inclus) s'exécute dans la même transaction que l'insertion des `MarketBar` ; tout échec (validation, dépassement de la limite, ou SQL) annule l'ensemble, y compris tout événement de gap qui aurait pu être écrit.
- Une révision de barre existante (même `bar_open_at`, contenu corrigé) ne crée ni faux gap ni fausse résolution, car la détection ne regarde que l'ensemble des `bar_open_at` du payload, pas leur contenu.
- Aucune `MarketBar` synthétique, aucun forward-fill, aucun backfill : uniquement de l'observation journalisée.
- Aucun gap de bord n'est inventé : seuls les intervalles intérieurs entre deux barres consécutives d'un même payload sont prouvables ; une barre isolée ne produit aucun événement.
- Un backfill historique discontinu (pages non contiguës, ingérées dans n'importe quel ordre) ne peut jamais être rejeté par accumulation de stock, et ne crée jamais de faux gap entre deux pages distinctes.
- Aucun forward-fill, aucun gap inter-payload : l'absence d'une barre dans un payload n'est jamais interprétée comme une preuve d'absence tant qu'elle n'est pas encadrée par deux barres du même payload ET absente du store.
- Une ouverture déjà connue du store — quel que soit le payload ou l'ingestion qui l'a produite — ne peut jamais devenir un `DETECTED`, même si elle est absente du payload courant (correctif CR3-F1).

### Tests de preuve

Tous dans `tests/crypto/test_market_data_store.py` (52 cas collectés — 51 fonctions de test, dont `test_invalid_response_leaves_store_empty_atomically` paramétrée ×2 — tous verts). Ce compteur est le résultat de `python -m pytest --collect-only -q tests/crypto/test_market_data_store.py`, un **test persistant automatisé** ; il est distinct des mesures ponctuelles effectuées par les probes indépendantes des contre-revues (voir plus bas, ex. la propriété observée à 10 000 barres/20 000 événements), qui ne sont pas rejouées automatiquement par cette suite :

Détection/résolution de base :
- `test_contiguous_series_produces_no_gap_events`
- `test_single_interior_missing_bar_creates_one_detected_gap_event`
- `test_multiple_consecutive_missing_bars_create_one_event_each`
- `test_gap_detection_replay_is_idempotent`
- `test_late_arrival_resolves_gap_with_single_resolved_event`
- `test_replaying_the_resolution_does_not_duplicate`
- `test_gap_events_are_isolated_by_product_and_timeframe`
- `test_gap_event_insertion_failure_rolls_back_atomically`
- `test_market_data_gap_events_table_is_insert_only`
- `test_isolated_bar_creates_no_boundary_gap_events`
- `test_bar_revision_does_not_create_or_resolve_gap_events`

Limite de cardinalité, par ingestion (F1) :
- `test_count_missing_opens_is_arithmetic_not_enumerative`
- `test_pathological_century_gap_is_rejected_atomically`
- `test_gap_limit_is_checked_before_materializing_events`
- `test_exactly_the_gap_limit_is_accepted`
- `test_one_more_than_the_gap_limit_is_rejected_atomically`
- `test_multi_segment_total_over_limit_is_rejected_before_partial_insert`
- `test_valid_ingestion_after_pathological_rejection_succeeds`

Sémantique causale (F2) :
- `test_first_seen_returns_earliest_declared_ingested_at_not_insert_order`
- `test_non_monotone_ingestion_order_detected_causal_at_uses_latest_bound`
- `test_retrodated_resupply_of_a_bound_shifts_its_first_seen_into_the_causal_max`
- `test_retrodated_late_arrival_resolved_never_precedes_detected`
- `test_monotone_flow_causal_timestamps_match_natural_expectation`
- `test_causal_timestamps_are_identical_after_replay`

Détection strictement intra-payload et coût borné (correctif après 2ᵉ contre-revue) :
- `test_gap_between_separate_payloads_is_never_detected`
- `test_discontinuous_historical_backfill_never_gets_rejected_or_saturates`
- `test_ingestion_cost_is_bounded_by_payload_not_by_domain_history`

Aucun faux gap sur ouverture déjà connue, isolation, limite sur les candidats (correctif après 3ᵉ contre-revue, CR3-F1) :
- `test_overlapping_payload_does_not_create_false_gap_for_known_bar`
- `test_bar_known_under_other_provider_does_not_suppress_detection`
- `test_bar_known_under_other_product_id_does_not_suppress_detection`
- `test_bar_known_under_other_timeframe_does_not_suppress_detection`
- `test_multiple_revisions_of_a_known_bar_still_count_as_present`
- `test_mixed_known_and_missing_candidates_creates_detected_only_for_missing`
- `test_replaying_overlapping_payload_creates_no_new_events`
- `test_gap_candidate_limit_is_checked_before_existence_lookup`
- `test_existence_lookup_failure_rolls_back_atomically`

Preuve d'accès indexé et de couverture des tests (correctif après 4ᵉ contre-revue, CR4-F2/F3) :
- `test_indexed_search_guard_rejects_full_table_scan_plans`
- `test_existence_check_and_first_seen_use_indexed_access`

## Snapshots causaux historiques — sélection (Phase 1C-A / 1C-B)

`scripts/trading_lab/market_snapshots.py` sélectionne, sans effet de bord et sans rien persister, quel receipt MarketBar déjà stocké représente l'état de la connaissance **déclarée** pour chaque `bar_open_at` d'une plage demandée.

Terminologie exacte : ce sont des **snapshots causaux historiques fondés sur le temps d'ingestion déclaré** (Contrat A). `as_of` est un seuil sur le `ingested_at` **déclaré** des receipts déjà persistés — jamais une horloge locale réelle, jamais une preuve qu'une connaissance était réellement disponible en temps réel à cet instant.

### Sélection (Phase 1C-A)

- éligibilité : `ingested_at <= as_of` ; isolation stricte `provider` / `product_id` / `timeframe` ; `range_start` inclusif, `range_end` exclusif ;
- pour chaque `bar_open_at`, le ou les receipts au `MAX(ingested_at)` éligible sont les candidats ;
- si ces candidats portent **plusieurs `bar_version_id` distincts**, c'est une contradiction réelle de l'historique déclaré : `SnapshotSelectionConflict`, fail-closed, aucun résultat partiel. `content_sha256` ne départage **jamais** deux contenus OHLCV contradictoires — uniquement des receipts partageant le **même** `bar_version_id` ;
- aucune dépendance au `rowid`, à l'ordre d'insertion, ni à l'ordre des lignes SQL ; résultat trié par `bar_open_at` croissant.

### Alignement des plages (Phase 1C-B)

Les deux bornes doivent tomber **exactement** sur la grille UTC du timeframe, ancrée sur `1970-01-01T00:00:00+00:00` :

- `1h` : `minute == second == microsecond == 0` ;
- `1d` : en plus `hour == 0` (minuit UTC).

Le contrôle est structurellement un **modulo en microsecondes entières** (`µs_depuis_époque % durée == 0`), appliqué **borne par borne**. Deux pièges que cette forme évite :

- la **divisibilité de la plage ne prouve pas l'alignement** : `[10:30Z, 12:30Z)` dure exactement deux heures mais aucune `bar_open_at` stockée ne peut y tomber ;
- `timedelta.total_seconds()` est un flottant et **tronque silencieusement** une microseconde résiduelle : toute l'arithmétique est donc entière.

La durée du timeframe est dérivée de `TIMEFRAME_DURATIONS` importé de `market_bar.py`, jamais redéclarée : une grille de snapshot en désaccord avec MarketBar V1 pourrait désigner des positions qu'aucune barre ne peut occuper. L'équivalence est prouvée dans les deux sens par `test_alignment_rules_match_market_bar_v1_exactly`.

Toute la validation de plage se produit **avant** le moindre accès SQLite, et `build_snapshot_request_id` partage exactement le même préflight que le sélecteur : une identité n'est jamais frappée pour une plage que le sélecteur refuserait de calculer.

### Limites de cardinalité (Phase 1C-B)

- `MAX_SNAPSHOT_RANGE_OPENS = 10 000` — nombre d'ouvertures **attendues** dans la grille, calculé par division entière `(end_us - start_us) // duration_us`. O(1), sans matérialiser aucun timestamp, sans requête. Couvre ~1,14 an en `1h` et ~27 ans en `1d`. Une plage de 87 millions d'ouvertures est rejetée en mémoire constante.
- `MAX_SNAPSHOT_ELIGIBLE_RECEIPTS = 100 000` — la plage borne les **ouvertures**, jamais les **révisions** derrière elles : sans cette seconde limite, une seule ouverture pathologique pourrait renvoyer un nombre non borné de lignes.

`LIMIT 100001` dans le SQL est un **détecteur de dépassement, jamais une troncature fonctionnelle**. Comportement : 0 à 100 000 receipts éligibles → traitement intégral ; 100 001 → `SnapshotEligibilityLimitExceeded`, rejet complet, **aucun résultat dérivé des 100 000 premières lignes**.

**Précédence d'erreurs V1, explicite** : *le dépassement l'emporte sur un conflit situé au-delà de la limite d'éligibilité.* Les deux branches sont fail-closed et aucune ne renvoie de résultat partiel ; la précédence ne décide donc que du refus typé que voit l'appelant, jamais du fait que la requête est refusée. En deçà de la limite, **toutes** les lignes sont traitées et tout conflit est détecté.

Contrainte croisée à connaître : à la plage maximale (10 000 ouvertures), 100 000 receipts n'autorisent qu'une moyenne de **10 révisions par ouverture**. Une demande légitime plus révisée doit être **découpée en plusieurs plages** ; le découpage est déterministe et le message d'erreur le rappelle.

### Stratégie de lecture (Phase 1C-B)

Lecture par `fetchmany(SNAPSHOT_RECEIPT_FETCH_CHUNK_SIZE = 1 000)` avec compteur exact, jamais `fetchall`. Chaque ligne est repliée dans un agrégat par ouverture qui ne retient que le `MAX(ingested_at)` courant, les `bar_version_id` distincts à ce maximum, et le receipt canonique au `content_sha256` maximal.

Conséquences vérifiées :

- la mémoire suit le **nombre d'ouvertures** (≤ 10 000), pas le nombre de révisions ;
- les frontières de chunk ne portent **aucune sémantique** : une même ouverture, ou un même `MAX(ingested_at)`, peut être répartie sur plusieurs chunks sans changer le résultat ;
- un conflit ancien **supplanté** par une révision plus récente non ambiguë n'est jamais levé — les conflits ne sont décidés qu'après avoir replié toutes les lignes.

### Requête et index (Phase 1C-B)

La requête filtre **par domaine d'abord** (sous-requête `IN` sur `market_ingestions`) puis par plage, et ne porte **aucun `ORDER BY`** : l'ordre final est produit en Python. Supprimer le tri SQL élimine le `TEMP B-TREE` et la latence de première ligne qu'il imposait.

Un unique index est ajouté au DDL de `MarketDataStore` :

```sql
CREATE INDEX IF NOT EXISTS market_bar_receipts_snapshot_domain_lookup
    ON market_bar_receipts (ingestion_id, bar_open_at, ingested_at,
                            content_sha256, bar_version_id, bar_id, available_at);
```

- **Couvrant** : il porte les six colonnes projetées plus la colonne de jointure, donc la sélection ne touche jamais la table.
- **Tête `ingestion_id`, délibérément** : c'est ce qui rend le coût indépendant du nombre d'**autres** domaines partageant les mêmes `bar_open_at`. Un index à tête `bar_open_at` ferait croître le coût avec chaque domaine colocalisé **et** détournerait les deux lookups Phase 1B vers un autre index, cassant `test_existence_check_and_first_seen_use_indexed_access` — vérifié empiriquement.
- **Aucun index n'est ajouté sur `market_ingestions`** : celui qui supprimerait le `SCAN i` résiduel casse justement l'assertion Phase 1B ci-dessus.

Plan réellement obtenu : `SEARCH r USING COVERING INDEX market_bar_receipts_snapshot_domain_lookup (ingestion_id=? AND bar_open_at>? AND bar_open_at<?)` · `LIST SUBQUERY 1` · `SCAN i`.

Migration : `CREATE INDEX IF NOT EXISTS` s'exécute à chaque `MarketDataStore.__init__`, donc une base Phase 1B antérieure gagne l'index à la simple réouverture — vérifié par test.

### Risques résiduels acceptés

- **`SCAN i` sur `market_ingestions`** : coût en O(nombre total d'ingestions), pas en O(receipts hors plage). Accepté ; l'index qui l'éliminerait casserait une assertion Phase 1B.
- **Petite plage sur énorme historique hors plage** : légère régression en absolu, négligeable.
- **Coût de stockage de l'index** : l'index ajoute **plusieurs centaines d'octets par receipt** dans les mesures réalisées — de l'ordre de **293 à 520 octets selon le dataset**. Le **pourcentage** de croissance de la base n'est pas une constante : il dépend fortement de la taille des `payload_json`, du remplissage des pages SQLite et de la distribution des données, et n'a donc pas de valeur universelle. Compromis accepté au vu de la forte réduction du coût multi-domaines.
- **Débit d'ingestion** : réduction de l'ordre de quelques pourcents.
- **Duplication de `_canonical_timestamp`** entre `market_data_store.py` et `market_snapshots.py` (dette, suivie).
- **`SelectedSnapshotReceipt.available_at` brut** (non canonicalisé) : sans effet sur la sélection ni sur les identités, atteignable uniquement par corruption SQL directe.

### Hors périmètre de 1C-B

Aucune table `market_snapshot_*`, aucune matérialisation, aucune API `create_snapshot` / `load_snapshot` / `list_snapshot_bars`. Une entrée absente d'un snapshot n'est **jamais** un gap fournisseur confirmé : c'est l'absence d'un receipt éligible à cet `as_of`. Les gaps confirmés ont leur propre mécanisme persisté (`market_data_gap_events`), décrit plus haut.

## Persistance des snapshots causaux (Phase 1C-C)

Deux tables append-only stockent un snapshot **immuable et reproductible**. Elles ne contiennent **aucune donnée de marché** : ni OHLCV, ni `payload_json`, ni `bar_id`, ni `bar_version_id`, ni `ingested_at`, ni `available_at`. `content_sha256` — clé primaire de `market_bar_receipts` — est **l'unique référence persistée** vers le receipt Phase 1B ; tout le reste s'obtient par jointure.

### `market_snapshot_manifests`

`snapshot_id` (PK) · `snapshot_request_id` · `entries_content_hash` · `snapshot_schema_version` · `selection_policy_version` · `provider` · `product_id` · `timeframe` · `range_start` · `range_end` · `as_of` · `entry_count` (`CHECK 0 ≤ n ≤ 10 000`, plus `CHECK range_end > range_start`).

**Aucun `created_at`, aucun `materialized_at`, aucune séquence locale.** Aucune logique n'en dépend, et en introduire un recréerait l'ambiguïté de `first_stored_at` décrite plus haut. Un seul index : `market_snapshot_manifests_request_lookup (snapshot_request_id, snapshot_id)`.

### `market_snapshot_entries`

`snapshot_id` · `bar_open_at` · `content_sha256`, `PRIMARY KEY (snapshot_id, bar_open_at)`, `WITHOUT ROWID`. La PK rend deux entries pour la même ouverture structurellement impossibles ; `WITHOUT ROWID` supprime le rowid, donc toute dépendance accidentelle à celui-ci. Deux clés étrangères : vers le manifest (**`DEFERRABLE INITIALLY DEFERRED`**) et vers `market_bar_receipts(content_sha256)`.

### Identités

`snapshot_request_id` = hash des paramètres canoniques (la **demande**) · `entries_content_hash` = hash versionné des couples ordonnés `[bar_open_at, content_sha256]` (le **contenu**) · `snapshot_id` = hash des deux (la **combinaison**). Aucun champ local non déterministe n'y participe. Le `selection_policy_version` persisté est exactement celui qui a servi à construire `snapshot_request_id`.

### Scellement : entries d'abord, manifest en dernier

L'ordre d'écriture est imposé par le schéma. La FK vers le manifest est **différée**, ce qui autorise les entries avant leur parent dans la même transaction ; un trigger `market_snapshot_entries_no_late_insert` refuse toute entry dont le manifest **existe déjà**. L'insertion du manifest **scelle** donc le snapshot définitivement : il ne peut plus jamais être enrichi. Une transaction qui n'insérerait jamais le manifest voit ses entries orphelines refusées **au `COMMIT`** par la FK différée.

S'ajoutent quatre triggers `<table>_no_update` / `<table>_no_delete` selon la convention Phase 1B.

### PRAGMAs obligatoires

`MarketDataStore._connect()` active **et vérifie** `foreign_keys` et `recursive_triggers`, puis échoue si l'un des deux ne vaut pas 1.

`recursive_triggers` n'est pas un détail : sans lui (valeur SQLite par défaut), **`INSERT OR REPLACE` supprime la ligne en conflit sans déclencher le trigger `BEFORE DELETE`**, réécrivant silencieusement une ligne que le schéma déclare immuable. Ce vecteur touchait **toutes** les tables append-only Phase 1B et n'était couvert par aucun test ; il l'est désormais pour chacune d'elles.

La primitive d'écriture vérifie les deux PRAGMAs **avant `BEGIN` et avant toute lecture** : ils sont par connexion, et SQLite les transforme en no-op silencieux à l'intérieur d'une transaction. Une connexion nue reste utilisable pour les primitives read-only, jamais pour l'écriture.

### Transaction

`_materialize_snapshot` est l'**unique propriétaire** de sa transaction. `BEGIN IMMEDIATE` est pris **avant la sélection causale** : sans cela, un backfill concurrent survenant entre la sélection et l'écriture produirait un manifest attestant un état qui n'a jamais existé. Une connexion déjà en transaction est **refusée** (`SnapshotWriteContextError`) — ni savepoint, ni participation à la transaction appelante en V1.

Toute exception provoque un `ROLLBACK` complet : un manifest sans ses entries, ou des entries sans leur manifest, sont des états que cette primitive ne laisse jamais derrière elle. `SnapshotSelectionConflict` et `SnapshotEligibilityLimitExceeded` remontent **inchangées** après le rollback ; les erreurs SQLite sont enveloppées dans `SnapshotPersistenceError` avec chaînage, sans jamais exposer de payload ni la liste des entries.

### Idempotence

Si le `snapshot_id` calculé existe déjà, **aucun `INSERT OR IGNORE`, aucune présomption de succès**. L'état persisté est vérifié intégralement : les onze champs du manifest, `entry_count`, l'ensemble ordonné exact des couples `(bar_open_at, content_sha256)` rechargés par la PK, et le `entries_content_hash` **recalculé depuis les entries réellement stockées**. Tout correspond → no-op, `created=False`, même `snapshot_id`, aucun doublon. Toute divergence → `SnapshotStateCorruption`, rollback, **aucune réparation silencieuse**.

### Backfill

Même demande, contenu différent → `entries_content_hash` différent → `snapshot_id` différent → **nouveau** manifest immuable. L'ancien reste strictement intact et chargeable. Deux manifests coexistent alors sous le même `snapshot_request_id`.

⚠️ **Aucun ordre chronologique n'est récupérable entre plusieurs versions d'une même demande** : il n'existe aucune horloge locale, et `as_of` est identique par définition. Le listage est déterministe (par `snapshot_id`), pas chronologique. Si un tel ordre devient nécessaire, ce sera une décision explicite, pas un effet de bord.

### Snapshot vide

Une plage valide sans receipt éligible produit un manifest avec `entry_count = 0` et le hash de l'enveloppe vide — un état précis et vérifiable, distinct d'une corruption. Il signifie **uniquement** « aucun receipt éligible à cet `as_of` », et **jamais** « gap fournisseur confirmé » : les gaps confirmés ont leur propre mécanisme persisté (`market_data_gap_events`).

### Vérification structurelle du schéma

`CREATE TABLE/INDEX/TRIGGER IF NOT EXISTS` **ne remplace pas** un objet incorrect portant déjà le même nom. Une vérification fail-closed s'exécute donc après la création : définition stockée comparée (normalisée) à celle que ce module aurait écrite, plus contrôle indépendant par `PRAGMA table_info` (colonnes, types, `NOT NULL`, PK), `PRAGMA foreign_key_list`, `PRAGMA index_list`/`index_info`, et sonde comportementale du `WITHOUT ROWID`. Une table mal formée, un index sur la mauvaise colonne ou un trigger inopérant portant le bon nom font **échouer explicitement l'initialisation** — jamais de réparation silencieuse. Une base Phase 1B antérieure, elle, est migrée normalement à la réouverture.

**F2 reste reporté** dans sa forme générale : `executescript` peut committer des tables avant l'échec ultérieur d'un trigger. 1C-C n'y touche pas, mais ajoute cette assertion locale, qui rend la conséquence détectable pour ses propres objets — une table de snapshot debout sans ses triggers d'immutabilité serait bien pire qu'en Phase 1B.

## Lecture, listing et replay des snapshots (Phase 1C-D)

Trois fonctions publiques, aucune dépendance runtime, aucun DDL, aucun index nouveau — le schéma 1C-C suffit.

```python
load_snapshot(connection, *, snapshot_id) -> LoadedSnapshot
list_snapshot_manifests(connection, *, snapshot_request_id,
                        after_snapshot_id=None, limit=100) -> SnapshotManifestPage
replay_snapshot(connection, *, snapshot_id) -> tuple[dict, ...]
```

### Dataclasses publiques

`SnapshotManifest` (les 12 champs du manifest) · `SnapshotEntryRef` · `LoadedSnapshot` · `SnapshotManifestPage`. Toutes gelées.

### Format exact des identifiants

Les identifiants publics sont **préfixés** — ce ne sont pas de simples hashs de 64 caractères :

| Identifiant | Forme | Longueur |
|---|---|---|
| `snapshot_id` | `hyprl-market-snapshot-` + 64 hex minuscules | **86** |
| `snapshot_request_id` | `hyprl-market-snapshot-request-` + 64 hex minuscules | **94** |
| `entries_content_hash` | SHA-256 **nu**, 64 hex minuscules, sans préfixe | **64** |

`snapshot_id` et `snapshot_request_id` sont validés par correspondance **exacte et ancrée** sur ces motifs. Sont rejetés par `SnapshotInputError`, **sans aucune normalisation** : préfixe absent ou incorrect, suffixe de 63 ou 65 caractères, majuscules, caractère non hexadécimal, espace ou saut de ligne en tête ou en fin, `bytes`, et toute valeur dont le type n'est pas exactement `str` (une sous-classe de `str` est refusée). `entries_content_hash` n'est pas un identifiant d'entrée publique : il est recalculé, jamais accepté d'un appelant.

`SnapshotEntryRef` est **délibérément minimal** : `bar_open_at` et `content_sha256`, rien d'autre. Le `bar_id`, le `bar_version_id`, `ingested_at`, `available_at`, le domaine du receipt et le `payload_json` sont tous **vérifiés** pendant le chargement mais **jamais publiés** — les exposer dupliquerait le domaine sur jusqu'à 10 000 objets et souderait cette API à la forme des lignes Phase 1B.

### Absent, vide, corrompu : trois réponses distinctes

`SnapshotNotFound` quand aucun manifest ne porte cet identifiant · un `LoadedSnapshot` avec `entries == ()` pour un snapshot **légitimement vide** · `SnapshotStateCorruption` quand le manifest existe mais que son état est incomplet ou incohérent. Un snapshot vide signifie **uniquement** « aucun receipt éligible à cet `as_of` », jamais un gap fournisseur confirmé. **Aucune réparation, aucune écriture, aucun réseau, aucun broker.**

### Validation content-addressed

Le chargement recalcule les **trois identités** avec les builders existants (`build_entries_content_hash`, `build_snapshot_request_id`, `build_snapshot_id`) et les compare au manifest **et** à l'identifiant demandé. Le replay va plus loin : chaque payload est rehashé, puis **reconstruit par `build_market_bar` lui-même** et comparé au stocké. Un payload altéré **puis rehashé** satisferait un contrôle de hash seul ; il ne survit pas à la reconstruction par MarketBar V1.

### LEFT JOIN obligatoire

Les jointures entries → receipts → ingestions sont **externes**. Mesuré : sur un snapshot de 3 entries dont le receipt d'une barre a été supprimé, une `INNER JOIN` renvoie 2 lignes — la référence brisée **disparaît silencieusement** et `entry_count` reste correct. La `LEFT JOIN` la rend visible comme `NULL`, et le chargement lève `SnapshotStateCorruption`.

### Transaction : capture seulement, validation après COMMIT

En `journal_mode=delete`, une transaction de lecture **empêche tout writer de committer**. La lecture est donc scindée : **phase A** capture sous une seule vue SQLite (manifest, métadonnées, éventuellement payloads) puis committe immédiatement ; **phase B** décode, rehashe et valide sur les copies, **sans transaction ouverte**. C'est sûr parce que manifests, entries, receipts et ingestions sont insert-only et scellés.

Observé à 10 000 barres, **sans instrumentation mémoire** : capture ≈ **50 ms**, replay total ≈ **761 ms**, soit une transaction tenue sur ≈ **6,6 %** du replay. `BEGIN` simple, jamais `BEGIN IMMEDIATE` — une lecture ne prend pas le verrou d'écriture. Une connexion déjà en transaction est refusée (`SnapshotReadContextError`).

⚠️ **Ce sont des observations de benchmark sur une machine donnée, pas un SLA.** Sous `tracemalloc` — nécessaire pour mesurer le pic mémoire mais qui perturbe fortement le temps — les mêmes appels donnent ≈ 130 ms de capture pour ≈ 4 432 ms de total, soit une **inflation d'environ ×5,8 du temps total**. Les deux jeux de chiffres ne sont pas comparables entre eux, et une mesure de latence ne doit jamais être prise sous `tracemalloc`.

⚠️ **Ceci ne résout pas complètement la limitation** : la capture bloque encore brièvement un writer. La gate `journal_mode`/WAL reste **séparée et non traitée ici**.

### Aucun résultat partiel

`replay_snapshot` ne retourne qu'après validation de la **dernière** barre. Aucun générateur, aucun `yield`, aucun callback ne voit une barre non vérifiée. **Aucune API de streaming en V1** — un futur itérateur devra être une fonction distincte, pas une modification de celle-ci.

### Budget de payload

`MAX_SNAPSHOT_REPLAY_PAYLOAD_BYTES = 32 000 000`, compté en **octets UTF-8**, jamais en `len(str)`. Préflighté via `length(CAST(payload_json AS BLOB))` : SQLite compte les octets **sans transférer les payloads**, et un dépassement lève `SnapshotReplayLimitExceeded` avant que la requête de payload ne soit exécutée.

⚠️ **Ce budget n'est pas une limite de RAM** : les objets Python décodés coûtent plusieurs fois la taille de leur JSON. C'est une **défense en profondeur contre une base sabotée**, pas une protection contre des données légitimes : un `payload_json` légitime est structurellement borné à **1 417 octets** (jeu de champs MarketBar figé, décimales bornées), soit ~14 Mo au pire cas légal absolu de 10 000 entries.

### Pagination

Contrat public exact : **« Une page est cohérente et ordonnée lexicalement. Une pagination effectuée par plusieurs appels ne garantit pas l'exhaustivité en présence de nouvelles matérialisations concurrentes, car `snapshot_id` est un hash non monotone. »**

Keyset strict (`snapshot_id > curseur`), `LIMIT limit + 1` pour détecter la page suivante, curseur émis uniquement si une ligne surnuméraire existe. **Aucun `OFFSET`, aucun `rowid`, aucune notion de `latest` ou `newest`, aucun tri chronologique** — aucune horloge locale n'existe et toutes les matérialisations d'une même demande partagent le même `as_of`.

## Gate commercial

Avant affichage dans un cockpit accessible à des tiers :

- obtenir et archiver l’autorisation applicable à l’usage prévu ;
- confirmer les droits de stockage historique, données dérivées, affichage et redistribution ;
- documenter attribution, rétention, territoire, utilisateurs autorisés et coût ;
- prévoir un fournisseur de remplacement si ces droits sont refusés ou incompatibles.

Tant que ce gate n’est pas fermé, les données Coinbase restent **internes, research/shadow et non redistribuées**.
