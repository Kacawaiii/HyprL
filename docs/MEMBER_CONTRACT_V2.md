# Contrat membre v2 : scores observés du duel

Le v2 représente la surperformance face au SPY aux horizons du protocole, avec
les engagements des **records entiers**. Il ne transforme aucune décision en
engagement unitaire rétrospectif. Les schémas/exporteurs v1 et les déclarations
de recherche restent inchangés. Le site v2 lit exclusivement `member-v2.json`
et expose `/api/member/v2/` ; aucun repli ni migration implicite vers le v1.

Le déploiement du site appartient à l’opérateur. Le producteur peut maintenir
les deux fichiers de snapshot pendant cette transition. Aucun appel réseau,
acquisition, modèle, entraînement, nouveau label ou ordre n’est exécuté.

## Schémas et identités

Les deux schémas fermés sont dans `apps/hyprl-site/docs/member/`. Leurs SHA-256
canoniques (JSON UTF-8, clés triées, séparateurs compacts) sont liés dans
`scripts/member_export/spec.py` ; le lecteur Node lie aussi le snapshot.

| Schéma | SHA-256 canonique |
| --- | --- |
| handoff-v2 | `2d31c6f0064fa3e938dad74421e0b5a6e53ed664e3e701896062a041efc6f04b` |
| snapshot-v2 | `12d94734cfdef908afb7d3b4d6282a2d8c607429d7e20b56f85fef0f32340435` |

Une modification de ces contrats exige une nouvelle révision et ses bindings.
Un snapshot v1 renommé v2 reste refusé. La provenance synthétique n’est permise
que dans les lecteurs de test, jamais dans le producteur/lecteur de production.
La limite reste 2 Mio ; les dépassements conservent le dernier fichier validé
et mettent le job en échec. Ne pas tronquer les scores ou inventer une continuité
pour contourner cette limite : une future extension exige sa propre révision.

## Prévisions, preuves et scores

Les noms publics sont HypR-A1 et HypR-E1. Une série est définie par
`agent / mode / variant / horizon` :

- modes `REGULAR_0905` et `INTRADAY`, jamais mélangés ;
- `a_v1_as_live` : 1d / 5d ; `c_news_small_mid` : 2d / 10d ;
- événement `outperform_spy`, observation `1` si le rendement relatif amont est
  strictement positif, `0` sinon. Ce résultat précède les coûts de trade ;
- les durées/entrées/maturités restent celles du protocole identifié par
  `source_spec_sha256`. Le suffixe d ne redéfinit pas un horizon de marché en
  nombre universel de jours civils, notamment pour les actifs crypto ;
- probabilité copiée exactement du record public ; hashes originaux de
  décision, snapshot, code, spécification et score immuable conservés.

Chaque prévision pointe par `reveal_id` vers une preuve par agent/séance.
La preuve contient le commit public de révélation, le fichier, son SHA-256, les
dates de publication/disponibilité et le lien au commit d’engagement antérieur.
Le producteur vérifie localement l’ascendance publiée des deux commits, les
octets Git, les reçus confirmés et l’égalité avec le hash original engagé.
Il vérifie l’heure avant ouverture du premier engagement et utilise un garde
conservateur de révélation à 16 h 15 New York, même lors des séances écourtées.
Il ne lance ni fetch ni publication. Une révélation peut être postérieure au
scoring : les deux preuves doivent être disponibles au moment de l’export.

Les métadonnées de prévision sont comparées au record entier révélé, sans en
publier le corps dans le snapshot. La page Preuves donne le fichier brut et la
commande SHA-256 ; elle ne prétend pas recalculer ce fichier dans le navigateur.
Pour NO_DECISION, le hash du README est explicitement distinct du hash d’échec
engagé. Aucune décision ni observation de rendement n’est créée pour un échec.

La référence est le taux de développement déjà figé : 590 succès / 1214
observations, lissage `(590 + 1) / (1214 + 2) = 0,48601973684210525`.
Son identité est
`0693f1dc690d690ea6e3bd1d797e0ba6f8506640173126c1565ae7d2466bb4f4`.
La page lie le gel public. Ce même taux primaire 1d sert de **référence commune**
aux autres horizons ; aucun taux spécifique n’est estimé avec les résultats
forward. Les autres horizons sont descriptifs et ne deviennent pas des critères
principaux ni des preuves de compétence. Le critère principal de recherche et
son verdict restent inchangés.

Pour chaque série :

- `n` et `hits` portent sur les probabilités effectivement notées et révélées.
  Une réussite signifie `(p >= 0,5) == outcome` ;
- Brier = moyenne de `(p − outcome)²` sur **toutes** ces probabilités, y compris
  les abstentions de trade ; Brier de référence sur les mêmes observations ;
- skill = `1 − somme des erreurs / somme des erreurs de référence`, nullement
  une différence brute de scores. Il est indéfini si la référence manque ou son
  erreur est nulle ; les valeurs négatives sont affichées ;
- taux glissant sur les 20 dernières observations, fenêtre partielle au début ;
- ordre déterministe : séance, heure de décision, symbole. L’axe est le numéro
  d’observation et le tableau conserve séance/symbole/date de notation ;
- exactement un point par ligne admissible : aucun point initial, interpolation
  de dates, label immature, duplication, remplissage ou série simulée ;
- `trade_n` / `trade_hits` recopient les trades analyste amont lorsqu’ils existent.
  Ils restent distincts des `hits` de prévision. Les trades news ont une durée
  distincte et ne sont pas joints artificiellement aux probabilités 2d / 10d.

Les 16 combinaisons déclarées restent présentes, y compris celles avec `n=0`,
scores indéfinis et `points=[]`. Les échecs MODEL_FAILURE sont des séances sans
décision, conservées avec leur code réel et leur preuve NO_DECISION. Les comptes
de séances sont rappelés sur chaque horizon du mode concerné : ils ne doivent
pas être additionnés entre horizons. L’export et le lecteur refusent les
doublons. Le lecteur Node recalcule l’intégralité des séries et refuse toute
différence de compte, point ou métrique.

Le Compteur de chance reste « En cours de constitution ». Le v2 ne publie pas
une probabilité fondée sur l’indépendance de prévisions corrélées. Le registre
« Chance ou pas ? » demeure vide en l’absence de verdict éditorial attesté.

## Calendrier et revenus

Un repère existant `at=00:00:00Z` ou `23:59:59Z` devient une **date seule**,
`precision=date`, `scheduled_at=null`. Les empreintes de la ligne et du jeu
observé, ainsi que sa date de réception originale, sont conservées.
`source=null` dit explicitement qu’aucune attestation de publication officielle
par événement n’est disponible. L’interface signale un repère observé dont la
date n’est pas confirmée par l’émetteur. Elle ne crée ni URL ni heure.
Une future heure confirmée exige `precision=time`, un horaire et une source
causale. Les événements de société suivent toujours la liste privée ; les
repères macro restent visibles au niveau Essentiel.

Le cache revenus réduit ne conserve ni accession ni intervalles originaux : le
producteur laisse donc `revenue=[]` et `screener=[]`. L’écran affiche « En cours
de constitution ». Aucune reconstruction n’a été faite. Une future ingestion
SEC XBRL devra conserver CIK, concept, accession de chaque dépôt, source/receipt,
devise et intervalles start/end réels de trois trimestres contigus comparables.
Le schéma v2 et les gardes imposent ces éléments ; aucune requête SEC n’est
autorisée par cette livraison.

## Exécution et tests

La configuration privée reprend les clés du producteur v1, avec un **autre**
handoff et fichier status. Le répertoire snapshot peut rester commun : les
noms de fichiers sont différents. Aucun chemin privé n’est livré dans Git.

```sh
cd "$MEMBER_CHECKOUT"
"$MEMBER_PYTHON" -m scripts.member_export.producer_v2 \
  --config "$MEMBER_V2_CONFIG" --expected-commit "$REVIEWED_REVISION"
```

Le checkout doit être propre et épinglé. Un verrou distinct sérialise le v2 ;
schéma, provenance, horloges, preuves et tailles sont validés avant publication
par remplacement atomique + fsync. Le snapshot est 0440, les sorties privées
0600. Une erreur ne remplace pas le dernier snapshot validé. Les limites des
feeds sont comptées dans le status privé, jamais masquées en succès complet.

Les tests utilisent des formes synthétiques du score LIVE_FORWARD, du record
entier, des reçus, d’un vrai dépôt Git local et du calendrier de journée. Ils
exercent les frontières Python → Node, les cinq niveaux Discord, les refus v1,
les métriques altérées, les horizons vides et les courbes sans remplissage.
Les captures livrées dans `apps/hyprl-site/docs/member/screenshots/v2/` portent
la bannière synthétique et ne constituent pas des performances réelles.

Pour installer le timer **membre uniquement**, voir le renderer
`scripts.member_export.jobs_v2`. Il maintient le producteur v1 inchangé puis le
v2, toutes les 30 minutes, avec AF_UNIX seulement et 300 Mo de mémoire. Il ne
rend aucun job recherche. Le OnSuccess existant de la révélation déclenche
toujours ce même service membre. Le job d’alerte existant n’est pas modifié ;
aucune notification manuelle n’est incluse dans cette tâche.

Le déploiement et le retour arrière opérateur sont décrits dans
[DEPLOY_V2.md](../apps/hyprl-site/docs/member/DEPLOY_V2.md).
