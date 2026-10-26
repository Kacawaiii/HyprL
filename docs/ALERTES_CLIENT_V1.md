# Alertes HyprL — accès anticipé

Les alertes transmettent des faits et leurs sources. Chaque message indique sa
date de publication, sa date de disponibilité et la mention **« information,
pas un conseil en investissement »**. Les noms publics des agents sont
**HypR-A1** et **HypR-E1**. Les performances publiées ne prédisent pas les
performances futures ; le duel et l'indicateur de chance décrivent des observations.

La première étape est réservée à l'équipe : seuls les comptes explicitement
autorisés peuvent configurer des destinations de test. Aucune destination client
réelle n'est activée par cette livraison.

## Choisir les informations

| Type dans la commande | Informations transmises |
| --- | --- |
| `sec_filing` | Dépôt SEC : 8-K, 10-Q, 10-K, S-1 ou Form 4 ; CIK et accession |
| `fed_release`, `cpi_release`, `nfp_release` | Publication Fed ou chiffres CPI/NFP sourcés |
| `earnings_date`, `earnings_result` | Date prévue ou résultats publiés : période, EPS, chiffre d'affaires, devise |
| `unusual_move` | Variation de prix et multiple de volume sur une fenêtre indiquée |
| `radar_headline` | Nouvelle actualité à fort impact : catégorie et lien source |
| `duel_hashes`, `duel_results` | Empreintes avant ouverture, puis résultats après clôture |
| `weekly_luck` | Percentiles observés et nombre d'observations de la semaine |

Les seuils de détection et la disponibilité des données dépendent du producteur
de données raccordé par l'équipe. Le routeur ne collecte pas lui-même les sources.
Un type pris en charge ne garantit donc pas qu'un flux soit déjà actif. Les
horodatages sont UTC. Une date de résultats prévue peut être modifiée par l'émetteur.
Le radar transmet un intitulé fixe et la catégorie de l'événement ; les extraits
de presse et les commentaires des modèles restent dans la source consultable.

## Connecter un canal

Dans le serveur HyprL, le rôle **Accès anticipé** est nécessaire. Utilisez
`/alertes ajouter`, choisissez le canal, puis renseignez le formulaire privé.
L'URL n'est jamais affichée dans les réponses ou dans la liste des destinations.
Les réponses sont éphémères, visibles seulement par vous. Discord reçoit toutefois
les données saisies dans le formulaire : traitez toujours une URL de webhook comme
un secret et révoquez-la chez le fournisseur si elle a été divulguée.

Exemple de paramètres sans secret :

```text
/alertes ajouter canal:discord symboles:AAPL,MSFT types:sec_filing,earnings_date,earnings_result
```

Pour les événements associés à des titres, seuls ceux qui croisent votre liste
de symboles sont envoyés. Une liste vide ne reçoit aucun événement associé à un
titre. Les événements globaux suivent uniquement les types choisis. Sans le
paramètre `types`, tous les types sont sélectionnés. Limites v1 : 50 symboles et
10 destinations par compte. Les symboles doivent correspondre à ceux du flux.

| Canal | Préparation et donnée à saisir dans le formulaire |
| --- | --- |
| Discord | Dans un salon de test dont vous avez la gestion, créez un webhook entrant dans les intégrations du salon. Saisissez son URL `https://discord.com/api/webhooks/…`. |
| Slack | Faites créer un webhook entrant autorisé pour votre espace et votre salon, puis saisissez l'URL `https://hooks.slack.com/services/…`. |
| Microsoft Teams | Faites créer un webhook entrant ou un workflow recevant des cartes adaptatives dans le canal. Saisissez l'URL HTTPS fournie. L'administrateur du tenant doit autoriser ce mécanisme et, pour un workflow, l'authentification attendue doit permettre cet envoi. |
| Telegram | Ouvrez le bot HyprL indiqué par l'équipe et lancez `/start`, ou ajoutez-le à votre groupe de test. Communiquez à l'équipe l'identifiant du chat pour vérification ; seul un identifiant approuvé pour votre compte est accepté. Saisissez ensuite cet identifiant. Vous ne fournissez jamais de jeton de bot. |
| Webhook personnalisé | Fournissez votre endpoint HTTPS public et une clé de signature aléatoire de 32 à 128 caractères ASCII. La clé doit aussi être installée sur votre récepteur. Voir le protocole ci-dessous. |
| E-mail | Prévu, sans envoi ni collecte d'adresse tant que l'expéditeur n'est pas choisi. |
| API | Contrat de lecture défini, avec un jeton par compte. Aucun endpoint n'est exposé à ce stade. |

Les URL Teams reconnues sont les sous-domaines de `webhook.office.com`,
`logic.azure.com` et `environment.api.powerplatform.com`. Les liens nécessitant
une authentification supplémentaire ou un autre format devront être validés
par l'équipe avant activation.

## Tester, mettre en pause, supprimer

1. `/alertes liste` affiche l'identifiant, le canal, l'état et un bouton
   **Mettre en pause** pour chaque destination. Ce bouton coupe cette destination
   en un clic depuis le panneau privé Discord, quel que soit son canal.
2. `/alertes test destination:IDENTIFIANT` met en file un message clairement
   marqué **TEST STAFF — données synthétiques**. « En file » ne confirme pas
   encore sa réception. Vérifiez le salon et faites contrôler l'état par l'équipe.
3. `/alertes pause destination:IDENTIFIANT active:True` suspend les envois.
   `active:False` les reprend.
4. `/alertes supprimer destination:IDENTIFIANT` supprime la configuration et
   les envois associés. Révoquez aussi le webhook chez son fournisseur si nécessaire.

La pause bloque les prochains départs. Un envoi déjà parti peut encore arriver
pendant au plus cinq secondes. Les messages déjà en attente restent en file ;
les nouveaux événements reçus pendant la pause ne sont pas ajoutés à cette
destination. La suppression vide sa file. Les événements du flux de lecture
restent soumis à la rétention de 30 jours.

Le réglage initial limite chaque destination à un départ toutes les dix secondes
et 60 départs par heure UTC, reprises comprises. Les limites peuvent retarder une
alerte. Après cinq tentatives au maximum, un échec rejoint la file d'échec pour
examen par l'équipe. Un message peut être reçu deux fois si le fournisseur l'a
accepté avant une interruption réseau ; aucun mécanisme ne garantit ici une
livraison exactement une fois.

## Protocole du webhook personnalisé

Le JSON UTF-8 respecte [le schéma v1](alerts/payload-v1.schema.json). Il porte
`schema_version: 1`, `event_id`, `kind`, `occurred_at`, `available_at`, `source`
(`name`, `url`, `sha256`), `symbols`, `facts`, `agents` et `disclaimer`.
Les nombres décimaux sont des chaînes afin de conserver leur précision. Les
champs supplémentaires sont refusés. Le digest de source identifie la donnée
locale fournie par le producteur ; ce n'est pas une certification de son exactitude.

```text
Content-Type: application/json
X-HyprL-Schema-Version: 1
X-HyprL-Timestamp: <secondes Unix de cette tentative>
X-HyprL-Signature: sha256=<hex HMAC-SHA256>
Idempotency-Key: <event_id>
```

La signature couvre exactement `timestamp + "." + corps_brut`. Vérifiez-la en
temps constant avec votre clé, refusez un écart supérieur à 300 secondes et
conservez les `event_id` déjà traités pour dédupliquer. Ne re-sérialisez pas le
JSON avant la vérification. Chaque nouvelle tentative reçoit un nouvel horodatage
et une nouvelle signature. Répondez rapidement avec un statut 2xx.

Les adresses privées, locales, de liaison, de métadonnées et les réseaux spéciaux
sont refusés. Toutes les réponses DNS doivent être publiques. Les redirections
ne sont jamais suivies et le délai total est de cinq secondes. Un statut 3xx
ou 4xx est définitif, sauf 408/429. Les erreurs réseau, 408, 429 et 5xx peuvent
être reprises, avec espacement croissant et prise en compte de `Retry-After`
numérique, plafonné à une heure.
