# Révélations — 2026-10-22

État : REVEALED.

Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.

Engagement antérieur : [registre à ce commit](https://github.com/Kacawaiii/HyprL/blob/50ac1f8c829dc4dee8b8acb81a86e2abe41d264a/forward-duel-ledger.jsonl).

Publication confirmée : 2026-10-22T13:16:34.210495Z.

| Agent | État | SHA-256 |
| --- | --- | --- |
| HypR-A1 | REVEALED | 9d1f9e757d34aedebf60f46f1d360851eb94c4265372ae1e4ac08c24b6bc9e0d |
| HypR-E1 | REVEALED | 8090e07d6311c63a33e410fce818c55e5f34ac2a55a6488e0801fd3287365ae8 |

Vérification des fichiers REVEALED, après téléchargement brut :

```sh
sha256sum HypR-A1.json
sha256sum HypR-E1.json
```

Le résultat doit égaler l’empreinte de décision de l’engagement antérieur (voir le lien).
Les octets sont ceux de sha256_canonical, sans le champ auto-référent decision_hash : aucune indentation ni fin de ligne ajoutée.
Les noms publics HypR-A1 / HypR-E1 désignent les fichiers. Le contenu conserve les identités réelles des modèles et les métadonnées originales : c’est la piste d’audit.
UNPUBLISHED : aucune décision révélée sans publication confirmée avant l’ouverture américaine, y compris pour une session intraday.
NO_DECISION : aucun record de décision correspondant ; un engagement d’échec ne constitue pas une décision.
