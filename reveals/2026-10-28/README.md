# Révélations — 2026-10-28

État : REVEALED.

Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.

Engagement antérieur : [registre à ce commit](https://github.com/Kacawaiii/HyprL/blob/b68dcd9e32f37893a79c6089dcdd3db9fae404c8/forward-duel-ledger.jsonl).

Publication confirmée : 2026-10-28T13:13:18.958456Z.

| Agent | État | SHA-256 |
| --- | --- | --- |
| HypR-A1 | REVEALED | e38720ea52e92eb7c1606701bf93eb7eb9b48167a03553ff4e5bc578186a13c0 |
| HypR-E1 | REVEALED | cf959cd9c1158232b045995c3be6c2a75c71e11237c8dfd9bf7cb6a7a7e59470 |

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
