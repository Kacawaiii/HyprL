# Révélations — 2026-10-23

État : REVEALED.

Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.

Engagement antérieur : [registre à ce commit](https://github.com/Kacawaiii/HyprL/blob/70d13b47e5cbd451cdca06247739ecb04ca710de/forward-duel-ledger.jsonl).

Publication confirmée : 2026-10-23T13:16:32.639959Z.

| Agent | État | SHA-256 |
| --- | --- | --- |
| HypR-A1 | REVEALED | f809ddd7fccee9b0631a8cff82946c66d5c77e91b114c320c1177b99c3d9c21f |
| HypR-E1 | REVEALED | c9d9abf8ab946ff24595912fb3d56ea32b16aaaf15f17cd77b37a8c6206b85e8 |

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
