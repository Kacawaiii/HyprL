# Révélations — 2026-11-06

État : PARTIAL.

Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.

Engagement antérieur : [registre à ce commit](https://github.com/Kacawaiii/HyprL/blob/8427d6c4da440672d31b65b7e05fd9085558d726/forward-duel-ledger.jsonl).

Publication confirmée : 2026-11-06T14:11:48.909805Z.

| Agent | État | SHA-256 |
| --- | --- | --- |
| HypR-A1 | NO_DECISION | f047fab442269303192fabcb05054e4370e4d33c24812cb719c80d9d3283d089 |
| HypR-E1 | REVEALED | c030f0a0b50bba5ca58b1b674ae5fd9c089f5d4310ab78f70d95fe44add5041c |

Vérification des fichiers REVEALED, après téléchargement brut :

```sh
sha256sum HypR-E1.json
```

Le résultat doit égaler l’empreinte de décision de l’engagement antérieur (voir le lien).
Les octets sont ceux de sha256_canonical, sans le champ auto-référent decision_hash : aucune indentation ni fin de ligne ajoutée.
Les noms publics HypR-A1 / HypR-E1 désignent les fichiers. Le contenu conserve les identités réelles des modèles et les métadonnées originales : c’est la piste d’audit.
UNPUBLISHED : aucune décision révélée sans publication confirmée avant l’ouverture américaine, y compris pour une session intraday.
NO_DECISION : aucun record de décision correspondant ; un engagement d’échec ne constitue pas une décision.
