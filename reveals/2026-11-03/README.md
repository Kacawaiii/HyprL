# Révélations — 2026-11-03

État : PARTIAL.

Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.

Engagement antérieur : [registre à ce commit](https://github.com/Kacawaiii/HyprL/blob/11fb86c473f974e301f655c88552da141e7c2a53/forward-duel-ledger.jsonl).

Publication confirmée : 2026-11-03T14:13:08.406019Z.

| Agent | État | SHA-256 |
| --- | --- | --- |
| HypR-A1 | NO_DECISION | 9c9a381425063c1d4de517c3e50e22f5d27718140c51e8d43e92b42f927d3081 |
| HypR-E1 | REVEALED | 534d84000a148df8d257058716bff8ae6e972bbf1c5200a48a50a4d4f7b132d6 |

Vérification des fichiers REVEALED, après téléchargement brut :

```sh
sha256sum HypR-E1.json
```

Le résultat doit égaler l’empreinte de décision de l’engagement antérieur (voir le lien).
Les octets sont ceux de sha256_canonical, sans le champ auto-référent decision_hash : aucune indentation ni fin de ligne ajoutée.
Les noms publics HypR-A1 / HypR-E1 désignent les fichiers. Le contenu conserve les identités réelles des modèles et les métadonnées originales : c’est la piste d’audit.
UNPUBLISHED : aucune décision révélée sans publication confirmée avant l’ouverture américaine, y compris pour une session intraday.
NO_DECISION : aucun record de décision correspondant ; un engagement d’échec ne constitue pas une décision.
