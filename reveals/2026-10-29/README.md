# Révélations — 2026-10-29

État : PARTIAL.

Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.

Engagement antérieur : [registre à ce commit](https://github.com/Kacawaiii/HyprL/blob/956fd7cd29a81e7ab426a7bead845685f5598a13/forward-duel-ledger.jsonl).

Publication confirmée : 2026-10-29T13:14:03.382296Z.

| Agent | État | SHA-256 |
| --- | --- | --- |
| HypR-A1 | REVEALED | d8cf19fa401ebdf33e1cd36c6c85fec12f944c99bc3111dd2d9b5d391fe71bc9 |
| HypR-E1 | NO_DECISION | b5bb058f4f395735b1242a7283f966ff91eb9d99ee4067e0932b5f4df020daca |

Vérification des fichiers REVEALED, après téléchargement brut :

```sh
sha256sum HypR-A1.json
```

Le résultat doit égaler l’empreinte de décision de l’engagement antérieur (voir le lien).
Les octets sont ceux de sha256_canonical, sans le champ auto-référent decision_hash : aucune indentation ni fin de ligne ajoutée.
Les noms publics HypR-A1 / HypR-E1 désignent les fichiers. Le contenu conserve les identités réelles des modèles et les métadonnées originales : c’est la piste d’audit.
UNPUBLISHED : aucune décision révélée sans publication confirmée avant l’ouverture américaine, y compris pour une session intraday.
NO_DECISION : aucun record de décision correspondant ; un engagement d’échec ne constitue pas une décision.
