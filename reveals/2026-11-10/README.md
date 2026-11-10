# Révélations — 2026-11-10

État : PARTIAL.

Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.

Engagement antérieur : [registre à ce commit](https://github.com/Kacawaiii/HyprL/blob/081099e5fbd1e4db8f525fcc5788a2390cf0c5ff/forward-duel-ledger.jsonl).

Publication confirmée : 2026-11-10T14:13:24.030045Z.

| Agent | État | SHA-256 |
| --- | --- | --- |
| HypR-A1 | NO_DECISION | 96f449b6844b3dbc77b310b47e23e0efd0366b1e1e90b9dd276d9e4c0b9058aa |
| HypR-E1 | REVEALED | 75c639a2ada271450e7e7a1df0ec4652094cadb571a8a690c338d2f6dc1dc9cb |

Vérification des fichiers REVEALED, après téléchargement brut :

```sh
sha256sum HypR-E1.json
```

Le résultat doit égaler l’empreinte de décision de l’engagement antérieur (voir le lien).
Les octets sont ceux de sha256_canonical, sans le champ auto-référent decision_hash : aucune indentation ni fin de ligne ajoutée.
Les noms publics HypR-A1 / HypR-E1 désignent les fichiers. Le contenu conserve les identités réelles des modèles et les métadonnées originales : c’est la piste d’audit.
UNPUBLISHED : aucune décision révélée sans publication confirmée avant l’ouverture américaine, y compris pour une session intraday.
NO_DECISION : aucun record de décision correspondant ; un engagement d’échec ne constitue pas une décision.
