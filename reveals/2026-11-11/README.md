# Révélations — 2026-11-11

État : PARTIAL.

Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.

Engagement antérieur : [registre à ce commit](https://github.com/Kacawaiii/HyprL/blob/688baa02740a3a777c3d9e32e9dbc5cf87bdabd0/forward-duel-ledger.jsonl).

Publication confirmée : 2026-11-11T14:12:33.574381Z.

| Agent | État | SHA-256 |
| --- | --- | --- |
| HypR-A1 | NO_DECISION | 0e10db70de4b6f970fe1359208ad75e2c4bc5bcc6868d99e34b2bef5824d8731 |
| HypR-E1 | REVEALED | 47c0c0f640ca827df3f7604bbefcb169e28ecd31fb91a2431a71a0527a513fb0 |

Vérification des fichiers REVEALED, après téléchargement brut :

```sh
sha256sum HypR-E1.json
```

Le résultat doit égaler l’empreinte de décision de l’engagement antérieur (voir le lien).
Les octets sont ceux de sha256_canonical, sans le champ auto-référent decision_hash : aucune indentation ni fin de ligne ajoutée.
Les noms publics HypR-A1 / HypR-E1 désignent les fichiers. Le contenu conserve les identités réelles des modèles et les métadonnées originales : c’est la piste d’audit.
UNPUBLISHED : aucune décision révélée sans publication confirmée avant l’ouverture américaine, y compris pour une session intraday.
NO_DECISION : aucun record de décision correspondant ; un engagement d’échec ne constitue pas une décision.
