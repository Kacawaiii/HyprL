# Révélations — 2026-11-09

État : PARTIAL.

Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.

Engagement antérieur : [registre à ce commit](https://github.com/Kacawaiii/HyprL/blob/c67822a194fdc1887f1bb851b2dd45a4103498e6/forward-duel-ledger.jsonl).

Publication confirmée : 2026-11-09T14:11:34.463755Z.

| Agent | État | SHA-256 |
| --- | --- | --- |
| HypR-A1 | NO_DECISION | b528307d7171732f2615dfa69d285474aa43be42ea6bc40961d07e6bfaf628c7 |
| HypR-E1 | REVEALED | 1911b19a8f5d375b69afd08650aa7b2e16d15e8d9d080f73b31e3ab9c4f5f237 |

Vérification des fichiers REVEALED, après téléchargement brut :

```sh
sha256sum HypR-E1.json
```

Le résultat doit égaler l’empreinte de décision de l’engagement antérieur (voir le lien).
Les octets sont ceux de sha256_canonical, sans le champ auto-référent decision_hash : aucune indentation ni fin de ligne ajoutée.
Les noms publics HypR-A1 / HypR-E1 désignent les fichiers. Le contenu conserve les identités réelles des modèles et les métadonnées originales : c’est la piste d’audit.
UNPUBLISHED : aucune décision révélée sans publication confirmée avant l’ouverture américaine, y compris pour une session intraday.
NO_DECISION : aucun record de décision correspondant ; un engagement d’échec ne constitue pas une décision.
