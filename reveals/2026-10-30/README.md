# Révélations — 2026-10-30

État : REVEALED.

Publication après la clôture américaine + 15 minutes ; contrôle du calendrier local, y compris DST et séances écourtées.

Engagement antérieur : [registre à ce commit](https://github.com/Kacawaiii/HyprL/blob/17b3c0fbf6199917dcc836ae916943f08a2bcd93/forward-duel-ledger.jsonl).

Publication confirmée : 2026-10-30T13:13:00.238871Z.

| Agent | État | SHA-256 |
| --- | --- | --- |
| HypR-A1 | REVEALED | 0a3169a8f2540c67bd3e7d198756d6e068948c4a25245cdcd512534a4e943f2b |
| HypR-E1 | REVEALED | 4a76cbc71ae4580ee2dccd12b4604cc4475642ec54f49a038bd170115833b7f4 |

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
