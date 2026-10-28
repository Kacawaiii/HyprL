# Captures synthétiques du contrat v2

Ces images montrent exclusivement le bootstrap de test, avec sa bannière
« Aperçu de test ». Elles ne montrent aucun résultat réel du duel. Les fichiers
du dossier parent documentent l’ancienne interface v1.

| Capture | Vérification |
| --- | --- |
| duel-synthetic-light.png / duel-synthetic-dark.png | Deux courbes par horizon et mode ; vraies observations du jeu synthétique, sans point de remplissage |
| duel-empty-dark.png | Export vide, aucune courbe |
| calendar-date-only.png | Date seule, heure inconnue, preuve de repère observé |
| revenue-empty.png | Aucun intervalle SEC reconstruit, flux vide |
| luck-synthetic-light.png | Probabilité de chance non publiée |
| radar-synthetic-dark.png / radar-synthetic-mobile.png | Sources, dates et liste privée ; adaptation à 320 px |

Reproduction locale : installer les dépendances Node du site et les dépendances
Python de `test/requirements-qa.txt`, puis lancer les deux bootstraps
`test/member-browser-server.js` (ports 3827 et 3828 avec `--empty`) et
`test/member_browser_audit.py`. L’audit vérifie aussi le passage d’un horizon
noté à un horizon vide, les preuves de séance et l’absence de requête externe.

Le test `member_observed_audit.py` peut lire un snapshot réel **hors Git** via
`MEMBER_QA_SNAPSHOT`, avec le bootstrap `--observed` sur le port 3829. Il utilise
le lecteur Node réel et une session de test locale. Les captures éventuelles
restent dans un répertoire privé externe, défini par
`MEMBER_QA_SCREENSHOT_DIR`. Ce contrôle ne vaut pas acceptation OAuth réelle.
