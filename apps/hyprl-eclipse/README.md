# HYPRL / Eclipse

Prototype visuel autonome : présentation, univers WebGL 3D en quatre chapitres et aperçu d'espace personnel. La direction artistique reprend les références fournies : une éclipse avec anneau et astéroïdes, une couronne en filaments, un monolithe dans la brume, un orbe prismatique qui éclate, un trou noir doré avec un vaisseau, et une landing sombre avec faisceau et cartes de verre. Tout est procédural : aucune image de référence n'est copiée ni chargée.

## Les quatre chapitres

| Chapitre | Section | Ce qu'on voit | Interaction |
| --- | --- | --- | --- |
| 01 Monolithe | héro | plaine sombre, petit monolithe lointain, arête enneigée et pic à droite, collines dans la brume à gauche, soleil étoilé rasant, anneau planétaire géant ; les cartes de verre du héro (CSS3D) | le soleil glisse avec le curseur ; le scroll avance la caméra |
| 02 Planète | Vision | la géante annelée du ciel du Monolithe, en croissant : nuages en bandes, limbe éclairé, ombre des anneaux sur la planète et de la planète sur les anneaux | le scroll rapproche la caméra du limbe ; le curseur tourne la vue |
| 03 Singularité | Interlude | disque d'accrétion doré vu par la tranche comme une mer miroir, anneau lentille géant en filaments qui tournent (photon ring blanc, vague à gauche), reflet de l'anneau sous la mer, nappes et panache de poussière d'or pailletée, étoiles déviées par la lentille, scintillement de chaleur, traînées de vitesse et étincelles, vaisseau en métal éclairé avec son reflet, halo anamorphique | le vaisseau suit le curseur ; le scroll agrandit l'ombre et accélère le flux |
| 04 Prisme | Plateforme + Approche | verre brisé : éclats polygonaux irréguliers (triangles, lamelles, trapèzes) à biseaux brillants, paillettes de verre pilé, réfraction chromatique de l'atmosphère ; orbe à nébuleuse et liseré fin qui éclate vers la droite sous un trait de flare rose ; lignes de lumière fines ; parquet violet réfléchissant | le curseur tourne le nuage d'éclats ; le scroll fait éclater l'orbe |

**Transitions, une par passage :** Monolithe → Planète, la caméra plonge vers la planète derrière un liseré de lumière ; Planète → Singularité, montée vers le haut, la planète tombe et le trou noir descend derrière une ligne d'horizon lumineuse ; Singularité → Prisme, l'écran se brise : l'image se fissure depuis un point d'impact puis part en éclats de verre 3D qui volent vers la caméra et révèlent le prisme (`lib/shatter.js`). Remonter la page rejoue chaque transition à l'envers.

## Ouvrir

Depuis ce dossier :

```bash
python3 -m http.server 8093 --bind 0.0.0.0
```

Ouvrir `http://localhost:8093/` pour la présentation, ou `http://localhost:8093/studio.html` pour choisir un chapitre et exporter. Un serveur HTTP est nécessaire pour les modules JavaScript ; ne pas ouvrir le HTML via `file://`.

Sous Windows, extraire l'archive puis double-cliquer sur `START-WINDOWS.cmd` (Python 3 requis). Fermer la fenêtre du serveur pour arrêter l'aperçu. Le port 8093 doit être disponible.

## Livrables

- `scene.js` : le moteur. Il gère le canvas fixe, le fondu entre chapitres selon le scroll, le post-traitement, la résolution adaptative, la pause et le mouvement réduit, et les exports.
- `chapters/monolith.js`, `planet.js`, `singularity.js`, `prism.js` : un fichier par chapitre. Chaque objet est nommé et éditable.
- `lib/shatter.js` : l'écran qui se brise (fracture de Voronoï en plaques de verre épaisses : biseaux, réfraction, franges RVB ; un seul appel de dessin). `lib/cards.js` : les cartes du héro.
- `lib/smooth-scroll.js` : le défilement amorti (molette, clavier, ancres), désactivé en mouvement réduit et sur écran tactile.
- `lib/kit.js` : le bruit GLSL, les astéroïdes procéduraux, le champ d'étoiles et les copies portables pour le glTF.
- `index.html`, `styles.css`, `app.js` : la présentation en français. Le texte du héro est dans le disque noir, le texte de la Vision reste fixé dans le ciel, un rail de chapitres est à gauche et les contrôles d'ambiance sont fixes.
- `studio.html` : un chapitre par bouton, l'export `.glb` des quatre chapitres et l'export `.png` du chapitre affiché.
- `hyprl-universe.glb` : la géométrie et les matériaux PBR des quatre chapitres (astre, anneau, astéroïdes, cartes, monolithe, terrain, orbe, éclats, horizon, disque, vaisseau). glTF Validator : zéro erreur, zéro avertissement.
- `previews/` : les captures vérifiées dans Chromium.

## Intégration

`createEclipseScene(container, labelContainer, { hero, stops, studio })` renvoie `setPalette`, `setPaused`, `setChapter`, `exportGLB`, `exportPNG`, `chapter` et `dispose`. Le paramètre `container` est le calque fixe plein écran, et `labelContainer` couvre le héro (textes des cartes en CSS3D). Le paramètre `stops` liste les sections portant `data-chapter` (`eclipse`, `monolith`, `prism`, `singularity`) et, en option, `data-dim` (de 0 à 1) pour assombrir la scène sous un texte. Une intégration React doit appeler `dispose()` au démontage.

La scène demande WebGL 2. Elle se met en pause quand l'onglet est caché, respecte `prefers-reduced-motion` (image fixe, mais le scroll change toujours de chapitre) et baisse sa résolution toute seule si le GPU peine. Sur mobile, elle allège les astéroïdes, les éclats, les traînées et l'anticrénelage. Sans WebGL, chaque section reçoit un décor CSS.

## Périmètre

L'espace personnel est une démonstration : pas d'authentification, de paiement, de connexion au broker ou de données live. Le journal est enregistré uniquement dans le stockage local de ce navigateur. Aucune performance, clientèle ou affiliation n'est revendiquée.

Les métadonnées de titre/description et le contenu sémantique sont prêts pour une intégration. Le prototype est volontairement `noindex` : avant publication, ajouter la bonne URL canonique, les images Open Graph, le sitemap et les pages légales, puis retirer `noindex`. Aucun résultat de classement SEO n'est garanti.

## Dépendances locales

Three.js `0.180.0` (MIT, cœur seul : le post-traitement est écrit à la main), Manrope via `@fontsource/manrope` `5.2.6` (OFL). Sources et licences dans `vendor/`. Aucune CDN ou API externe n'est requise à l'ouverture. Aucun abonnement Spline/Higgsfield, poids de modèle ML ou GPU serveur n'est requis ; le navigateur utilise WebGL.

Les dépôts étudiés sont [WorldSculpt](https://github.com/AlayaLab/WorldSculpt), un pipeline de reconstruction 3D, et [UniMate](https://github.com/Friedrich-M/UniMate), un modèle d'animation de squelettes. Ils ne sont pas utilisés dans ce prototype.

## Vérification

Chromium (SwiftShader) : les 4 chapitres en desktop 1440 px et en mobile 390 px ; les palettes Or et Glace ; le fondu entre chapitres et le rail ; la fenêtre de l'espace (pause et reprise, Échap, focus restitué) ; le mouvement réduit ; le repli sans WebGL ; les exports GLB et PNG. Aucune erreur console et aucune requête externe. glTF Validator : zéro erreur, zéro avertissement. Les résultats sont dans `previews/verification.json`. Ces contrôles ne remplacent pas une mesure de performance sur de vrais appareils : à faire avant publication, surtout sur mobile.
