# HYPRL / Eclipse

Prototype visuel autonome : présentation, univers WebGL 3D en quatre chapitres et aperçu d'espace personnel. La direction artistique reprend les références fournies : une éclipse avec anneau et astéroïdes, une couronne en filaments, un monolithe dans la brume, un orbe prismatique qui éclate, un trou noir doré avec un vaisseau, et une landing sombre avec faisceau et cartes de verre. Tout est procédural : aucune image de référence n'est copiée ni chargée.

## Les quatre chapitres

| Chapitre | Section | Ce qu'on voit | Interaction |
| --- | --- | --- | --- |
| 01 Éclipse | héro | astre noir, couronne en filaments, liseré argenté, anneau orbital incliné, ceinture d'astéroïdes, débris au premier plan, diamant + traînée anamorphique, faisceau, cartes de verre | le diamant suit le curseur autour du limbe ; parallaxe en profondeur |
| 02 Monolithe | Vision | brume au sol, crêtes enneigées, pic à droite, anneau planétaire géant, soleil rasant à 4 rayons, monolithe | le soleil glisse sur l'horizon avec le curseur ; le scroll avance la caméra |
| 03 Prisme | Plateforme | orbe fissuré (Voronoï), éclats de verre irisés, lignes de lumière, poussière, sol en lattes, aberration chromatique | le curseur tourne le nuage d'éclats ; le scroll fait éclater l'orbe |
| 04 Singularité | Interlude + Approche | disque d'accrétion argenté, image lentille, anneau de photons, nappes de brume, traînées de vitesse, vaisseau | le vaisseau suit le curseur et s'incline ; le scroll recule la caméra |

**Chaque chapitre suit sa référence.** Éclipse : espace bleu nuit, flammes violet-magenta, limbe doré dans un losange crème à quatre pointes, traînée violette. Monolithe : monochrome, brume, lumière rasante. Prisme : indigo profond, grandes lames de verre irisées aux franges chromatiques, reflet au sol. Singularité : disque d'accrétion doré, poussière chaude, paillettes d'or. Les textes gardent un halo sombre discret pour rester lisibles sur les scènes lumineuses.

Un seul canvas fixe rend les chapitres. Au scroll, la caméra plonge dans le chapitre courant (zoom et flou radial) pendant que le suivant s'ouvre depuis le centre derrière un fin liseré de lumière, comme un limbe d'éclipse. Le défilement est amorti à la molette et au clavier (`lib/smooth-scroll.js`, piloté par la boucle de rendu pour que la page et la 3D bougent dans la même image) ; le tactile garde son inertie native. Chaque bloc de texte apparaît en montant depuis un léger flou, en cascade dans sa section. Le post-traitement est écrit à la main : aberration chromatique, bloom, vignette, grain et tonemapping ACES. Les ambiances **Or** et **Glace** recolorent les quatre chapitres.

## Ouvrir

Depuis ce dossier :

```bash
python3 -m http.server 8093 --bind 0.0.0.0
```

Ouvrir `http://localhost:8093/` pour la présentation, ou `http://localhost:8093/studio.html` pour choisir un chapitre et exporter. Un serveur HTTP est nécessaire pour les modules JavaScript ; ne pas ouvrir le HTML via `file://`.

Sous Windows, extraire l'archive puis double-cliquer sur `START-WINDOWS.cmd` (Python 3 requis). Fermer la fenêtre du serveur pour arrêter l'aperçu. Le port 8093 doit être disponible.

## Livrables

- `scene.js` : le moteur. Il gère le canvas fixe, le fondu entre chapitres selon le scroll, le post-traitement, la résolution adaptative, la pause et le mouvement réduit, et les exports.
- `chapters/eclipse.js`, `monolith.js`, `prism.js`, `singularity.js` : un fichier par chapitre. Chaque objet est nommé et éditable.
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
