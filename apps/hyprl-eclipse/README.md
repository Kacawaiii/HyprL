# HYPRL / Eclipse

Prototype visuel autonome : présentation, univers WebGL 3D en quatre chapitres et aperçu d'espace personnel. La direction artistique reprend les références fournies : une éclipse avec anneau et astéroïdes, une couronne en filaments, un monolithe dans la brume, un orbe prismatique qui éclate, un trou noir doré avec un vaisseau, et une landing sombre avec faisceau et cartes de verre. Tout est procédural : aucune image de référence n'est copiée ni chargée.

## Les quatre chapitres

| Chapitre | Section | Ce qu'on voit | Interaction |
| --- | --- | --- | --- |
| 01 Monolithe | héro | plaine sombre, petit monolithe lointain, arête enneigée adoucie et pic à droite, bancs de brume à plusieurs profondeurs dans les vallées, neige arrachée à la crête, rayons du soleil occultés par le relief, poussières fines, anneau planétaire géant ; les cartes de verre du héro (CSS3D) | le soleil glisse avec le curseur ; le scroll avance la caméra |
| 02 Planète | Vision | la géante annelée du ciel du Monolithe, en croissant : nuages en bandes, limbe éclairé, ombre des anneaux sur la planète et de la planète sur les anneaux | le scroll rapproche la caméra du limbe ; le curseur tourne la vue |
| 03 Singularité | Interlude | disque d'accrétion doré vu par la tranche comme une mer miroir, anneau lentille géant en filaments qui tournent (photon ring blanc, vague à gauche), reflet de l'anneau sous la mer, nuages dorés en volumes à cœurs sombres et bords lumineux, paillettes ponctuelles intégrées, étoiles déviées par la lentille, scintillement de chaleur, traînées de vitesse et étincelles, vaisseau en métal éclairé avec son reflet, halo anamorphique | le vaisseau suit le curseur ; le scroll agrandit l'ombre et accélère le flux |
| 04 Prisme | Plateforme + Approche | verre brisé : éclats polygonaux irréguliers (triangles, lamelles, trapèzes) à biseaux brillants, paillettes de verre pilé, réfraction chromatique de l'orbe, de l'atmosphère et du sol rendus ; orbe à nébuleuse et liseré fin qui éclate vers la droite sous un trait de flare rose ; lignes de lumière fines ; grands éclats de premier plan flous et chromatiques, flou de profondeur des fragments éloignés, reflets spéculaires mobiles, faisceaux diffus depuis l’orbe à travers le verre ; parquet indigo avec caustiques mouvantes et reflets étirés des éclats sous un large éblouissement blanc | le curseur tourne le nuage d'éclats ; le scroll fait éclater l'orbe |

**Transitions, une par passage :** Monolithe → Planète, un recul continu : la caméra s'éloigne du monolithe et prend de l'altitude, le relief rétrécit sous elle, puis une brume gris-blanc lumineuse masque le passage au limbe éclairé de la planète ; la caméra planétaire démarre dans le halo de son atmosphère et recule jusqu'au cadrage normal. Planète → Singularité, la caméra continue de s'éloigner en accélérant : planète et anneaux deviennent un point lumineux, les étoiles s'étirent radialement avec une légère dispersion chromatique, un bref flash chaud passe, puis le petit anneau de Singularité, sa mer et son panache arrivent du centre et grandissent jusqu'au cadrage approuvé pendant que les traînées ralentissent. Aucun cadre 2D rétréci, masque en iris ou vortex. Singularité → Prisme conserve l'écran brisé : l'image se fissure depuis un point d'impact puis part en éclats de verre 3D qui volent vers la caméra et révèlent le prisme (`lib/shatter.js`). Remonter la page rejoue chaque transition à l'envers ; le mouvement réduit utilise des fondus simples.

## Ouvrir

Depuis ce dossier :

```bash
python3 -m http.server 8093 --bind 127.0.0.1
```

Ouvrir `http://localhost:8093/` pour la présentation, ou `http://localhost:8093/studio.html` pour choisir un chapitre et exporter. Un serveur HTTP est nécessaire pour les modules JavaScript ; ne pas ouvrir le HTML via `file://`.

Sous Windows, extraire l'archive puis double-cliquer sur `START-WINDOWS.cmd` (Python 3 requis). Fermer la fenêtre du serveur pour arrêter l'aperçu. Le port 8093 doit être disponible.

## Livrables

- `scene.js` : le moteur. Prisme utilise une copie HDR du fond à demi-résolution, puis dessine le verre et les lumières : aucune image externe et aucune boucle de lecture/écriture du même framebuffer. Les autres chapitres gardent leur passe habituelle. Il gère le canvas fixe, le fondu entre chapitres selon le scroll, le post-traitement, la résolution adaptative, la pause et le mouvement réduit, et les exports.
- `chapters/monolith.js`, `planet.js`, `singularity.js`, `prism.js` : un fichier par chapitre. Chaque objet est nommé et éditable.
- Pendant les deux premiers passages, `scene.js` transmet à `update` un objet facultatif `transition: { role: 'out' | 'in', progress, kind }`. La progression est celle du scroll (0 à 1), identique dans les deux sens ; `local` est fixé au cadrage de début ou de fin du chapitre pendant le passage. Les chapitres déplacent leurs caméras, et Singularité projette son décor analytique à l'échelle de son arrivée ; la passe finale ajoute seulement le fondu de brume, les traînées et le flash. Hors transition, les cadrages et animations approuvés restent les mêmes. Les volumes de brume et les faisceaux sont des approximations procédurales projetées, sans ray marching plein écran. Prisme utilise cinq échantillons d'ouverture sur ses petits éclats éloignés et des silhouettes douces au premier plan, en gardant le grand V net. Les reflets étirés et les empreintes de caustiques sont dessinés sans seconde scène de réflexion.
- `lib/shatter.js` : l'écran qui se brise (fracture de Voronoï en plaques de verre épaisses : biseaux, réfraction, franges RVB ; un seul appel de dessin). `lib/cards.js` : les cartes du héro.
- `lib/smooth-scroll.js` : un seul amortissement pour la molette avec pointeur fin ; le clavier et les liens activés au clavier sont immédiats, les ancres au pointeur suivent un trajet interruptible de 420 à 900 ms. Le tactile et le mouvement réduit gardent le défilement natif. Un changement de préférence est pris en compte sans rechargement.
- `lib/kit.js` : le bruit GLSL, les astéroïdes procéduraux, le champ d'étoiles et les copies portables pour le glTF.
- `index.html`, `styles.css`, `app.js` : la présentation en français. Le texte du héro est dans le disque noir, le texte de la Vision reste fixé dans le ciel, un rail de chapitres est à gauche et les contrôles d'ambiance sont fixes.
- `studio.html` : un chapitre par bouton, l'export `.glb` des quatre chapitres et l'export `.png` du chapitre affiché.
- `hyprl-universe.glb` : la géométrie et les matériaux PBR des quatre chapitres (astre, anneau, astéroïdes, cartes, monolithe, terrain, orbe, éclats, horizon, disque, vaisseau). glTF Validator : zéro erreur, zéro avertissement.
- `previews/` : les captures vérifiées dans Chromium.

## Intégration

`createEclipseScene(container, labelContainer, { hero, stops, studio })` renvoie `setPalette`, `setPaused`, `setChapter`, `exportGLB`, `exportPNG`, `chapter` et `dispose`. Le paramètre `container` est le calque fixe plein écran, et `labelContainer` couvre le héro (textes des cartes en CSS3D). Le paramètre `stops` liste les sections portant `data-chapter` (`monolith`, `planet`, `singularity`, `prism`) et, en option, `data-dim` (de 0 à 1) pour assombrir la scène sous un texte. Une intégration React doit appeler `dispose()` au démontage.

La scène demande WebGL 2. Elle se met en pause quand l'onglet est caché, respecte `prefers-reduced-motion` (poses fixes et fondus simples entre chapitres, sans warp, éclatement ni déplacement de caméra ; le scroll change toujours de chapitre) et baisse sa résolution toute seule si le GPU peine. Sur mobile, elle allège les astéroïdes, les éclats, les traînées et l'anticrénelage. Sans WebGL, chaque section reçoit un décor CSS.

## Mouvement et lisibilité

La caméra lit la position de la page à chaque image : pas de second retard après le défilement. Remonter le scroll inverse exactement les trajets et la fracture à temps d’animation égal ; le scintillement ambiant continue lorsque l’animation tourne. Les révélations utilisent des transitions d’opacité et de translation de 280 ms, une courbe `cubic-bezier(0.23,1,0.32,1)` et un décalage de 40 ms, sans flou. Le clavier les affiche immédiatement. En mouvement réduit, les fondus et retours de couleur sont conservés ; les déplacements sont retirés.

Les textes et les légendes de la plateforme ont des fonds sombres locaux pour traverser la brume et l’éblouissement du parquet. Sur téléphone, le rail devient quatre liens de 44 px, à côté des contrôles d’ambiance. Les ancres transfèrent aussi le focus à leur destination. Le rail reste utilisable sans WebGL. La fenêtre de l’espace restaure la pause choisie par l’utilisateur, y compris si sa préférence de mouvement change pendant son ouverture.

Le verre conserve la copie HDR du décor à demi-résolution : réfraction avec IOR 1,52, absorption selon l’épaisseur optique, dispersion fine, biseaux et caustiques procédurales. La chaîne reste HDR linéaire → bloom à seuil par chapitre → exposition/vignette → ACES → conversion de sortie et grain léger. Sur mobile : pas de MSAA dans les cibles, deux passes de flou au lieu de quatre, pas de flare anamorphique ni de copies d’ouverture pour les fragments. Le DPR reste plafonné à 1,25 sur mobile et 1,5 sur desktop, avec résolution adaptative. La boucle est annulée pendant que l’onglet est caché et reprend avec une horloge réinitialisée.

## Périmètre

L'espace personnel est une démonstration : pas d'authentification, de paiement, de connexion au broker ou de données live. Le journal est enregistré uniquement dans le stockage local de ce navigateur. Aucune performance, clientèle ou affiliation n'est revendiquée.

Les métadonnées de titre/description et le contenu sémantique sont prêts pour une intégration. Le prototype est volontairement `noindex` : avant publication, ajouter la bonne URL canonique, les images Open Graph, le sitemap et les pages légales, puis retirer `noindex`. Aucun résultat de classement SEO n'est garanti.

## Dépendances locales

Three.js `0.180.0` (MIT, cœur seul : le post-traitement est écrit à la main), Manrope via `@fontsource/manrope` `5.2.6` (OFL). Sources et licences dans `vendor/`. Aucune CDN ou API externe n'est requise à l'ouverture. Aucun abonnement Spline/Higgsfield, poids de modèle ML ou GPU serveur n'est requis ; le navigateur utilise WebGL.

Les dépôts étudiés sont [WorldSculpt](https://github.com/AlayaLab/WorldSculpt), un pipeline de reconstruction 3D, et [UniMate](https://github.com/Friedrich-M/UniMate), un modèle d'animation de squelettes. Ils ne sont pas utilisés dans ce prototype.

## Vérification

Vérification initiale dans Chromium (SwiftShader) : les 4 chapitres en desktop 1440 px et en mobile 390 px ; les palettes Or et Glace ; le fondu entre chapitres et le rail ; la fenêtre de l'espace (pause et reprise, Échap, focus restitué) ; le mouvement réduit ; le repli sans WebGL ; les exports GLB et PNG. Aucune erreur console et aucune requête externe. glTF Validator : zéro erreur, zéro avertissement. Les résultats sont dans `previews/verification.json`. Ces contrôles ne remplacent pas une mesure de performance sur de vrais appareils : à faire avant publication, surtout sur mobile.


La passe qualité vérifie aussi des planches de neuf images pour chaque transition, dans les deux sens, en 1440×900 et 390×844 ; le retour du canvas au même pixel à temps d’animation figé ; le clavier, le tactile, les préférences de mouvement changées à chaud et le cycle pause/reprise de la fenêtre. Chaque JavaScript modifié passe `node --check`. Le coût des images est comparé au même viewport/DPR et à qualité fixe dans SwiftShader, avec quatre images de chauffe puis une lecture de pixel qui attend la fin du rendu. La résolution adaptative est désactivée pour cette comparaison.
