# HYPRL / Eclipse

Prototype visuel autonome : présentation, scène WebGL 3D et aperçu d'espace personnel. Direction inspirée des sept références fournies : noir, éclipse, lumière champagne, accents bleus, verre fumé. Aucun visuel Pinterest n'est copié dans le site.

## Ouvrir

Depuis ce dossier :

```bash
python3 -m http.server 8093 --bind 0.0.0.0
```

Ouvrir `http://localhost:8093/` pour la présentation, ou `http://localhost:8093/studio.html` pour la scène seule et les exports. Un serveur HTTP est nécessaire pour les modules JavaScript ; ne pas ouvrir le HTML via `file://`.

Sous Windows, extraire l'archive puis double-cliquer sur `START-WINDOWS.cmd` (Python 3 requis). Fermer la fenêtre du serveur pour arrêter l'aperçu. Le port 8093 doit être disponible.

## Livrables

- `scene.js` : planète et halo, atmosphère procédurale, faisceau, étoiles, panneaux 3D, animation, parallax, palettes or/glace. Les objets sont nommés et éditables.
- `index.html`, `styles.css`, `app.js` : présentation française, contenu HTML indexable, interface responsive, espace démo avec onglets et journal local.
- `studio.html` : export `.glb` de la géométrie et des matériaux PBR ; export `.png` du fond WebGL.
- `previews/` : captures du rendu vérifié dans Chromium.
- `hyprl-eclipse.glb` : géométrie exportée et validée, prête à importer dans un logiciel 3D compatible glTF.

Le GLB exporte la planète, les panneaux et un faisceau simplifié. Les shaders procéduraux, les animations et les textes HTML sont propres au navigateur et restent dans `scene.js`. Ce n'est pas un fichier natif Spline. Un moteur cible doit refaire l'éclairage pour reproduire exactement le rendu web.

## Intégration

`createEclipseScene(container, labelContainer)` retourne `setPalette`, `setPaused`, `exportGLB`, `exportPNG`, `dispose`. Les deux conteneurs doivent partager le même rectangle. Les textes sur les panneaux restent en HTML via CSS3DRenderer ; la géométrie est rendue en WebGL. Une intégration React doit appeler `dispose()` au démontage.

La scène se met en pause hors écran ou en onglet caché, respecte `prefers-reduced-motion`, limite la résolution et simplifie les panneaux sur mobile. Un fond CSS prend le relais si WebGL est indisponible.

## Périmètre

L'espace personnel est une démonstration : pas d'authentification, de paiement, de connexion au broker ou de données live. Le journal est enregistré uniquement dans le stockage local de ce navigateur. Aucune performance, clientèle ou affiliation n'est revendiquée.

Les métadonnées de titre/description et le contenu sémantique sont prêts pour une intégration. Le prototype est volontairement `noindex` : avant publication, ajouter la bonne URL canonique, les images Open Graph, le sitemap et les pages légales, puis retirer `noindex`. Aucun résultat de classement SEO n'est garanti.

## Dépendances locales

Three.js `0.180.0` (MIT), Manrope via `@fontsource/manrope` `5.2.6` (OFL). Sources et licences dans `vendor/`. Aucune CDN ou API externe n'est requise à l'ouverture. Aucun abonnement Spline/Higgsfield, poids de modèle ML ou GPU serveur n'est requis ; le navigateur utilise WebGL.

Les dépôts étudiés sont [WorldSculpt](https://github.com/AlayaLab/WorldSculpt), un pipeline de reconstruction 3D, et [UniMate](https://github.com/Friedrich-M/UniMate), un modèle d'animation de squelettes. Ils ne sont pas utilisés dans ce prototype.

## Vérification

Chromium : desktop 1440 px, mobile 390 et 320 px, navigation clavier, fermeture et restitution du focus, palettes, mouvement/pause, préférence de mouvement réduit, repli sans WebGL, persistance du journal et rendu des notes comme texte. Exports GLB/PNG vérifiés ; glTF Validator : zéro erreur et zéro avertissement. Aucune requête vers un service externe à l'ouverture. Résultats dans `previews/verification.json`. Ces contrôles locaux ne remplacent pas les mesures sur appareils physiques avant publication.
