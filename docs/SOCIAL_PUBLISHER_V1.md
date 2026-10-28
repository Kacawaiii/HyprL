# Publieur social v1 — carrousels, guide opérateur

Le programme est dans `scripts/social_publisher`. Les cinq destinations sont indépendantes,
toutes désactivées initialement, et le mode simulation est activé. Aucun compte, aucune application,
aucun jeton ni aucun audit n'est créé par cette livraison. Les appels de publication réels restent
à valider par l'opérateur. La règle SINGLE OWNER réserve l'installation à sa session.

## Contrat avec la fabrique et horaire

Le mode par défaut est désormais `media_kind=carousel`, producteur épinglé à
`8979fada748458208509b7377a45eacdaf27c3a4`, schéma `hyprl-carousel-v1`.
Le publieur lit exclusivement `carousel-v1-daily-AAAA-MM-JJ` et les reçus
`delivery.json`, `carousel-v1-daily-AAAA-MM-JJ.json`, `slot-daily-AAAA-MM-JJ.json`.
Tous doivent être acquittés, liés au manifeste canonique et au contenu attendu.
Une sortie seulement rendue, un ancien exemple ou un reçu incertain sont refusés.

Il vérifie les empreintes et tailles de tous les fichiers, légende et ZIP compris, les
listes complètes, le disclaimer et les tailles minimales de texte enregistrées. Il ouvre
chaque PNG : RGB, 1080 × 1350 pour Instagram/X, 1080 × 1920 pour TikTok. Les packs ZIP ne
sont jamais extraits. La conversion JPEG qualité 95 ne modifie ni dimensions, ni ordre,
ni texte ; les JPEG restent sous 5 Mo. Facebook reçoit le pack Instagram ; X reçoit le
pack X, au plus quatre images. La légende complète validée accompagne Meta ; X reprend
le titre, le disclaimer et les hashtags, dans sa limite de texte.

Le publieur fait confiance à la fabrique pour les faits, les sources et la validation éditoriale.
Il ne refait pas sa recherche, ne consulte aucun document officiel et ne transforme pas un
ancien brief en information du jour. Les autres créneaux de la fabrique (calendrier du lundi,
recherche du mercredi, documents) restent dans les brouillons : cette v1 automatise un seul
carrousel quotidien par plateforme. Une évolution distincte sera nécessaire pour ces créneaux.

La fabrique quotidienne démarre à 18 h 30 Paris et dispose de dix minutes. Le publieur passe à
**18 h 42 Europe/Paris**, les jours de séance NYSE ; il accepte seulement la fenêtre
18 h 42–19 h 02. Le fuseau Paris gère ses changements d'heure indépendamment de New York.
Pas de rattrapage (`Persistent=false`) : une sortie absente fait échouer et alerte.
La marge ne garantit pas une réussite amont ; son reçu reste obligatoire.

YouTube n'a pas d'endpoint carrousel dans cette implémentation : il est explicitement ignoré
pour les images, sans réservation, secret ni appel API. Le support vidéo demeure optionnel :
configuration séparée `media_kind=video`, `factory_revision=f11d90c2191991b1b64b8c6c75e2b714bed733ee`,
répertoires de la fabrique vidéo v2 et timer 17 h 20 New York. Le lecteur conserve le contrôle
des reçus et ffprobe (H.264 silencieux, 720 × 1280, 25–45,1 s). Le timer vidéo amont étant
désactivé, aucune nouvelle vidéo n'est supposée disponible. Ne pas le réactiver implicitement.

## Créer la configuration et préparer les secrets

Depuis le checkout dédié, avec Python 3.11+, Pillow et
`pandas_market_calendars==5.4.0` déjà utilisés par la fabrique. ffprobe sert au mode vidéo optionnel.
Réutiliser son environnement validé ; ne pas mettre à jour
un environnement partagé pendant le déploiement.

```bash
umask 077
python -m scripts.social_publisher --config "$HOME/data/social-publisher-v1/config.json" init-config
```

Cette commande crée un fichier privé, sans secret, sans appel réseau. Modifier :

| Champ | Valeur attendue |
| --- | --- |
| `factory_root` | Répertoire de sortie déjà configuré dans la fabrique |
| `factory_state` | Le `state_dir` réel de la configuration de fabrique, **pas** son fichier de configuration |
| `media_kind` | `carousel` par défaut ; `video` uniquement pour le contrat vidéo optionnel |
| `factory_revision` | Révision de producteur acceptée ci-dessus |
| `state_dir` | Nouveau répertoire privé propre au publieur ; garder ce chemin lors des mises à jour |
| `authorization` | Fichier opérateur dans `~/authorizations/`, voir plus bas |
| `dry_run` | Garder `true` jusqu'à la revue des destinations et du texte |
| `platforms.<nom>.enabled` | `true` pour chaque destination choisie individuellement |
| `platforms.<nom>.account_id` | Identifiant exact de l'IG professionnel, Page, utilisateur X, chaîne YouTube ou open ID TikTok |
| `platforms.instagram.public_media_ready` | `true` seulement après ajout et contrôle du proxy HTTPS d'images par l'opérateur |

Les secrets vont **uniquement** dans `~/private/social/instagram.env`, `facebook.env`, `x.env`,
`youtube.env`, `tiktok.env`. Répertoire `0700`, fichiers réguliers appartenant à l'utilisateur,
mode exactement `0600`, sans lien symbolique ni lien physique. Saisie par éditeur local privé,
jamais en argument de commande, dans Git, une capture, le shell history ou un ticket.
Ne pas sourcer les fichiers : le parseur lit des lignes `CLE="valeur"` sans expansion shell.
Aucun fichier d'exemple contenant une valeur de secret n'est fourni dans Git.

| Destination | Noms des clés à remplir dans son fichier |
| --- | --- |
| Instagram / Facebook | `ACCESS_TOKEN` (jeton de Page), `EXPIRES_AT` si expiration connue, date ISO avec fuseau |
| X | `CLIENT_ID`, `CLIENT_SECRET` pour client confidentiel, `ACCESS_TOKEN`, `REFRESH_TOKEN`, `EXPIRES_AT` |
| YouTube | `CLIENT_ID`, `CLIENT_SECRET`, `ACCESS_TOKEN`, `REFRESH_TOKEN`, `EXPIRES_AT` |
| TikTok | `CLIENT_ID` = client key, `CLIENT_SECRET`, `ACCESS_TOKEN`, `REFRESH_TOKEN`, `EXPIRES_AT` |

X, Google et TikTok renouvellent un jeton expirant et remplacent atomiquement le même fichier
privé ; Meta exige un renouvellement opérateur. Aucune valeur de jeton, URL de session sensible,
réponse d'erreur brute ou en-tête d'authentification n'est journalisée.

## Autorisation réseau obligatoire

La tâche n'autorise pas un essai réel sans le fichier opérateur exigé par AGENTS.md / HOST.md.
Le programme refuse un fichier ailleurs que `~/authorizations/`, un fichier non `0600`, une
période expirée, un compte différent ou une requête hors périmètre. Le budget compte **chaque**
requête HTTP, y compris consultations, renouvellements, répétitions et sondages d'état, et survit
au redémarrage dans SQLite. Ne pas recréer le fichier avec un nouveau digest pour réinitialiser
artificiellement un budget. Aucun fichier d'autorisation n'est créé par le programme.

Structure à remplir et signer/déclarer par l'opérateur après revue ; ce fragment est volontairement
inexécutable et ne donne aucune autorisation :

```text
status: AUTHORIZED
purpose: social-publisher-v1
not_before: <instant ISO avec fuseau>
expires_at: <instant ISO avec fuseau>
max_requests: <budget entier, maximum 10000>
accounts: {"instagram": "<IG_ID>", "facebook": "<PAGE_ID>", ...}
allowed_requests: [{"platform": "instagram", "method": "POST",
                    "host": "graph.facebook.com", "path_prefix": "/v25.0/"}, ...]
x_zero_cost_write_access_confirmed: false
```

Le fichier réel est du JSON. N'y mettre que les destinations autorisées et les lignes nécessaires :

| Plateforme | Méthode | Hôte | Préfixe de chemin |
| --- | --- | --- | --- |
| Instagram | POST, GET (deux lignes) | `graph.facebook.com` | `/v25.0/` |
| Instagram | POST | `rupload.facebook.com` | `/ig-api-upload/` |
| Facebook | POST, GET (deux lignes) | `graph.facebook.com` | `/v25.0/` |
| Facebook | POST | `rupload.facebook.com` | `/video-upload/` |
| X | GET | `api.x.com` | `/2/users/me` |
| X | GET, POST (deux lignes) | `api.x.com` | `/2/media/upload` |
| X | POST | `api.x.com` | `/2/tweets` |
| X | POST | `api.x.com` | `/2/oauth2/token` |
| YouTube | GET | `www.googleapis.com` | `/youtube/v3/channels` |
| YouTube | POST, PUT (deux lignes) | `www.googleapis.com` | `/upload/youtube/` |
| YouTube | POST | `oauth2.googleapis.com` | `/token` |
| TikTok | GET | `open.tiktokapis.com` | `/v2/user/info/` |
| TikTok | POST | `open.tiktokapis.com` | `/v2/post/publish/` |
| TikTok | POST | `open.tiktokapis.com` | `/v2/oauth/token/` |

Les préfixes Graph couvrent les identifiants de conteneur créés à la volée ; les identifiants des
comptes cibles sont séparément fixés dans `accounts`. Les tokens ne suivent aucune redirection,
les proxys d'environnement sont ignorés et les hôtes d'upload retournés sont contrôlés.
L'autorisation du premier consentement OAuth / échange de code devra aussi nommer ses propres
endpoints et son budget : le publieur ne fait pas cet échange initial.

## Meta : Instagram professionnel et Page Facebook

1. Se connecter à [Meta for Developers](https://developers.facebook.com/), créer une application
   Business avec les cas d'usage de gestion de Page / publication Instagram ; les noms de menus
   peuvent varier. Remplir `hyprlcore.com`, les liens réels des pages confidentialité et conditions
   existantes, l'icône et le contact développeur dans la console privée. Ne pas inventer les chemins
   des pages juridiques : reprendre ceux réellement publiés.
2. Dans Facebook, créer ou choisir la Page et lui relier le compte Instagram **professionnel**
   Business/Creator. Donner au compte opérateur les droits de création de contenu. Terminer les
   éventuelles vérifications de publication de Page. Ajouter l'opérateur comme administrateur/
   développeur/testeur de l'application et accepter l'invitation.
3. Ajouter **Facebook Login for Business**, pour le compte Instagram relié à la Page.
   Enregistrer exactement `https://hyprlcore.com/auth/meta/callback` comme URI de retour HTTPS.
   Les deux adaptateurs partagent ce retour ; ne pas utiliser le flux Instagram Login à sa place.
4. Demander `pages_show_list`, `pages_read_engagement`, `pages_manage_posts`, `instagram_basic`,
   `instagram_content_publish`. Ne demander des permissions de publicité que si le mode précis
   d'attribution Business Manager l'exige et après revue ; aucun outil publicitaire n'est implémenté.
5. Effectuer le consentement opérateur, obtenir le jeton utilisateur puis le jeton de la Page et
   l'identifiant `instagram_business_account` lié. Reporter les identifiants dans la configuration,
   les jetons de Page dans les deux fichiers privés distincts. Vérifier leur expiration et les droits.
   La [collection officielle Meta](https://www.postman.com/meta/facebook/documentation/r56bjfd/facebook-api)
   décrit la récupération des jetons de Page et l'envoi binaire des Reels Facebook.
6. Conserver le mode développement pour les comptes ayant un rôle sur l'application ; vérifier
   l'accès standard effectif aux permissions dans la console. L'accès à des comptes tiers exige
   les vérifications et accès avancés applicables. Ne pas promettre un accès universel en mode dev.
7. Faire ajouter le proxy HTTPS décrit ci-dessous avant d'activer Instagram images, puis déclarer
   `public_media_ready=true`. Après autorisation écrite, essayer sur les comptes opérateur.
   Instagram crée un enfant par JPEG avec `image_url` et `is_carousel_item=true`, sans légende ;
   il crée ensuite le parent `media_type=CAROUSEL`, avec les enfants dans l'ordre et la légende,
   attend `FINISHED`, puis appelle `media_publish`. Facebook envoie chaque JPEG en multipart
   à `/{PAGE_ID}/photos` avec `published=false`, puis une seule publication `/{PAGE_ID}/feed`
   avec `message` et `attached_media[i]={"media_fbid":"..."}`. Facebook n'exige pas d'URL publique.
   Références : [publication Instagram](https://developers.facebook.com/docs/instagram-platform/instagram-api-with-facebook-login/content-publishing/)
   et [publications de Page](https://developers.facebook.com/docs/pages-api/posts/).
   Les pages Meta ont répondu 429 au lecteur documentaire pendant cette tâche ; le contrat
   est testé hors ligne, mais son acceptation par la version Graph et les comptes réels reste à vérifier.
8. Le mode vidéo facultatif conserve les flux resumable des Reels Instagram/Facebook et l'envoi
   binaire local ; ne pas envoyer un PNG à ces endpoints vidéo.

Texte de revue Meta, à adapter seulement à ce qui a été réellement déployé :

> HyprL publie sur la Page et le compte Instagram professionnel explicitement reliés par leur
> propriétaire. L'utilisateur choisit séparément chaque destination et autorise la publication de
> ses carrousels originaux de synthèse d'information. Nous utilisons la liste des Pages pour sélectionner
> la destination, les droits de lecture nécessaires pour le lien Page–Instagram, et les droits de
> publication pour envoyer des images validées, en carrousel Instagram ou publication multi-photo de Page.
> Nous n'automatisons ni abonnements, ni likes, ni commentaires.
> La révocation arrête les futurs envois ; les jetons sont conservés dans des fichiers privés
> accessibles au seul compte de service.

Ne pas annoncer un chiffrement applicatif inexistant : ici, les fichiers sont privés `0600` ;
le chiffrement du disque doit être confirmé par l'opérateur avant de le mentionner dans un audit.

## X : application et accès sans coût

1. Créer projet/application dans la [console développeur X](https://developer.x.com/).
   Vérifier **avant toute activation** le contrat réel et les coûts du compte. La page officielle
   consultée décrit désormais une facturation à l'usage : ce programme ne peut garantir l'existence
   d'un ancien palier Free. Ne pas acheter de crédits pour contourner la contrainte de la tâche.
2. Activer OAuth 2.0, application Web confidentielle, permissions lecture/écriture.
   URI exacte : `https://hyprlcore.com/auth/x/callback`. Website : `https://hyprlcore.com`.
   Utiliser PKCE S256, scopes `tweet.read tweet.write users.read media.write offline.access`.
3. Faire autoriser le compte opérateur et stocker ses jetons utilisateur. Un Bearer d'application
   seul ne permet pas de publier à sa place. Le premier appel lit `/2/users/me` et compare l'identité
   à `account_id` avant tout envoi de média.
4. Garder `enabled=false` si aucun accès gratuit en écriture n'est confirmé. Si un droit sans coût
   applicable est effectivement établi, enregistrer cette décision dans la configuration
   (`zero_cost_write_access_confirmed=true`) **et** l'autorisation
   (`x_zero_cost_write_access_confirmed=true`). Fixer `monthly_post_cap` entre 1 et 25 selon le droit
   le plus restrictif. Le programme réserve au plus un post par séance et 25 par mois, y compris
   les tentatives incertaines. Les autres outils partageant l'application consomment aussi ses limites.
5. Le protocole implémenté utilise les chemins v2 `initialize`, `/{id}/append`, `/{id}/finalize`, puis
   le statut `GET /2/media/upload?command=STATUS&media_id=...` et `POST /2/tweets`.
   Chaque JPEG du pack X passe par ce flux, puis les quatre identifiants au plus sont joints à un
   seul post. Les segments multipart font 4 Mo ; catégorie `tweet_image`, MIME `image/jpeg`.
   La vidéo optionnelle utilise `tweet_video`. Le texte n'ajoute aucun lien ; le disclaimer reste présent.
   Revoir le [guide officiel des médias](https://docs.x.com/x-api/media/quickstart/media-upload-chunked)
   lors du premier essai autorisé : l'ancien exemple officiel publié sur GitHub utilise encore
   le protocole à commandes et ne suffit pas à prouver l'accès effectif de votre application.

Déclaration d'usage X :

> Publication quotidienne de nos carrousels originaux de présentation et d'information de marché,
> uniquement sur notre compte autorisé. Aucun suivi, like, commentaire, message privé, collecte
> de données d'audience ou conseil de placement automatisé. Limites locales : une tentative par
> séance, plafond mensuel conservateur et arrêt sur quota, erreur ou incertitude de publication.

## Google Cloud et YouTube Shorts

Cette destination sert uniquement au mode vidéo optionnel. Aucun diaporama vidéo n'est fabriqué
par le publieur à partir du carrousel ; le mode images indique `youtube_requires_optional_video`.

1. Créer un projet dans [Google Cloud Console](https://console.cloud.google.com/), activer
   **YouTube Data API v3**, et choisir la chaîne opérateur destinée aux Shorts.
2. Configurer Google Auth Platform / écran de consentement : nom HyprL, domaine vérifié
   `hyprlcore.com`, liens réels de confidentialité/conditions, coordonnées privées requises.
   Ajouter l'opérateur comme utilisateur de test si l'application reste en Testing.
3. Créer un client OAuth Web ; URI exacte : `https://hyprlcore.com/auth/youtube/callback`.
   Origine Web si nécessaire : `https://hyprlcore.com`. Scopes :
   `https://www.googleapis.com/auth/youtube.upload` et
   `https://www.googleapis.com/auth/youtube.readonly` (vérification de l'identité de la chaîne).
4. Demander l'accès hors ligne (`access_type=offline`, consentement explicite initial) pour obtenir
   un refresh token. Stocker client et jetons dans le fichier privé YouTube. Comparer la chaîne
   retournée par `channels.list(mine=true)` avec `account_id` avant l'upload.
5. Garder `audit_approved=false` : ce programme impose alors `privacyStatus=private` même si
   `privacy=public` est demandé. Après validation opérateur des exigences actuelles du projet,
   un passage explicite à `audit_approved=true` permet `private`, `unlisted` ou `public`.
   Les règles d'accès ont évolué : la [référence actuelle videos.insert](https://developers.google.com/youtube/data/reference/rest/v3/videos/insert)
   indique que les projets non vérifiés ne sont plus systématiquement restreints au privé.
   Notre garde privée demeure une exigence locale de cette livraison ; ne pas la confondre avec
   une garantie universelle sur les décisions de Google.
6. La vidéo reste verticale, strictement sous 60 s ; le contrat réel de la fabrique est plus court.
   L'API reçoit `#Shorts`, `selfDeclaredMadeForKids=false` et `containsSyntheticMedia=true`.
   Ce dernier champ existe dans [la ressource videos](https://developers.google.com/youtube/data/reference/rest/v3/videos).
   L'upload est resumable, par blocs de 1 Mio ; après timeout, le programme demande l'offset de
   **la même session**. Il ne crée pas une deuxième vidéo pour réessayer.
7. Vérifier la vidéo privée dans Studio : texte, cadrage, provenance, absence de promesse financière,
   divulgation et audience. L'acquittement d'upload ne prouve pas la fin du transcodage ni la visibilité
   publique. Surveiller les quotas réels de la console sans supposer les anciennes valeurs.
   Les applications en Testing peuvent perdre leur refresh token après la durée imposée par Google ;
   préparer la vérification OAuth et le renouvellement opérateur.

Texte proposé pour vérification OAuth / audit de conformité YouTube, après déploiement effectif :

> HyprL (https://hyprlcore.com) produit des vidéos originales en français qui résument de
> l'information de marché sourcée. Nous demandons youtube.upload pour envoyer une vidéo choisie
> sur la chaîne consentante et youtube.readonly pour vérifier l'identité de cette chaîne, sans
> analyse d'audience. Le propriétaire contrôle titre, description, destination et visibilité dans
> sa configuration privée avant activation. Nos tests commencent en privé. Le système transmet
> la divulgation de contenu synthétique, ne promet aucun rendement et n'automatise aucun engagement.
> Les secrets restent dans des fichiers privés ; la révocation interrompt les publications futures.
> Nous conservons les identités de publication et digests nécessaires à l'absence de doublons.

Ne soumettre ce texte qu'avec captures fidèles. Distinguer vérification OAuth, revue de conformité
YouTube et augmentation de quota : leurs formulaires et critères ne sont pas interchangeables.
La configuration locale permet de choisir la visibilité ; pour un futur produit multi-utilisateur,
il faudra une véritable interface de choix conforme aux
[fonctionnalités minimales YouTube](https://developers.google.com/youtube/terms/required-minimum-functionality).

Script de démonstration YouTube (3–4 minutes, aucune clé visible) :

1. Présenter hyprlcore.com, les pages juridiques réelles et le compte de démonstration.
2. Montrer le consentement Google et expliquer les deux scopes. Masquer les valeurs secrètes.
3. Montrer la vidéo originale, son titre, sa description sourcée et son disclaimer ; choisir la chaîne,
   la visibilité privée et la destination YouTube seule dans une copie de configuration sans secret.
4. Après autorisation de l'essai, lancer une seule publication. Montrer l'identité acquittée et la
   vidéo privée dans Studio, `#Shorts`, audience et divulgation synthétique.
5. Rejouer la même journée : `already_posted`, aucune seconde vidéo. Désactiver la destination,
   puis montrer comment révoquer l'accès dans le compte Google.

## TikTok : limites bloquantes, configuration et audit honnête

Les [directives TikTok](https://developers.tiktok.com/docs/en/content-sharing-guidelines) excluent
l'utilitaire interne demandé et l'habillage promotionnel ajouté par l'intégration. Les carrousels
actuels portent le nom et le lien du produit sur chaque slide ; le passage aux photos ne lève
donc pas le blocage antérieur. **L'orchestrateur refuse ces carrousels vers TikTok**, même si
`enabled=true`, avec `carousel_branding_incompatible_with_tiktok_guidelines`.
Ne pas falsifier un dossier d'audit ni retirer cette garde pour obtenir un résultat vert.
Les tests du protocole sont synthétiques et ne prouvent aucune admissibilité à l'audit.

L'adaptateur photo complet est testé : `POST /v2/post/publish/content/init/`, `media_type=PHOTO`,
`post_mode=DIRECT_POST` ou `MEDIA_UPLOAD`, `source_info.source=PULL_FROM_URL`, liste ordonnée
`photo_images` et `photo_cover_index=0`. `is_aigc=true` est **à la racine**, dans les deux modes.
Il vérifie l'identité et creator info, envoie puis suit `publish_id` jusqu'au privé ou à la boîte
de réception. La [référence Photo](https://developers.tiktok.com/docs/en/content-posting-api-reference-photo-post)
décrit ce contrat ; ce n'est ni l'endpoint vidéo ni l'API Business.

Le mode privé impose `SELF_ONLY`, musique automatique désactivée et déclarations commerciales
explicites. Le brouillon transmet titre/description et le champ IA photo ; il attend
`SEND_TO_USER_INBOX`. Il ne prétend pas avoir publié : le créateur termine dans TikTok et doit
garder une visibilité privée jusqu'à validation du parcours. Aucun passage public automatique.

1. Dans [TikTok for Developers](https://developers.tiktok.com/), créer l'application, compléter nom,
   site, domaine vérifié, confidentialité et conditions existantes. Ajouter Login Kit et Content
   Posting API, configurer la cible Web avec `https://hyprlcore.com/auth/tiktok/callback`.
2. Demander `user.info.basic`, `video.publish` pour Direct Post, `video.upload` pour brouillon ;
   ces [scopes](https://developers.tiktok.com/docs/en/tiktok-api-scopes) ont des rôles distincts.
   Faire approuver les scopes et autoriser le compte cible ; conserver les secrets privés.
3. Sans audit, les essais Direct Post restent `SELF_ONLY` sur un compte privé ; TikTok limite aussi
   le nombre d'utilisateurs d'un client non audité. Ne pas promettre un passage public automatique.
   Notre adaptateur refuse la visibilité publique **même après audit** dans cette version.
4. Pour un futur produit admissible, construire d'abord une vraie prévisualisation avec nickname,
   texte modifiable, visibilité choisie sans valeur initiale et options de commentaires/disclosure.
   Pour les photos, ne pas afficher duo/collage. Consentement explicite par carrousel avant tout
   transfert ; le manifeste privé lie SHA du contenu, SHA des métadonnées, compte, nickname,
   prévisualisation, choix, confirmation musique, mode et échéance de moins de 24 h.
   Il ne remplace pas le parcours créateur exigé. Les tests vidéo historiques restent applicables
   uniquement au mode vidéo optionnel.
   Le protocole photo n'accepte que le titre et la légende déjà validés par la fabrique : une
   modification dans l'interface future devra repasser ses contrôles avant consentement et envoi.
5. Le serveur possède déjà le fichier : utiliser `PULL_FROM_URL` après vérification de propriété de
   `https://hyprlcore.com/social-media/`. Ne pas détourner `FILE_UPLOAD`, destiné aux fichiers du
   terminal de l'utilisateur, pour éviter cette vérification. Voir l'hébergement ci-dessous.
6. Contrairement à la photo, l'[Upload API vidéo](https://developers.tiktok.com/docs/en/content-posting-api-reference-upload-video)
   ne transmet pas les mêmes champs de titre/IA. Pour une vidéo optionnelle en brouillon, le
   créateur doit appliquer ces choix dans TikTok. Ne pas copier les paramètres photo dans ce flux.

Texte **à ne pas soumettre comme demande d'audit de l'outil interne actuel** :

> Notre implémentation actuelle est un utilitaire interne HyprL ; nous comprenons qu'elle n'est pas
> admissible à l'audit Direct Post destiné aux outils pour créateurs. Nous ne sollicitons pas la
> levée de restriction sur la base d'une présentation trompeuse. Nous souhaitons confirmer auprès
> du support le parcours autorisé pour nos comptes et les conditions d'un éventuel produit public.

Texte de soumission **conditionnel**, uniquement après réalisation d'un vrai produit créateur :

> HyprL propose aux créateurs un outil de production d'informations sourcées sur hyprlcore.com.
> Chaque utilisateur connecte son propre compte, voit son identité, prévisualise son carrousel original,
> modifie le texte et choisit explicitement la visibilité et les divulgations. Aucun contenu n'est
> transmis avant son consentement. Nous respectons les limites de durée retournées, les restrictions
> d'interactions et le label IA. Aucun habillage promotionnel n'est ajouté aux images exportées.
> Nous demandons l'audit de ce parcours effectivement accessible au public, sans automatisation
> d'engagement et sans promesse de rendement financier.

Script de démonstration TikTok **à réaliser après ces développements**, pas avec une maquette
présentée comme existante (4–5 minutes) :

1. Montrer le vrai produit ouvert aux créateurs, le site, les pages juridiques et le parcours Login Kit.
2. Afficher le nickname provenant de creator info et les images originales propres produites dans l'outil.
3. Éditer le titre ; ouvrir la liste de visibilité sans valeur sélectionnée ; choisir privé.
   Montrer les commentaires désactivés initialement et l'option grisée si le compte l'interdit.
4. Montrer les choix contenu commercial / marque propre / partenariat, leurs conséquences et
   l'option IA. Recueillir l'accord relatif à l'usage de musique, même sans ajout de musique.
5. Valider explicitement ; montrer le transfert depuis le domaine vérifié, l'état de traitement et
   le résultat privé. Pour le mode brouillon, montrer aussi la notification et la finition dans TikTok.
6. Révoquer l'accès et montrer le contrôle des doublons. Ne montrer aucun token ni de tiers identifiable.

## Retours OAuth : ce que le site doit ajouter

Aucune modification du site ou de nginx n'est faite par cette branche. Le propriétaire du site doit
implémenter les quatre retours exacts, **avant** de promettre que le consentement fonctionne :

| Plateforme | URI enregistrée |
| --- | --- |
| Meta, pour Instagram et Facebook | `https://hyprlcore.com/auth/meta/callback` |
| X | `https://hyprlcore.com/auth/x/callback` |
| YouTube / Google | `https://hyprlcore.com/auth/youtube/callback` |
| TikTok | `https://hyprlcore.com/auth/tiktok/callback` |

Pour chacun : session opérateur authentifiée ; départ OAuth avec état aléatoire lié à la session,
TTL court et usage unique ; validation stricte `state` et des erreurs au retour ; PKCE S256 pour X
(et les flux qui le demandent) ; échange du code **côté serveur** contre jetons ; aucune clé côté
navigateur. Supprimer les codes de l'URL par redirection immédiate vers une page neutre et exclure
ces routes des journaux d'accès/analytics. Les jetons ne doivent pas traverser une redirection,
être envoyés sur Discord ou copiés dans une ligne de commande.

Le site ne doit pas écrire dans les fichiers de l'utilisateur agent : une session opérateur autorisée
transfère les valeurs par un canal privé contrôlé puis les saisit directement dans les fichiers `0600`.
L'échange initial n'est pas implémenté dans ce CLI et reste un prérequis de déploiement.
Prévoir aussi une action de déconnexion/révocation, les retours de suppression de données exigés
par Meta et les confirmations visibles des choix de publication. Reprendre les liens juridiques
existants et documenter durée de conservation, révocation et effacement des jetons.

## HTTPS éphémère : étape nginx réservée au propriétaire

Instagram images exige maintenant ce proxy ; Facebook images reste en envoi binaire local.
`hyprl-social-media.service` écoute **uniquement 127.0.0.1:8767** et lit le spool privé du publieur.
Chaque JPEG validé reçoit une URL indépendante
`https://hyprlcore.com/social-media/<jeton-aléatoire-256-bits>/image.jpg`, TTL **deux heures**.
Le mode vidéo optionnel utilise `/video.mp4`, TTL une heure. Limites configurables dans le code :
cinq minutes à six heures. GET/HEAD et Range sont
pris en charge, sans index ni journaux d'accès. Les URLs expirées répondent 404 ; le ménage local
retire les blobs expirés. Les fichiers restent privés, liés à leur digest ; un verrou protège la
création contre le nettoyage par l'autre processus. Aucune URL de capacité n'est imprimée par le CLI.

Le propriétaire nginx doit ajouter lui-même, dans le vhost HTTPS existant de hyprlcore.com :

```nginx
location ^~ /social-media/ {
    access_log off;
    error_log /dev/null crit;
    limit_except GET HEAD { deny all; }
    proxy_pass http://127.0.0.1:8767;
    proxy_set_header Host $host;
    proxy_read_timeout 60s;
    proxy_hide_header Cache-Control;
    add_header Cache-Control "private, no-store" always;
    add_header X-Robots-Tag "noindex, nofollow, noarchive" always;
}
```

Ne jamais exposer le dossier des carrousels ni un listing nginx ; garder l'URI complète lors du proxy.
Le propriétaire vérifie puis recharge sa configuration selon ses procédures. Cette tâche n'y touche
pas. Les routes OAuth et les pages juridiques restent hors de ce proxy.

Pour la vérification TikTok future : ouvrir l'application dans TikTok for Developers, puis
**Manage URL properties**, ajouter le domaine `hyprlcore.com`, faire ajouter par le propriétaire
DNS la valeur de vérification fournie par TikTok et valider dans la console. Autre choix : propriété
de préfixe exactement `https://hyprlcore.com/social-media/`, avec la preuve proposée par la console
servie par le propriétaire du site. Ne pas fabriquer de valeur DNS ou de nom de fichier de preuve.
La [documentation de transfert](https://developers.tiktok.com/docs/en/content-posting-api-media-transfer-guide)
exige HTTPS sans redirection et une disponibilité pendant le téléchargement ; le délai possible
est d'une heure, couvert par les deux heures de nos URLs. Le JPEG conserve les dimensions du pack
photo 1080 × 1920. Aucun PNG ni ZIP n'est envoyé à TikTok. Tester l'accès avec un fichier synthétique
seulement après autorisation ; ne pas copier une URL réelle dans un ticket ou des analytics.

## Installation par la session opérateur

Après revue et intégration autorisée, garder un checkout propre, son SHA complet et le chemin de
configuration privé. Ne jamais déplacer le ledger ni supprimer les réservations pour « réparer ».
Les commandes suivantes ne déploient que les nouvelles unités marketing, après vérification de
leur absence ; elles n'écrasent aucun service existant.

```bash
publisher_revision=$(git rev-parse HEAD)
python -m scripts.social_publisher --config "$HOME/data/social-publisher-v1/config.json" \
  --expected-revision "$publisher_revision" stage-units \
  --output "$HOME/data/social-publisher-v1/units-reviewed"
systemd-analyze --user verify "$HOME/data/social-publisher-v1/units-reviewed/"*.service \
  "$HOME/data/social-publisher-v1/units-reviewed/"*.timer
python -m scripts.social_publisher --config "$HOME/data/social-publisher-v1/config.json" daily --dry-run
```

La simulation ne construit ni transport réseau, ni lecteur de secrets, ni ledger. Elle signale les
préconditions locales mais ne prouve pas la validité du consentement, du compte, du token ou de
l'autorisation réseau. Elle nécessite une sortie quotidienne terminée et une heure admissible.
Le mode `dry_run=true` de configuration gagne toujours ; `--dry-run` ne peut pas déclencher d'envoi.

L'opérateur installe les trois unités du publieur après contrôle qu'elles n'existent pas, puis
recharge le gestionnaire utilisateur et active son timer. Pour Instagram images, installer aussi
`hyprl-social-media.service` depuis le même staging et l'activer après revue du proxy ; cette
unité n'est pas nécessaire à Facebook/X. Aucun service n'est installé par la branche :

```bash
mkdir -p "$HOME/.config/systemd/user"
for unit in hyprl-social-publisher.service hyprl-social-publisher.timer hyprl-social-publisher-alert.service; do
  test ! -e "$HOME/.config/systemd/user/$unit" || exit 1
  install -m 600 "$HOME/data/social-publisher-v1/units-reviewed/$unit" "$HOME/.config/systemd/user/$unit"
done
systemctl --user daemon-reload
systemctl --user enable --now hyprl-social-publisher.timer
systemctl --user list-timers hyprl-social-publisher.timer
```

Le service est borné à 300 Mo, CPU 25 %, Nice 19, I/O idle et vingt minutes ; le service médias
séparé à 100 Mo. Aucune variable secrète ne figure dans une unité. L'OnFailure appelle uniquement
`~/bin/hyprl-notify alertes`, avec un message fixe, pour un code d'échec, timeout ou processus tué.
Une invocation manuelle ne notifie pas automatiquement ; `python -m scripts.social_publisher alert`
est un envoi réel, à lancer uniquement comme essai opérateur explicitement autorisé.

Lorsque les prérequis sont satisfaits, l'opérateur peut passer `dry_run=false` pour les seules
destinations validées et autorisées. Observer le premier passage réel : aucune publication réelle
n'a été faite pour tester cette branche.

## Incidents, reprises et arrêt

Le verrou sérialise les passages ; SQLite réserve `(plateforme, SHA-256 contenu)`, l'empreinte
de la liste ordonnée des JPEG de cette plateforme, et `(plateforme, séance)` avant le premier
appel réseau. Un changement de titre, de compte ou un redémarrage ne
permet pas de renvoyer un carrousel déjà acquitté. Le même evergreen un autre jour reste déjà
publié : cette v1 ne le répète pas pour remplir le calendrier. La réservation demeure après crash.
Garder le ledger durablement, y compris après rétention des images ou changement de branche.
Une table supplémentaire indexe les albums JPEG ; les tables et anciens reçus vidéo sont conservés.

GET et consultations POST sans effet de publication sont répétés au plus quatre fois avec
attente exponentielle ; un rejet explicite HTTP 429 peut aussi être répété avec `Retry-After`.
Un délai supérieur à trente secondes arrête l'essai. Les créations dont le résultat est ambigu
(timeout/5xx) ne sont **jamais** répétées à l'aveugle. Exception sûre : sondage et reprise par offset
sur la même session resumable YouTube. Les phases de traitement sont sondées avec un plafond.
Une attente persistante laisse une identité connue et une réservation à rapprocher.
Un échec peut laisser des conteneurs Instagram ou des photos Facebook non publiées. Le programme
ne les republie ni ne les supprime automatiquement ; la dernière identité intermédiaire reste
dans le ledger pour le rapprochement opérateur sur les outils officiels.

```bash
python -m scripts.social_publisher --config "$HOME/data/social-publisher-v1/config.json" ledger
```

Inspecter ensuite le compte concerné avec les outils officiels. Si une publication existe, rapprocher
son identifiant vérifié ; cette action n'envoie rien :

```bash
python -m scripts.social_publisher --config "$HOME/data/social-publisher-v1/config.json" \
  reconcile-posted --platform instagram --sha256 '<digest du contenu>' \
  --remote-id '<identifiant officiel vérifié>' --confirmation I_VERIFIED_REMOTE_POST
```

Si l'absence de publication est prouvée, cette v1 ne propose pas de purge automatique ni de bouton
« retenter quand même ». Faire examiner la réservation et son identité avant une réparation dédiée.
Même un refus après réservation garde la place de la séance : c'est le coût volontaire de la
prévention des doublons. Les autres plateformes continuent lorsqu'une destination échoue.

Arrêt : `systemctl --user disable --now hyprl-social-publisher.timer` puis arrêter son seul service
s'il tourne. Si le service médias a été déployé ultérieurement, arrêter aussi cette unité dédiée.
Conserver les reçus ; révoquer les tokens dans les consoles officielles si nécessaire. Aucun
follow, like, commentaire, message privé ou promesse de performance financière n'est implémenté.
