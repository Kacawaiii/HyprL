# Registre interne des traitements — modèle v1

Dossier de travail interne, hors du répertoire `public/` et non servi par HTTP.
Ce modèle ne contient aucune coordonnée personnelle. L'opérateur tient la version
renseignée, les contrats et preuves dans son dossier privé, jamais dans Git.
Mise à jour : 2026-10-31. Revue avant lancement et à chaque changement.

## R-01 — Liste d'accès anticipé

- Responsable / représentant : [IDENTITÉ, STATUT, SIREN, ADRESSE À RENSEIGNER HORS GIT].
- Contact droits et sécurité : `NOTIFY_EMAIL` en configuration privée ; publié à
  l'exécution sur `/contact`. DPO : [ÉVALUER L'OBLIGATION / CONTACT SI DÉSIGNÉ].
- Personnes : visiteurs volontaires inscrits à l'accès anticipé, aucun achat.
- Finalité : contacter les personnes sur l'accès anticipé à HyprL. Aucune publicité
  ciblée, revente, recommandation personnalisée, profilage publicitaire. Les contrôles automatiques du formulaire servent uniquement à la prévention des abus.
- Base : consentement, article 6.1.a RGPD. Case obligatoire non précochée ; texte
  et version exacts conservés, retrait à tout moment. Les anciennes versions du
  texte de consentement sont immuables.
- Données : adresse e-mail, canaux et actions facultatifs, UUID, création UTC,
  version/texte/date UTC du consentement, empreintes des jetons de confirmation et désinscription, statut pending/confirmed et date de confirmation.
  Aucun mot de passe d'inscrit ; ni IP ni jeton en clair dans la liste.
- Double opt-in : unique e-mail après preuve de travail locale, délai minimal, jeton signé lié à l’IP et Origin exact. Confirmation explicite POST, lien 48 h, aucune relance. Lettre facultative seulement si demandée séparément. Anciennes inscriptions sans preuve : non confirmées, même purge 7 jours.
- Notification éditeur : bilan des seules inscriptions confirmées de dates UTC précédentes, au plus un par 24 h, réservation durable avant SMTP et aucun renvoi incertain.
- Destinataires : éditeur seul, accès administrateur restreint. Hébergeur : Hetzner
  Online GmbH, Allemagne/UE. Contrat article 28 : [DPA ET LOCALISATION À ARCHIVER].
- Messagerie : [PRESTATAIRE UE À CHOISIR], DPA, sous-traitants ultérieurs, lieu de
  stockage, support et accès à contrôler avant activation. Envoi bloqué tant que
  `SMTP_EU_APPROVED` n'est pas explicitement activé. Publier son identité dans la
  politique avant activation ; cette variable est une attestation opérateur,
  elle ne vérifie pas les clauses contractuelles ou la localisation du fournisseur.
- Transferts : aucun hors UE autorisé. Ne pas connecter une messagerie ou un widget
  incompatible. Les canaux choisis restent des préférences ; ils ne déclenchent
  aucune transmission vers Discord, Slack, Telegram, Teams, API ou webhook.
- Durée : demandes non confirmées effacées après 7 jours ; confirmées, 24 mois calendaires depuis la création, sans prolongation par doublon ;
  effacement plus tôt au retrait ou sur demande. Purge au démarrage, chaque heure,
  avant les requêtes API et désinscription. Échéance calculée en UTC, 29 février
  ramené au 28 février si nécessaire. Une panne d'écriture bloque l'accès API.
- Effacement : lien individuel confirmé en POST ou suppression admin ; suppression
  du fichier actif atomique, sans journal des données supprimées. Contrôler aussi
  notifications, copies et boîtes de messagerie. Les sauvegardes ne doivent pas
  dépasser l'échéance individuelle ; purge obligatoire avant toute restauration.
  Aucun archivage légal de cette liste gratuite n'est présumé nécessaire.
- Droits : accès, rectification par l'éditeur (effacement/réinscription si besoin),
  effacement, limitation, portabilité JSON individuelle après vérification
  proportionnée, retrait, réclamation CNIL. Délai un mois, prolongation motivée
  de deux mois si nécessaire. Pour une limitation, suspendre tout envoi jusqu'à
  résolution ; consigner la demande dans le dossier privé, sans réutiliser la liste.
- Portabilité : admin → personne concernée → Exporter JSON ; vérifier identité et
  destinataire, remettre de façon privée, supprimer la copie de travail après
  remise. L'export exclut les jetons et leurs empreintes, contient uniquement la
  personne demandée et les dates UTC. Ne jamais diffuser la liste entière.
- Mesures : TLS/HSTS au point d'entrée, CSP locale, absence de scripts externes,
  cookie admin HttpOnly/Secure/SameSite=Strict, sessions en mémoire 30 min/12 h,
  limitation de connexion, contrôle d'origine, stockage fichier 0600/répertoire
  0700 à créer à l'exploitation, écritures atomiques et sérialisées, mot de passe
  long hors Git, accès au serveur limité. Un seul processus écrivain par fichier.
- Vérifications d'exploitation : vérifier l'hébergement réel en Allemagne, les
  permissions des répertoires existants, la désactivation des logs HTTP contenant
  IP/jetons à tous les niveaux, les boîtes e-mail et la politique de sauvegarde.
  Ces mesures externes ne sont pas prouvées par les tests applicatifs.

## R-02 — Sécurité d'accès

- Finalité/base : prévention des abus, intérêt légitime article 6.1.f ; analyse
  de nécessité : compteurs et empreintes à clé persistants pour empêcher les contournements par redémarrage et les renvois à une même adresse ; ni IP ni adresse en clair dans ce fichier distinct.
- Données : empreintes à clé des IP et adresses, empreintes des jetons consommés, compteurs et dates ; IP en mémoire pour la connexion administrateur, empreinte de session admin et dates
  techniques ; aucun suivi intersite, pas de cookie visiteur.
- Durées : compteurs IP une heure, jetons consommés dix minutes, empreinte de suppression par adresse au plus 24 mois (même après effacement du dossier personnel, pour empêcher un nouveau message). Connexion 15 min, nettoyage chaque minute ; sessions 30 min d'inactivité ou 12 h maximum, disparition au redémarrage.
- Quotas : 5 tentatives vérifiées/IP/heure, 30 globales/heure ; 30 confirmations/24 h. Jetons : 20/IP/heure et 300 globales/heure. Un seul écrivain, fichiers 0600, clé HMAC stable hors Git, réservations atomiques et synchronisées sur disque avant SMTP.
- Sous-traitant : Hetzner, UE ; autres journaux à inventorier et désactiver s'ils
  capturent des requêtes, des jetons ou des adresses. Aucun log de corps HTTP.

## Futurs changements

Abonnements, facturation, traceurs, alertes API et intégrations nécessitent une
nouvelle entrée ou révision avant activation : finalités/bases/durées distinctes,
DPA, destinataires, droits, registre et politique publique. Le module de consentement
facultatif reste désactivé ; aucune conservation locale du visiteur actuellement.
Si activé après revue : finalités séparées non cochées, refus aussi accessible que
l'acceptation, choix six mois, nouvel accord ensuite (avant treize mois), changement
à tout moment, aucun chargement facultatif avant accord, retrait puis rechargement.

## Références pour la validation opérateur

RGPD articles 5, 6, 12–22, 28, 30, 32–34 ; recommandations CNIL relatives aux
traceurs. Les références sont à revalider par l'opérateur avant publication ;
aucune consultation réseau d'une source officielle n'a été effectuée sur ce serveur.
