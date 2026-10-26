# Procédure interne — violation de données v1

Modèle d'exploitation, non publié par le serveur. Responsable d'incident :
[OPÉRATEUR À DÉSIGNER DANS LE DOSSIER PRIVÉ]. Suppléant : [À DÉSIGNER].
Ne stocker dans Git ni coordonnées, ni preuves brutes, ni adresses d'inscrits.

1. Dès connaissance : ouvrir le registre privé d'incident et noter l'heure ISO
   8601 UTC de prise de connaissance, les faits et la personne responsable. Le
   délai de 72 heures court dès cette connaissance, week-ends compris.
2. Contenir : couper les accès compromis et l'envoi de messages ; révoquer ou
   renouveler les secrets, isoler une copie d'investigation dans un espace privé
   à accès limité. Ne pas détruire les preuves ; ne pas les publier. L'opérateur
   décide des actions sur le service, contacte l'hébergeur et le prestataire UE.
3. Évaluer immédiatement : nature de la violation (confidentialité, intégrité,
   disponibilité), catégories et nombres approximatifs de personnes et données,
   durée, conséquences possibles, risque pour les droits et libertés, mesures
   déjà prises et prévues. Documenter même une violation peu risquée.
4. Notifier la CNIL sans retard injustifié et si possible dans les 72 heures,
   sauf si la violation est peu susceptible d'engendrer un risque pour les droits
   et libertés. Documenter cette exception. La notification précise les faits,
   catégories/nombres approximatifs, contact responsable/DPO, conséquences,
   mesures et remédiation. Si incomplet, notifier les informations disponibles
   puis compléter par étapes sans retard. Tout retard au-delà de 72 h est motivé.
5. Si risque élevé : informer aussi les personnes concernées sans retard
   injustifié, en langage clair (nature, conséquences, mesures, contact utile),
   sauf exception légale documentée. La seule notification CNIL ne suffit pas.
6. Remédier : corriger la cause, vérifier permissions, secrets, purge et copies ;
   contrôler la restauration sur données privées avec purge avant réouverture.
   L'opérateur valide le retour en service. Consigner les dates, décisions,
   notifications et actions dans le registre privé, puis revoir les mesures.

Notification : téléservice officiel de la CNIL
<https://www.cnil.fr/fr/notifier-une-violation-de-donnees-personnelles> (l'opérateur
vérifie le canal en vigueur). RGPD articles 33 et 34. Aucune adresse personnelle
ou preuve brute dans les tickets publics. Conservation des éléments d'incident :
[durée proportionnée à définir par l'opérateur], accès et suppression contrôlés.
