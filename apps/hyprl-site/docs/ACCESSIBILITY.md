# Auto-audit d'accessibilité — 2026-10-26

Périmètre : accueil, confidentialité, mentions légales, projet de CGV, déclaration,
résiliation future, sécurité, contact, remerciement, page 404, administration,
inscription et module facultatif de consentement. Référentiels visés : WCAG 2.1 AA
et RGAA 4.1.2. Évaluation interne limitée ; aucun taux RGAA ni certification.
La déclaration publique annonce une conformité partielle provisoire.

## Méthode et corrections

| Sujet / critères indicatifs | Vérification et résultat de la passe |
| --- | --- |
| Structure et régions (WCAG 1.3.1, 2.4.1 / RGAA 9, 12) | HTML sémantique, un titre principal par état, navigation nommée, titres hiérarchisés, tableaux à en-têtes et légendes. Lien d'évitement vers un `main` focalisable. Attributs dupliqués supprimés. |
| Langue et noms accessibles (3.1.1, 4.1.2 / RGAA 8) | `lang=fr`, liens français équivalents `hreflang=fr` sur les pages publiques statiques ; aucun contenu anglais inexistant annoncé. Icônes décoratives ignorées. |
| Contrastes texte (1.4.3 / RGAA 3) | Calcul WCAG sRGB : texte clair 17,01:1 ; secondaire clair 6,05:1 ; accent clair sur blanc 6,88:1 ; texte sombre 15,18:1 ; secondaire sombre sur surface 6,49:1 ; accent sombre 7,56:1. Les palettes visent au moins 4,5:1 pour les textes ordinaires. |
| Composants (1.4.11 / RGAA 3) | Bordures de champs/boutons renforcées : clair sur blanc 4,57:1 ; sombre sur surface 4,48:1, au-dessus de 3:1. Focus visible conservé. |
| Clavier et focus (2.1.1, 2.1.2, 2.4.7 / RGAA 12) | Lien d'évitement, menu mobile par Entrée, fermeture Échap et retour du focus au bouton ; contrôle d'export et suppression nommés par personne. Focus déplacé vers une commande après connexion. Aucun piège clavier ajouté. |
| Formulaires/erreurs (3.3.1, 3.3.2 / RGAA 11) | Labels et légendes, champs requis natifs, aides reliées par `aria-describedby`, consentement non coché, erreur de connexion annoncée et `aria-invalid`. Les erreurs HTML du formulaire `/inscription` conservent une page lisible ; les clients `/api/subscribe` reçoivent les détails RFC 9457. |
| Redistribution (1.4.10 / RGAA 10) | Texte long redistribuable ; contrôles à 640 px et 320 px, cette dernière largeur représente la redistribution attendue à 400 % d'une fenêtre de 1280 px. Tableaux conservés dans une région défilable nommée, focalisable. |
| Mouvement (2.2, 2.3 / RGAA 13) | Absence d'animations et de transitions ; pas de lecture automatique, flash ou carrousel. |
| Choix de traceurs facultatifs | Module livré désactivé. Essai d'une variante locale activée : cases non cochées, refus/acceptation de présentation identique, réglages accessibles, aucun chargement avant accord, retrait conservé puis rechargement. |

Le script `test/browser_audit.py` exécute axe-core avec les règles WCAG 2 A/AA et
2.1 A/AA en Chromium, modes clair/sombre, puis vérifie les parcours clavier,
mobile, inscription, connexion, export et suppression avec des données fictives.
Résultat : 24 contrôles de pages/états axe sans violation détectée, parcours clavier,
redistribution, inscription/admin/export/suppression et consentement réussis.
Il bloque les requêtes externes. La version activée du consentement est injectée
uniquement en mémoire pour le test ; le fichier distribué reste désactivé.
Le rapport privé de livraison contient le résultat et la commande exacte.

## Limites et suites

Ce contrôle ne parcourt pas la totalité des 106 critères RGAA ni tous leurs tests.
L'absence de violation axe n'établit pas une conformité complète. Les lecteurs
d'écran (NVDA, VoiceOver, JAWS), plusieurs navigateurs, le zoom réel à 200/400 %,
les modes forcés/contrastes utilisateurs et les cas non couverts restent à évaluer.
L'administration exige JavaScript. Aucun contenu audio/vidéo n'est présent ; les
critères correspondants n'ont pas été simulés. Aucune dérogation n'est invoquée.

Avant les ventes : audit manuel complet, tests avec technologies d'assistance,
traitement des écarts, puis mise à jour de la déclaration et du plan d'amélioration.
Responsable : [ÉDITEUR À RENSEIGNER]. Signalements par la page `/contact`.
