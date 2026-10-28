"""Product-presentation copy approved by the operator on 2026-10-28."""

DISCLAIMER = "Information, pas un conseil en investissement"
INVITE = "discord.gg/BCgmuk6kXE"
SITE = "hyprlcore.com"
HASHTAGS = ["#HyprL", "#Marchés", "#BriefMarché", "#Transparence"]
SCRIPT_REVISION = "v2"
SCRIPT_SHA256 = "5023a8ca5ffd4bacb9f52e53b34bf26aeb34960faac09fc857caa1727ad1880e"

EVERGREENS = (
    (
        "HyprL en 30 secondes",
        (
            "Chaque matin, les marchés produisent des centaines d'informations.",
            "HyprL fait le tri : ce qui s'est passé, ce qui compte aujourd'hui, et pourquoi.",
            "Un brief clair avant l'ouverture de New York.",
            "Les publications officielles et les rendez-vous macro, au moment où ils tombent.",
            "Et des résultats que chacun peut vérifier.",
            "HyprL. L'information de marché, vérifiable. Accès anticipé gratuit, lien en bio.",
        ),
    ),
    (
        "Votre matinée de marché en deux minutes",
        (
            "Avant l'ouverture, vous n'avez pas le temps de tout lire.",
            "Le brief HyprL résume la séance précédente, les mouvements de la nuit et l'agenda du jour.",
            "Chaque information a sa source et son heure.",
            "Deux minutes de lecture, et vous savez où regarder.",
            "À découvrir gratuitement pendant le lancement. Lien en bio.",
        ),
    ),
    (
        "Ne ratez plus un événement",
        (
            "Un dépôt officiel tombe sur une action que vous suivez. Les chiffres de l'inflation sortent. Une entreprise publie ses résultats.",
            "HyprL vous prévient, avec la source, là où vous travaillez déjà : Discord, Slack, Telegram.",
            "Vous choisissez vos actions et vos sujets.",
            "De l'information, jamais de conseil. Lien en bio.",
        ),
    ),
    (
        "La transparence comme règle",
        (
            "Beaucoup montrent leurs réussites, rarement leurs erreurs.",
            "Chez HyprL, nos analyses sont horodatées et publiées avant l'ouverture.",
            "Les résultats sont affichés en entier, bons ou mauvais.",
            "Et nous indiquons même la part que le hasard peut expliquer.",
            "Vous jugez sur pièces. Lien en bio.",
        ),
    ),
    (
        "L'accès anticipé est ouvert",
        (
            "HyprL ouvre ses portes.",
            "Le brief du jour, le radar des événements, le calendrier et l'espace membre, gratuitement pendant le lancement.",
            "Ensuite, trois formules, à partir de 19 euros par mois.",
            "Les premiers inscrits seront prévenus en premier.",
            "Rejoignez-nous : lien en bio.",
        ),
    ),
)


def evergreen(index):
    title, captions = EVERGREENS[index % len(EVERGREENS)]
    return {
        "kind": "evergreen",
        "title": title,
        "captions": list(captions),
        "edition": "version présentation produit",
        "notes": "HyprL : le brief de marché, les événements et leurs sources. "
        "Accès anticipé gratuit pendant le lancement.",
        "provenance": {
            "kind": "operator_approved_product_script",
            "revision": SCRIPT_REVISION,
            "sha256": SCRIPT_SHA256,
        },
    }


def blocked_verdict():
    return {
        "kind": "verdict",
        "title": "BTC le vendredi : verdict non vérifiable",
        "provenance": {
            "kind": "research_blockage",
            "study_status": "BLOCKED",
            "reason": "expired_research_grant_and_daily_only_local_candles",
        },
        "captions": [
            "Le bas asiatique du bitcoin serait pris 60 % des vendredis. Chance ou pas ?",
            "La règle testée : Asie de 2 h à 11 h, heure de Paris. Puis un prix strictement inférieur, avant minuit.",
            "Les archives disponibles sont journalières. Elles ne disent pas quand le creux asiatique a été cassé.",
            "La collecte intrajournalière n'est pas autorisée actuellement pour cette étude. Aucun chiffre ne sera inventé.",
            "Verdict : non vérifiable à ce stade. Il faudra comparer le vendredi aux autres jours, année par année.",
        ],
        "notes": "Étude BLOQUÉE : aucun taux ni intervalle de confiance observé. "
        "60 % n'est ni confirmé ni réfuté. Aucune conclusion sur le hasard. "
        "Asie [02:00,11:00), Londres [09:00,18:00), New York [14:00,23:00), "
        "Europe/Paris, changements d'heure inclus. Événement : low après 11:00 "
        "strictement inférieur au low asiatique, jusqu'à minuit local. "
        "Le plus bas journalier en Asie est une autre hypothèse. "
        "Aucune donnée BTC/ETH du holdout septembre–novembre 2026 utilisée.",
    }
