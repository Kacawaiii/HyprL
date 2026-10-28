"""Operator-approved evergreen copy (2026-10-27), with no invented observations."""

DISCLAIMER = "Information, pas un conseil en investissement"
INVITE = "discord.gg/BCgmuk6kXE"
HASHTAGS = ["#HyprL", "#ChanceOuPas", "#Transparence", "#Bourse", "#IA"]

# Keep these sentences verbatim. The teaser's wording is not the study's event:
# a later sweep of the Asia low does NOT mean the daily low occurred in Asia.
EVERGREENS = (
    ("Deux IA, la même info", (
        "J'ai mis deux IA en duel sur la bourse. Mêmes infos, même heure, chaque matin.",
        "Elles s'appellent HypR-A1 et HypR-E1. À 9 h 05 heure de New York, elles reçoivent exactement les mêmes news.",
        "Leurs décisions sont scellées avant l'ouverture : impossible de tricher après coup.",
        "Le soir, on compare. Et on dit aussi quand c'est juste de la chance.",
        "Les résultats sont publics chaque jour. Lien en bio.",
    )),
    ("Pourquoi tu peux me croire", (
        "La plupart des comptes de trading montrent leurs gains après coup.",
        "Nous, avant l'ouverture, on publie une empreinte numérique de chaque décision.",
        "Si on modifiait une seule virgule, l'empreinte ne correspondrait plus.",
        "Le soir, on révèle les décisions. Tout le monde peut vérifier.",
        "Transparence totale. Lien en bio.",
    )),
    ("10 trades gagnants ne prouvent rien", (
        "Quelqu'un gagne 10 trades d'affilée. Génie ?",
        "Pas forcément. Avec assez de gens qui lancent une pièce, quelqu'un fera 10 piles d'affilée.",
        "Pour prouver qu'on gagne 55 % du temps, il faut environ 380 trades.",
        "On affiche donc, pour chaque IA, la probabilité que son résultat soit juste du hasard.",
        "Personne d'autre ne le fait. Lien en bio.",
    )),
    ("Mes IA ont perdu", (
        "On a fait rejouer à nos IA juillet et août, sans qu'elles connaissent la suite.",
        "Résultat : sur les grosses actions, au jour le jour, elles ne battent pas le hasard.",
        "Pourquoi je te le dis ? Parce que tout le monde cache ses échecs.",
        "Maintenant, on les teste là où une IA peut vraiment aider : lire les rapports officiels que personne ne lit.",
        "La suite en direct. Lien en bio.",
    )),
    ("Chance ou pas ?", (
        "Cette affirmation tourne en ce moment : le bitcoin ferait son plus bas de la journée pendant la séance asiatique, 60 % des vendredis.",
        "60 %, ça a l'air énorme. Mais sur 52 vendredis, une pièce peut faire ça.",
        "On va le tester honnêtement, sur des années qu'elle n'a jamais vues.",
        "Chance ou pas ? Verdict dans la prochaine vidéo.",
        "Abonne-toi pour la suite.",
    )),
)


def evergreen(index):
    title, captions = EVERGREENS[index % len(EVERGREENS)]
    notes = (
        "Script éditorial approuvé par l'opérateur. Les exemples de pièce et de taille "
        "d'échantillon sont pédagogiques ; aucune promesse de rendement. Les prévisions "
        "ne sont pas des trades. Un indicateur de hasard n'est pas une probabilité que "
        "l'IA ait ou non du talent. Les affirmations éditoriales générales ne constituent "
        "pas un audit de tous les autres comptes."
    )
    if index % 5 == 4:
        notes += (
            " Précision : le test porte sur une cassure du bas asiatique APRÈS 11 h, "
            "pas sur l'heure du plus bas journalier. Le teaser reprend le texte approuvé."
        )
    return {"kind": "evergreen", "title": title, "captions": list(captions), "notes": notes}


def blocked_verdict():
    return {
        "kind": "verdict", "title": "BTC le vendredi : verdict non vérifiable",
        "provenance": {"kind": "research_blockage", "study_status": "BLOCKED",
                       "reason": "expired_research_grant_and_daily_only_local_candles"},
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
