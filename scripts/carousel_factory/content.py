"""French editorial templates; observed excerpts are supplied by the adapters."""

DISCLAIMER = "Information, pas un conseil en investissement"
SITE = "hyprlcore.com"
INVITE = "discord.gg/BCgmuk6kXE"
EDITION = "carrousels v1"


def slide(title, body="", *, source=None, metric=None, chart=None, cta=None):
    return {
        "title": title,
        "body": body,
        "source": source,
        "metric": metric,
        "chart": chart,
        "cta": cta,
    }


def post(kind, framework, title, slides, context, *, sources=(), evidence=(), provenance=None):
    return {
        "kind": kind,
        "framework": framework,
        "title": title,
        "slides": slides,
        "caption_hook": context,
        "sources": sorted(set(sources)),
        "source_evidence": list(evidence),
        "provenance": provenance,
        "hashtags": ["#bourse", "#actions", "#investir", "#wallstreet"],
    }


def evergreen(index, *, proof=None, excerpt=None, agenda=None):
    """The four requested products. Missing real examples stay explicitly unavailable."""
    if index == 0:
        return post(
            "product-a",
            "Demo Walkthrough",
            "HyprL en 5 slides",
            [
                slide("HyprL en 5 slides", "Votre information de marché, avec ses sources."),
                slide(
                    "Le brief du matin",
                    "Les faits qui comptent, réunis avant l'ouverture de New York.",
                ),
                slide(
                    "Le radar, puis vos alertes",
                    "Une publication officielle rejoint le radar. Vos actions suivies permettent de cibler les alertes.",
                ),
                slide(
                    "Le calendrier",
                    "Les publications macro et les résultats US à venir. Un horaire inconnu reste indiqué comme tel.",
                ),
                slide(
                    "Des résultats vérifiables",
                    "HypR-A1 et HypR-E1 : empreinte, révélation et résultats publiés.",
                    cta="Découvrez HyprL sur hyprlcore.com",
                ),
            ],
            "Votre veille de marché tient dans un même parcours. Voici le brief, le radar, les alertes, le calendrier et les preuves.",
        )
    if index == 1:
        detail = proof or {}
        stamp = detail.get("source")
        return post(
            "product-b",
            "Problem-Proof",
            "Comment vérifier nos analyses",
            [
                slide(
                    "Comment vérifier nos analyses", "Une décision peut se contrôler sur pièces."
                ),
                slide(
                    "L'empreinte avant l'ouverture",
                    detail.get(
                        "commit_text",
                        "Le registre engage le contenu avant la séance. Exemple public indisponible dans l'export courant.",
                    ),
                    source=detail.get("commit_source") or stamp,
                ),
                slide(
                    "La révélation après la clôture",
                    detail.get(
                        "reveal_text",
                        "Le contenu doit ensuite être révélé avec sa date. Aucun exemple n'est fabriqué.",
                    ),
                    source=stamp,
                ),
                slide(
                    "Recalculer l'empreinte",
                    "Téléchargez le fichier révélé, calculez son SHA-256, puis comparez-le à l'empreinte du registre. Une égalité vérifie l'intégrité du contenu ; elle ne prouve pas sa qualité.",
                    source=stamp,
                ),
                slide(
                    "La preuve se consulte",
                    "Les dates, le fichier et le registre permettent de refaire cette vérification.",
                    cta="Consultez les preuves sur hyprlcore.com/membre",
                ),
            ],
            "Avant de juger un résultat, vérifiez quand la décision a été engagée. Le fichier et le registre publics permettent la comparaison.",
            sources=detail.get("urls", ()),
            evidence=detail.get("evidence", ()),
            provenance=detail.get("provenance"),
        )
    if index == 2:
        return post(
            "product-c",
            "Rant Callout",
            "Une capture de gains ne prouve rien",
            [
                slide(
                    "Une capture de gains ne prouve rien",
                    "Un résultat isolé laisse trop de questions ouvertes.",
                ),
                slide(
                    "La sélection des réussites",
                    "Une capture montre une opération choisie. Les autres décisions peuvent changer le bilan.",
                ),
                slide(
                    "Les absents du tableau",
                    "Les comptes qui échouent deviennent moins visibles. Comparer seulement les survivants déforme le jugement.",
                ),
                slide(
                    "Le hasard compte aussi",
                    "Une série gagnante peut arriver par chance. Il faut une règle, un échantillon complet et une comparaison adaptée.",
                ),
                slide(
                    "Gagner est possible",
                    "Le prouver demande des décisions publiées avant le résultat.",
                ),
                slide(
                    "Regarder la série complète",
                    "C'est la règle du banc d'essai HyprL. Les résultats restent des observations, avec leurs limites.",
                    cta="Vérifiez la méthode sur hyprlcore.com",
                ),
            ],
            "Un gain affiché raconte une opération. Pour juger une méthode, regardez aussi les décisions manquantes et la place du hasard.",
        )
    if index == 3:
        item = excerpt or {}
        return post(
            "product-d",
            "Demo Walkthrough",
            "Votre matinée de marché en 2 minutes",
            [
                slide(
                    "Votre matinée de marché en 2 minutes",
                    "Un parcours de lecture du brief, sans promesse de résultat.",
                ),
                slide(
                    "Commencer par les faits",
                    "Lisez le résumé, puis sa source et son heure de publication.",
                ),
                slide(
                    item.get("title", "Un extrait du brief"),
                    item.get(
                        "body",
                        "Extrait réel indisponible dans l'export courant. Le prochain brief fournira cet exemple.",
                    ),
                    source=item.get("source"),
                ),
                slide(
                    "Regarder le prochain rendez-vous",
                    (agenda or {}).get(
                        "body",
                        "Le calendrier indique la date et l'heure lorsqu'elle est confirmée.",
                    ),
                    source=(agenda or {}).get("source"),
                ),
                slide(
                    "Votre brief chaque matin",
                    "Les informations gardent leurs dates et leurs liens d'origine.",
                    cta="Recevez le brief gratuit : lien en bio",
                ),
            ],
            "Deux minutes pour retrouver le fait, sa source et le prochain rendez-vous. L'extrait porte sa vraie date.",
            sources=list(item.get("urls", ())) + list((agenda or {}).get("urls", ())),
            evidence=list(item.get("evidence", ())) + list((agenda or {}).get("evidence", ())),
            provenance=item.get("provenance"),
        )
    raise ValueError("unknown product carousel")
