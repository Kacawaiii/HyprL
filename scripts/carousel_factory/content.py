"""French editorial templates; observed excerpts are supplied by the adapters."""

DISCLAIMER = "Information, pas un conseil en investissement"
SITE = "hyprlcore.com"
INVITE = "discord.gg/BCgmuk6kXE"
EDITION = "carrousels v2"


def slide(title, body="", *, source=None, metric=None, chart=None, cta=None, flip=False):
    return {
        "title": title,
        "body": body,
        "source": source,
        "metric": metric,
        "chart": chart,
        "cta": cta,
        "flip": flip,
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
            "Vos actions. Les événements qui les concernent.",
            [
                slide(
                    "Vos actions. Les événements qui les concernent.",
                    "HyprL réunit le brief, les publications officielles et votre calendrier.",
                ),
                slide(
                    "Avant New York, les faits sont réunis.",
                    "Le brief du matin garde la source et l'heure de chaque information.",
                ),
                slide(
                    "Une publication touche une action suivie.",
                    "Retrouvez le document officiel dans le radar, avec sa date et son lien.",
                ),
                slide(
                    "La prochaine date est déjà au calendrier.",
                    "Résultats US, Fed, inflation, emploi. Les horaires confirmés sont distingués des dates seules.",
                    flip=True,
                ),
                slide(
                    "Retrouvez les sources en quelques gestes.",
                    "Brief, radar, calendrier et preuves dans votre espace HyprL.",
                    cta="Découvrez HyprL",
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
            "Une analyse retouchée, ça se vérifie.",
            [
                slide(
                    "Une analyse retouchée, ça se vérifie.",
                    "Une empreinte avant la séance. Le fichier exact après la clôture.",
                ),
                slide(
                    "L'analyse est engagée avant la séance.",
                    detail.get(
                        "commit_text",
                        "Le registre engage le contenu avant la séance. Exemple public indisponible dans l'export courant.",
                    ),
                    source=detail.get("commit_source") or stamp,
                ),
                slide(
                    "Le fichier est révélé après la clôture.",
                    detail.get(
                        "reveal_text",
                        "Le contenu doit ensuite être révélé avec sa date. Aucun exemple n'est fabriqué.",
                    ),
                    source=stamp,
                ),
                slide(
                    "Même fichier, même empreinte.",
                    "Recalculez l'empreinte du fichier révélé. Elle doit correspondre au registre. Cela vérifie le contenu, pas la qualité de l'analyse.",
                    source=stamp,
                    flip=True,
                ),
                slide(
                    "Vérifiez sur pièces.",
                    "Le fichier, les dates et l'empreinte complète sont accessibles depuis les preuves HyprL.",
                    cta="Consultez les preuves",
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
                    "On voit le gain choisi. Où sont les autres décisions ?",
                ),
                slide(
                    "Le reste de la série manque.",
                    "Une opération choisie peut cacher des pertes. Il faut le bilan complet.",
                ),
                slide(
                    "Les comptes perdants disparaissent du fil.",
                    "Comparer seulement ceux qui restent visibles déforme le bilan.",
                ),
                slide(
                    "Une série gagnante peut arriver par chance.",
                    "La méthode doit être comparée à des journées semblables, avec toutes ses décisions.",
                ),
                slide(
                    "Gagner est possible",
                    "Le prouver demande des décisions publiées avant le résultat.",
                    flip=True,
                ),
                slide(
                    "Regardez la série complète.",
                    "HyprL publie les décisions du banc d'essai et leurs résultats, même défavorables.",
                    cta="Vérifiez la méthode",
                ),
            ],
            "Un gain affiché raconte une opération. Pour juger une méthode, regardez aussi les décisions manquantes et la place du hasard.",
        )
    if index == 3:
        item = excerpt or {}
        return post(
            "product-d",
            "Demo Walkthrough",
            "Avant New York, retrouvez les faits sourcés.",
            [
                slide(
                    "Avant New York, retrouvez les faits sourcés.",
                    "Un extrait réel du brief. Sa source. Puis le prochain rendez-vous.",
                ),
                slide(
                    "Le résumé mène au document d'origine.",
                    "Chaque fait garde son lien et son heure. Vous pouvez remonter à la publication.",
                ),
                slide(
                    item.get("title", "Un extrait du brief")
                    + (" · traduit" if item.get("translated") else ""),
                    item.get(
                        "body",
                        "Extrait réel indisponible dans l'export courant. Le prochain brief fournira cet exemple.",
                    ),
                    source=item.get("source"),
                ),
                slide(
                    "Gardez la prochaine date en vue.",
                    (agenda or {}).get(
                        "body",
                        "Le calendrier indique la date et l'heure lorsqu'elle est confirmée.",
                    ),
                    source=(agenda or {}).get("source"),
                    flip=True,
                ),
                slide(
                    "Le prochain brief vous attend.",
                    "Retrouvez les faits et leurs sources dans l'accès anticipé HyprL.",
                    cta="Le brief : lien en bio",
                ),
            ],
            "Le fait, sa source et le prochain rendez-vous dans un même brief. L'extrait porte sa vraie date ; ce carrousel présente le produit.",
            sources=list(item.get("urls", ())) + list((agenda or {}).get("urls", ())),
            evidence=list(item.get("evidence", ())) + list((agenda or {}).get("evidence", ())),
            provenance=item.get("provenance"),
        )
    raise ValueError("unknown product carousel")
