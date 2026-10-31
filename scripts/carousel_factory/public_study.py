"""Price-free counts already published by the completed study; no research or store access."""

from .content import BTC_HOOK, post, slide

REVISION = "98c7189b207e857777e188318b7248e87616ddd1"
BASE = "https://github.com/Kacawaiii/HyprL/blob/" + REVISION
REPORT = BASE + "/docs/BTC_ASIA_STUDY_20261029.md"
EVIDENCE = BASE + "/docs/artifacts/btc_asia_study_result_v1.json"
EVIDENCE_SHA256 = "b7583307e2b785702a06cb935c958c3f9d150e04f8175195ad2aa36d1972064e"


def verdict():
    # These are received public aggregate counts, not a rerun on prices or a protected interval.
    friday, others, empirical = 251 / 450, 1567 / 2701, 1818 / 3151
    stamp = "Étude HyprL · agrégats publics\nCalculée le 29/10/2026 à 14:36 UTC"
    cards = [
        slide(
            BTC_HOOK,
            "des vendredis. »",
            metric="60 %",
            quote=True,
        ),
        slide(
            "Le bitcoin fixe un plus bas pendant la séance asiatique.",
            "Le vendredi, ce plus bas serait enfoncé plus tard dans la journée, 6 fois sur 10.",
            eyebrow="L'idée qui circule",
            source=stamp,
        ),
        slide(
            "Ce qu'on a mesuré, vendredis de 2018 à août 2026",
            "251 vendredis sur 450.",
            metric=f"{friday * 100:.1f} %".replace(".", ","),
            source=stamp,
            after="Pas loin des 60 % annoncés. Alors, c'est vrai ?",
        ),
        slide(
            "Sauf que tous les jours font pareil.",
            chart=[
                {"label": "Vendredi", "value": friday},
                {"label": "Autres jours", "value": others},
                {"label": "Un jour au hasard", "value": empirical},
            ],
            source=stamp,
        ),
        slide(
            "Le vendredi n'a rien de spécial.",
            "Ce plus bas tombe plus d'un jour sur deux, quel que soit le jour. Le vendredi ne bat les autres jours que 4 années sur 9.",
            source=stamp,
            flip=True,
            eyebrow="Verdict",
        ),
        slide(
            "Une stat qui claque ne prouve rien tant qu'on ne la compare pas.",
            "Chaque semaine, on vérifie une idée qui circule sur les marchés.",
            cta="La suite : lien en bio",
        ),
    ]
    value = post(
        "chance",
        "Problem-Proof",
        cards[0]["title"],
        cards,
        "Le bitcoin casse le plus bas d'Asie dans 55,8 % des vendredis observés, contre 58,0 % les autres jours. Aucun avantage du vendredi établi. "
        "Étude historique exploratoire du 1er janvier 2018 au 31 août 2026 ; année 2026 partielle, 14 journées incomplètes exclues. "
        "Intervalle de Wilson à 95 % : 51,2 à 60,3 %. Test bilatéral contre 60 % : p = 0,075 ; ne pas rejeter 60 % ne le confirme pas. "
        "Intervalle à 95 % de la différence vendredi moins autres jours, par blocs hebdomadaires : −7,1 à +2,9 points. "
        "Les intervalles de Wilson et le test binomial supposent l'indépendance ; les blocs n'excluent pas une dépendance plus longue. "
        "Cette fréquence ne démontre ni rendement, ni rentabilité après coûts, ni effet causal. Le repère « Un jour au hasard » est la fréquence empirique de tous les jours réunis : 1818 sur 3151, soit 57,7 % ; ce n'est pas un tirage simulé.",
        sources=[REPORT, EVIDENCE],
        evidence=[
            {
                "friday_k": 251,
                "friday_n": 450,
                "other_k": 1567,
                "other_n": 2701,
                "empirical_k": 1818,
                "empirical_n": 3151,
                "years_friday_beats_other_days": 4,
                "years_comparable": 9,
                "sha256": EVIDENCE_SHA256,
            }
        ],
        provenance={"public_study_revision": REVISION, "evidence_sha256": EVIDENCE_SHA256},
    )
    value["cover_action"] = "On a vérifié 450 vendredis"
    value["hashtags"] = ["#bitcoin", "#bourse", "#investir"]
    return value
