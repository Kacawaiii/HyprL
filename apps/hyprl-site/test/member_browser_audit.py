"""Member UI offline audit; explicit synthetic banner in every data screenshot."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:3827"
EMPTY = "http://127.0.0.1:3828"
OUT = ROOT / "docs/member/screenshots"
OUT.mkdir(parents=True, exist_ok=True)
AXE = (ROOT / "node_modules/axe-core/axe.min.js").read_text()
ROUTES = ["", "/brief", "/radar", "/calendrier", "/duel", "/chance", "/screener", "/chance-ou-pas", "/alertes", "/compte"]
checks = 0


def audit(page, label):
    global checks
    page.evaluate(AXE)
    violations = page.evaluate("""async () => (await axe.run(document, {runOnly: {type: 'tag', values: ['wcag2a','wcag2aa','wcag21a','wcag21aa']}})).violations.map(v=>({id:v.id,targets:v.nodes.map(n=>n.target)}))""")
    assert not violations, (label, violations)
    checks += 1


def login(context, page, tier="premium", base=BASE):
    # Offline Discord boundary double; callback still exercises the real Strict
    # cookie relay, PKCE/state verification and session creation.
    response = context.request.get(base + "/auth/discord/start", max_redirects=0)
    from urllib.parse import urlsplit, parse_qs
    state = parse_qs(urlsplit(response.headers["location"]).query)["state"][0]
    page.goto(base + "/auth/discord/callback?code=" + tier + "&state=" + state)
    page.wait_for_url(base + "/membre")
    page.wait_for_load_state("networkidle")
    page.get_by_text("Un point de repère pour votre journée.", exact=True).wait_for()


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 1100}, locale="fr-FR")
    external, errors, csp, limited = [], [], [], []
    def intercept(route):
        if any(route.request.url.startswith(b + "/") for b in [BASE, EMPTY]):
            route.continue_()
        else:
            external.append(route.request.url)
            route.abort()
    context.route("**/*", intercept)
    page = context.new_page()
    pending = set()
    page.on("response", lambda r: limited.append(r.url) if r.status == 429 else None)
    page.on("request", lambda r: pending.add(r.url))
    page.on("requestfinished", lambda r: pending.discard(r.url))
    page.on("requestfailed", lambda r: pending.discard(r.url))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
    page.on("console", lambda m: csp.append(m.text) if "Content Security Policy" in m.text else None)
    page.goto(BASE + "/membre/connexion")
    page.wait_for_load_state("networkidle")
    audit(page, "login")
    login(context, page)
    for scheme in ["light", "dark"]:
        page.emulate_media(color_scheme=scheme)
        for route in ROUTES:
            page.goto(BASE + "/membre" + route, wait_until="networkidle")
            page.wait_for_load_state("networkidle")
            assert "Chargement" not in page.locator("#page").inner_text()
            assert "Service indisponible" not in page.locator("#notice").inner_text(), (route, errors, csp)
            audit(page, scheme + route)
            if route in ["/duel", "/chance"]:
                assert page.locator("canvas").count() >= 1
                assert page.locator("#notice").inner_text().startswith("Aperçu de test")
            if route == "/duel":
                page.screenshot(path=str(OUT / ("duel-synthetic-" + scheme + ".png")), full_page=False)
                page.locator("#variant").select_option("1")
                page.get_by_role("button", name="Changer de thème").click()
                assert page.locator("canvas").count() == 2
            if route == "/chance" and scheme == "light":
                page.screenshot(path=str(OUT / "luck-synthetic-light.png"))
    page.goto(BASE + "/membre/radar")
    page.wait_for_load_state("networkidle")
    page.locator("#symbols").fill("DEMO")
    with page.expect_navigation(wait_until="networkidle"):
        page.get_by_role("button", name="Enregistrer la liste").click()
    assert page.get_by_text("Dépôt synthétique réservé aux tests.", exact=True).count() == 2
    page.screenshot(path=str(OUT / "radar-synthetic-dark.png"))
    page.goto(BASE + "/membre/duel")
    page.wait_for_load_state("networkidle")
    page.get_by_role("link", name="Vérifier ").first.click()
    page.wait_for_load_state("networkidle")
    page.get_by_text("Empreinte recalculée : correspondance confirmée.", exact=True).wait_for()
    audit(page, "proof")
    page.goto(BASE + "/membre/alertes")
    page.wait_for_load_state("networkidle")
    page.locator("#channel").select_option("webhook")
    page.locator("#endpoint").fill("https://example.com/synthetic-receiver")
    page.locator("#signing-key").fill("synthetic-signing-key-for-browser-qa")
    page.get_by_role("button", name="Ajouter la destination").click()
    page.get_by_role("button", name="Mettre en pause ").wait_for()
    assert page.locator("#endpoint").input_value() == ""
    assert "synthetic-receiver" not in page.locator("body").inner_text()
    page.get_by_role("button", name="Mettre en pause ").click()
    page.get_by_text("En pause", exact=True).wait_for()
    audit(page, "alert destination")
    page.get_by_role("button", name="Reprendre ").click()
    page.get_by_role("button", name="Supprimer ").click()
    page.get_by_text("Ajoutez votre premier canal", exact=False).wait_for()
    # Use a second independent synthetic member for the next request batch;
    # retain production per-member budgets rather than relaxing rate limits.
    page.get_by_role("button", name="Se déconnecter").click()
    page.wait_for_url(BASE + "/membre/connexion")
    login(context, page, "premium-mobile")
    page.goto(BASE + "/membre/radar", wait_until="networkidle")
    assert page.locator("#symbols").input_value() == "", "watchlist leaked across browser sessions"
    # Mobile reflow, including data tables and chart resize.
    for width in [640, 320]:
        page.set_viewport_size({"width": width, "height": 900})
        for route in ROUTES:
            print(f"Reflow {width}: {route or '/'}", flush=True)
            page.goto(BASE + "/membre" + route, wait_until="networkidle")
            try:
                page.wait_for_load_state("networkidle")
            except Exception as error:
                raise AssertionError((width, route, sorted(pending), errors, csp)) from error
            assert "Service indisponible" not in page.locator("#notice").inner_text(), (width, route)
            assert "Chargement" not in page.locator("#page").inner_text(), (width, route)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (width, route)
            if width == 320 and route == "":
                audit(page, "mobile")
        if width == 320:
            page.goto(BASE + "/membre/radar")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(OUT / "radar-synthetic-mobile.png"))
    page.set_viewport_size({"width": 1440, "height": 1100})
    page.goto(BASE + "/membre")
    page.wait_for_load_state("networkidle")
    page.keyboard.press("Tab")
    assert page.locator(".skip-link").evaluate("el=>el===document.activeElement")
    page.keyboard.press("Enter")
    assert page.locator("main").evaluate("el=>el===document.activeElement")
    assert page.evaluate("localStorage.length+sessionStorage.length") == 0
    page.get_by_role("button", name="Se déconnecter").click()
    page.wait_for_url(BASE + "/membre/connexion")
    # Lower tier cannot reach Premium data through the browser.
    login(context, page, "early")
    page.goto(BASE + "/membre/duel")
    page.wait_for_load_state("networkidle")
    page.get_by_text("Cet espace est compris dans une formule supérieure", exact=False).wait_for()
    assert page.locator("canvas").count() == 0
    audit(page, "locked early")
    # Empty export is the real production presentation until data exists.
    login(context, page, base=EMPTY)
    page.goto(EMPTY + "/membre/duel")
    page.wait_for_load_state("networkidle")
    page.get_by_role("heading", name="En cours de constitution").wait_for()
    assert page.locator("canvas").count() == 0
    audit(page, "empty export")
    page.screenshot(path=str(OUT / "duel-empty-dark.png"))
    assert not limited, limited
    assert not errors, errors
    assert not csp, csp
    assert not external, external
    context.close()
    browser.close()
print(json.dumps({"axe_checks": checks, "console_errors": 0, "csp_violations": 0, "external_requests": 0, "screenshots": len(list(OUT.glob('*.png')))}))
