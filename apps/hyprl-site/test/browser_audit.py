"""Local-only browser and axe checks; all identities and data are synthetic."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:3817"
AXE = (ROOT / "node_modules/axe-core/axe.min.js").read_text()
ROUTES = ["/", "/confidentialite", "/mentions-legales", "/cgv", "/accessibilite", "/resiliation", "/securite", "/contact", "/merci", "/not-found", "/admin"]
checks = 0


def audit(page, label):
    global checks
    page.evaluate(AXE)
    violations = page.evaluate("""async () => (await axe.run(document, {
        runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']}
    })).violations.map(v => ({id: v.id, impact: v.impact, targets: v.nodes.map(n => n.target)}))""")
    assert not violations, (label, violations)
    checks += 1


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    external = []

    def route_request(route):
        if route.request.url.startswith(BASE + "/"):
            route.continue_()
        else:
            external.append(route.request.url)
            route.abort()

    context.route("**/*", route_request)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    for scheme in ["light", "dark"]:
        page.emulate_media(color_scheme=scheme, reduced_motion="reduce")
        for route in ROUTES:
            page.goto(BASE + route)
            page.wait_for_load_state("networkidle")
            assert page.locator("html").get_attribute("lang") == "fr"
            audit(page, scheme + route)
            assert page.evaluate("localStorage.length + sessionStorage.length") == 0
            assert page.locator(".consent-panel").count() == 0
    assert context.cookies() == [], "visitor storage must be empty"

    # Keyboard skip link, mobile menu, escape and reflow equivalent to 400% of 1280px.
    page.goto(BASE)
    page.wait_for_load_state("networkidle")
    page.keyboard.press("Tab")
    assert page.locator(".skip-link").evaluate("el => el === document.activeElement")
    page.keyboard.press("Enter")
    assert page.locator("main").evaluate("el => el === document.activeElement")
    for width in [640, 320]:
        page.set_viewport_size({"width": width, "height": 900})
        for route in ["/", "/confidentialite", "/cgv", "/accessibilite", "/admin"]:
            page.goto(BASE + route)
            page.wait_for_load_state("networkidle")
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (width, route)
        page.goto(BASE)
        page.wait_for_load_state("networkidle")
        toggle = page.get_by_role("button", name="Menu", exact=True)
        toggle.focus()
        page.keyboard.press("Enter")
        assert toggle.get_attribute("aria-expanded") == "true"
        page.keyboard.press("Escape")
        assert toggle.get_attribute("aria-expanded") == "false"
        assert toggle.evaluate("el => el === document.activeElement")

    # Native form validation, consent, admin login, individual JSON export, deletion.
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(BASE)
    page.wait_for_load_state("networkidle")
    form = page.locator("form[action='/inscription']")
    assert not form.evaluate("el => el.checkValidity()")
    page.locator("#email").fill("browser-person@example.test")
    assert not page.locator("#consent").is_checked()
    assert not form.evaluate("el => el.checkValidity()")
    page.locator("#consent").check()
    form.get_by_role("button", name="Rejoindre l’accès anticipé").click()
    page.wait_for_url(BASE + "/merci")
    page.goto(BASE + "/admin")
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Se connecter").click()
    assert page.locator("#password").get_attribute("aria-invalid") == "true"
    page.locator("#password").fill("synthetic-browser-test-password")
    page.get_by_role("button", name="Se connecter").click()
    page.get_by_role("link", name="Exporter JSON — browser-person@example.test").wait_for()
    audit(page, "authenticated admin")
    with page.expect_download() as download_info:
        page.get_by_role("link", name="Exporter JSON — browser-person@example.test").click()
    download = download_info.value
    body = json.loads(Path(download.path()).read_text())
    assert body["subscriber"]["email"] == "browser-person@example.test"
    assert "token" not in json.dumps(body)
    page.on("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Supprimer browser-person@example.test").click()
    page.get_by_text("Aucune inscription.", exact=True).wait_for()
    page.get_by_role("button", name="Se déconnecter").click()
    page.get_by_role("button", name="Se connecter").wait_for()

    # Opt-in UI is exercised from a local test-only in-memory variant.
    # The shipped file remains disabled. No optional network scripts are registered.
    page.goto(BASE)
    page.wait_for_load_state("networkidle")
    enabled_source = (ROOT / "public/consent.js").read_text().replace("var ENABLED = false;", "var ENABLED = true;")
    page.evaluate(enabled_source)
    page.evaluate("window.optionalLoads = 0; HyprLConsent.mount(document, localStorage, [{purpose: 'functional', load: () => window.optionalLoads++}])")
    panel = page.locator(".consent-panel")
    assert panel.is_visible()
    assert panel.locator("input:checked").count() == 0
    assert page.evaluate("window.optionalLoads") == 0
    audit(page, "optional consent panel")
    refuse = panel.get_by_role("button", name="Tout refuser")
    accept = panel.get_by_role("button", name="Tout accepter")
    assert refuse.get_attribute("class") == accept.get_attribute("class")
    refuse.click()
    assert page.evaluate("window.optionalLoads") == 0
    page.get_by_role("button", name="Modifier mes choix de cookies").click()
    accept.click()
    assert page.evaluate("window.optionalLoads") == 1
    page.get_by_role("button", name="Modifier mes choix de cookies").click()
    refuse.click()
    page.wait_for_load_state("networkidle")
    saved = page.evaluate("JSON.parse(localStorage.getItem('hyprl_optional_consent_v1'))")
    assert saved["purposes"] == {"functional": False, "analytics": False}
    assert not external, "attempted external request"
    assert not errors, errors
    context.close()
    browser.close()
print(f"PASS: {checks} axe page/state checks; keyboard, mobile/reflow, empty visitor storage, sign-up/admin/export/delete, disabled and opt-in consent flows; no external requests")
