"""Real reader/UI with private snapshot and offline synthetic session boundary.

No production login or source request. Observed screenshots stay outside Git.
"""

import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:3829"
ROOT = Path(__file__).resolve().parents[1]
snapshot = json.loads(Path(os.environ["MEMBER_QA_SNAPSHOT"]).read_text())
assert snapshot["provenance"] == "observed" and snapshot["schema_version"] == 2
AXE = (ROOT / "node_modules/axe-core/axe.min.js").read_text()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 1100}, locale="fr-FR")
    external, errors = [], []

    def intercept(route):
        if route.request.url.startswith(BASE + "/"):
            route.continue_()
        else:
            external.append(route.request.url)
            route.abort()

    context.route("**/*", intercept)
    page = context.new_page()
    page.on("pageerror", lambda e: errors.append(str(e)))
    response = context.request.get(BASE + "/auth/discord/start", max_redirects=0)
    state = parse_qs(urlsplit(response.headers["location"]).query)["state"][0]
    page.goto(BASE + "/auth/discord/callback?code=premium&state=" + state)
    page.wait_for_url(BASE + "/membre")
    page.wait_for_load_state("networkidle")
    page.goto(BASE + "/membre/duel", wait_until="networkidle")
    assert "Aperçu de test" not in page.locator("#notice").inner_text()
    panels = 0
    for i, series in enumerate(snapshot["series"]):
        page.locator("#variant").select_option(str(i))
        if not series["n"]:
            assert page.locator("canvas").count() == 0
            page.get_by_role("heading", name="En cours de constitution").wait_for()
            continue
        assert page.locator("canvas").count() == 2
        values = page.locator("table").filter(
            has=page.locator("caption", has_text="Valeurs observées")
        )
        assert values.locator("tbody tr").count() == series["n"]
        panels += 1
    observed = page.locator("table").filter(
        has=page.locator("caption", has_text="Prévisions effectivement notées")
    )
    assert observed.locator("tbody tr").count() == len(snapshot["forecasts"])
    page.locator("#variant").select_option(
        str(next(i for i, s in enumerate(snapshot["series"]) if s["n"]))
    )
    page.evaluate(AXE)
    assert (
        page.evaluate(
            "async () => (await axe.run(document, {runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})).violations.length"
        )
        == 0
    )
    if os.environ.get("MEMBER_QA_SCREENSHOT_DIR"):
        out = Path(os.environ["MEMBER_QA_SCREENSHOT_DIR"]).resolve()
        assert ROOT not in out.parents and out != ROOT
        out.mkdir(mode=0o700, parents=True, exist_ok=True)
        page.screenshot(path=str(out / "duel-observed.png"))
    for proof in snapshot["reveals"]:
        page.goto(BASE + "/membre/preuves?id=" + proof["id"], wait_until="networkidle")
        page.get_by_text(proof["sha256"], exact=True).first.wait_for()
        assert (
            page.get_by_role(
                "link", name="Ouvrir le fichier au commit de révélation"
            ).get_attribute("href")
            == proof["url"]
        )
    page.goto(BASE + "/membre/calendrier", wait_until="networkidle")
    assert "heure inconnue" in page.locator("tbody").inner_text()
    assert "23:59" not in page.locator("tbody").inner_text()
    page.goto(BASE + "/membre/screener", wait_until="networkidle")
    if not snapshot["screener"]:
        page.get_by_role("heading", name="En cours de constitution").wait_for()
        assert page.locator("table").count() == 0
    assert not external and not errors
    context.close()
    browser.close()
print(
    json.dumps(
        {
            "observed_forecasts": len(snapshot["forecasts"]),
            "panels_with_points": panels,
            "proofs": len(snapshot["reveals"]),
            "external_requests": 0,
            "page_errors": 0,
        }
    )
)
