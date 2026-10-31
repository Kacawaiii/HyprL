"""Local browser proof, synthetic content and recipients. External traffic is blocked."""
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:3818'
OUT = ROOT / 'docs/growth/screenshots'
OUT.mkdir(parents=True, exist_ok=True)
AXE = (ROOT / 'node_modules/axe-core/axe.min.js').read_text()
checks = 0


def audit(page):
    global checks
    page.evaluate(AXE)
    errors = page.evaluate("""async () => (await axe.run(document, {
      runOnly: {type:'tag', values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}
    })).violations.map(v => ({id:v.id, targets:v.nodes.map(n => n.target)}))""")
    assert not errors, errors
    checks += 1


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={'width': 1280, 'height': 960})
    external, errors = [], []

    def route_request(route):
        if route.request.url.startswith(BASE + '/'):
            route.continue_()
        else:
            external.append(route.request.url)
            route.abort()

    context.route('**/*', route_request)
    page = context.new_page()
    page.on('pageerror', lambda error: errors.append(str(error)))
    for scheme in ['light', 'dark']:
        page.emulate_media(color_scheme=scheme, reduced_motion='reduce')
        for route, slug in [('/calendrier-resultats-us', 'calendar-synthetic'), ('/chance-ou-pas', 'claims-synthetic'), ('/comprendre-depot-sec', 'sec-guide'), ('/parrainage', 'referrals'), ('/chance-ou-pas/btc-vendredi-plus-bas-asie', 'btc-study-public')]:
            page.goto(BASE + route)
            page.wait_for_load_state('networkidle')
            assert page.locator('h1').count() == 1
            assert page.locator('script[type="application/ld+json"]').count() == 1
            schema = json.loads(page.locator('script[type="application/ld+json"]').text_content())
            assert schema['inLanguage'] == 'fr-FR'
            assert page.locator('link[rel="canonical"]').get_attribute('href') == BASE + route
            audit(page)
            if scheme == 'light' or slug in ['calendar-synthetic', 'btc-study-public']:
                page.screenshot(path=str(OUT / f'{slug}-{scheme}.png'), full_page=True)
            if slug == 'btc-study-public':
                assert 'Aucun avantage vendredi établi' in page.locator('main').inner_text()
                assert '251/450' in page.locator('main').inner_text()
    page.goto(BASE + '/chance-ou-pas')
    page.locator('main a[href^="/chance-ou-pas/"]').filter(has_text='Affirmation synthétique').click()
    page.wait_for_load_state('networkidle')
    audit(page)
    page.screenshot(path=str(OUT / 'claim-article-synthetic-dark.png'), full_page=True)
    assert context.cookies() == []
    assert page.evaluate('localStorage.length + sessionStorage.length') == 0

    page.emulate_media(color_scheme='light')
    page.goto(BASE + '/#acces')
    page.wait_for_load_state('networkidle')
    newsletter = page.get_by_role('checkbox', name='Recevoir la lettre hebdomadaire gratuite', exact=True)
    assert not newsletter.is_checked()
    assert newsletter.get_attribute('required') is None
    audit(page)
    page.locator('form').screenshot(path=str(OUT / 'newsletter-unticked.png'))
    page.locator('#email').fill('reader@example.test')
    page.locator('#consent').check()
    newsletter.check()
    page.get_by_role('button', name='Rejoindre l’accès anticipé', exact=True).click()
    page.wait_for_url('**/merci')
    page.wait_for_load_state('networkidle')
    mails = context.request.get(BASE + '/__qa/mail').json()
    confirmation = next(m for m in mails if m['subject'].startswith('Confirmez'))
    link = re.search(r'http://127\.0\.0\.1:3818/lettre/confirmation\?token=[\w-]+', confirmation['text']).group(0)
    page.goto(link)
    page.wait_for_load_state('networkidle')
    audit(page)
    page.get_by_role('button', name='Confirmer mon inscription à la lettre', exact=True).click()
    page.wait_for_load_state('networkidle')
    assert page.get_by_role('heading', name='Votre inscription à la lettre est confirmée.').is_visible()
    page.screenshot(path=str(OUT / 'newsletter-confirmed.png'), full_page=True)
    unsubscription = re.search(r'http://127\.0\.0\.1:3818/lettre/desabonnement\?token=[\w-]+', confirmation['text']).group(0)
    page.goto(unsubscription)
    page.wait_for_load_state('networkidle')
    page.get_by_role('button', name='Me désabonner de la lettre', exact=True).click()
    page.wait_for_load_state('networkidle')
    assert page.get_by_role('heading', name='Vous êtes désabonné de la lettre.').is_visible()
    audit(page)

    page.set_viewport_size({'width': 320, 'height': 800})
    page.goto(BASE + '/calendrier-resultats-us')
    page.wait_for_load_state('networkidle')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.get_by_role('button', name='Menu', exact=True).click()
    assert page.locator('#site-nav').is_visible()
    page.keyboard.press('Escape')
    audit(page)
    page.screenshot(path=str(OUT / 'calendar-synthetic-mobile.png'), full_page=True)
    page.goto(BASE + '/chance-ou-pas/btc-vendredi-plus-bas-asie')
    page.wait_for_load_state('networkidle')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    audit(page)
    page.screenshot(path=str(OUT / 'btc-study-public-mobile.png'), full_page=True)
    assert context.cookies() == []
    assert page.evaluate('localStorage.length + sessionStorage.length') == 0
    assert not external, external
    assert not errors, errors
    print(json.dumps({'axe_checks': checks, 'external_requests': 0, 'page_errors': 0, 'screenshots': len(list(OUT.glob('*.png')))}))
    browser.close()
