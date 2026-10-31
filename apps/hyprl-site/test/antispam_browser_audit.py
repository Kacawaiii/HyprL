"""Synthetic browser acceptance. Allow only the isolated loopback fixture server."""
import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:3818'
AXE = (ROOT / 'node_modules/axe-core/axe.min.js').read_text()
checks = 0
external, errors = [], []


def intercept(route):
    if route.request.url.startswith(BASE + '/'):
        route.continue_()
    else:
        external.append(route.request.url)
        route.abort()


def audit(page):
    global checks
    page.evaluate(AXE)
    result = page.evaluate("""async () => (await axe.run(document, {
        runOnly: {type:'tag', values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}
    })).violations.map(v => ({id:v.id, targets:v.nodes.map(n => n.target)}))""")
    assert not result, result
    checks += 1


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    nojs = browser.new_context(java_script_enabled=False)
    nojs.route('**/*', intercept)
    page = nojs.new_page()
    page.goto(BASE)
    page.wait_for_load_state('networkidle')
    assert page.get_by_role('button', name='Rejoindre l’accès anticipé').is_disabled()
    assert page.locator('noscript').get_by_role('link', name='Discord').is_visible()
    assert len(nojs.request.get(BASE + '/__qa/mail').json()) == 0
    nojs.close()

    context = browser.new_context(viewport={'width': 1280, 'height': 900})
    context.route('**/*', intercept)
    page = context.new_page()
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(BASE + '/#acces')
    page.wait_for_load_state('networkidle')
    assert "script-src 'self'" in context.request.get(BASE).headers['content-security-policy']
    assert not page.locator('#newsletter').is_checked()
    audit(page)
    page.locator('#email').fill('browser-antispam@example.test')
    page.locator('#consent').check()
    page.locator('#newsletter').check()
    page.locator('#actions').fill('I wanted to know your price')
    page.get_by_role('button', name='Rejoindre l’accès anticipé').click()
    expect(page.locator('#signup-status')).to_contain_text('Les phrases', timeout=30000)
    assert page.locator('#signup-status').evaluate('el => el === document.activeElement')
    assert len(context.request.get(BASE + '/__qa/mail').json()) == 0
    audit(page)
    page.locator('#actions').fill('AAPL, Bank of America')
    page.get_by_role('button', name='Rejoindre l’accès anticipé').click()
    page.wait_for_url(BASE + '/merci')
    page.wait_for_load_state('networkidle')
    audit(page)
    mails = context.request.get(BASE + '/__qa/mail').json()
    assert len(mails) == 1
    confirmation = mails[0]
    link = re.search(r'http://127\.0\.0\.1:3818/confirmation\?token=[\w-]+', confirmation['text']).group(0)
    page.goto(link)
    page.wait_for_load_state('networkidle')
    audit(page)
    page.get_by_role('button', name='Confirmer mon inscription à HyprL', exact=True).click()
    page.wait_for_load_state('networkidle')
    assert page.get_by_role('heading', name='Votre inscription à HyprL est confirmée.').is_visible()
    assert len(context.request.get(BASE + '/__qa/mail').json()) == 1
    audit(page)
    page.goto(link)
    assert page.get_by_role('heading', name='Lien invalide ou expiré').is_visible()
    withdrawal = context.request.get(BASE + '/__qa/newsletter-withdrawal').json()['url']
    page.goto(withdrawal)
    page.get_by_role('button', name='Me désabonner de la lettre').click()
    assert page.get_by_role('heading', name='Vous êtes désabonné de la lettre.').is_visible()
    audit(page)
    erase = confirmation['list']['unsubscribe']['url']
    page.goto(erase)
    page.get_by_role('button', name='Confirmer la désinscription').click()
    page.wait_for_load_state('networkidle')
    assert page.get_by_role('heading', name='Vous êtes désinscrit.').is_visible()
    audit(page)
    # A second form POST for an erased address cannot trigger another message.
    page.goto(BASE + '/#acces')
    page.locator('#email').fill('browser-antispam@example.test')
    page.locator('#consent').check()
    page.get_by_role('button', name='Rejoindre l’accès anticipé').click()
    page.wait_for_url(BASE + '/merci')
    assert len(context.request.get(BASE + '/__qa/mail').json()) == 1
    assert context.cookies() == []
    assert page.evaluate('localStorage.length + sessionStorage.length') == 0
    assert not external, external
    assert not errors, errors
    print(json.dumps({'axe_checks': checks, 'nojs_discord': True, 'confirmation_emails': 1,
        'invalid_actions_emails': 0, 'operator_emails': 0, 'repeat_after_erasure_emails': 0,
        'external_requests': 0, 'page_errors': 0, 'visitor_storage': 0}))
    browser.close()
