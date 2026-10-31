"""Admin triage on isolated synthetic state; abort every non-loopback request."""
import json
import os
import re
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:3819'
AXE = (ROOT / 'node_modules/axe-core/axe.min.js').read_text()
checks, external, errors = 0, [], []


def audit(page):
    global checks
    page.evaluate(AXE)
    violations = page.evaluate("""async () => (await axe.run(document, {
      runOnly: {type: 'tag', values: ['wcag2a','wcag2aa','wcag21a','wcag21aa']}
    })).violations.map(v => ({id:v.id, targets:v.nodes.map(n => n.target)}))""")
    assert not violations, violations
    checks += 1


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={'width': 1280, 'height': 900})

    def intercept(route):
        if route.request.url.startswith(BASE + '/'):
            route.continue_()
        else:
            external.append(route.request.url)
            route.abort()

    context.route('**/*', intercept)
    page = context.new_page()
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(BASE + '/admin')
    page.wait_for_load_state('networkidle')
    assert context.request.get(BASE + '/api/admin/subscribers?tab=suspect').status == 401
    page.locator('#password').fill('synthetic-triage-browser-password')
    page.get_by_role('button', name='Se connecter', exact=True).click()
    expect(page.locator('#rows')).to_contain_text('confirmed@example.test')
    expect(page.locator('#rows')).not_to_contain_text('pending@example.test')
    expect(page.locator('#tab-suspect')).to_have_text('Bots / suspects (2)')
    for scheme in ['light', 'dark']:
        page.emulate_media(color_scheme=scheme)
        for key, address in [('confirmed', 'confirmed@example.test'), ('pending', 'pending@example.test'), ('suspect', 'review@example.test')]:
            page.locator('#tab-' + key).click()
            expect(page.locator('#rows')).to_contain_text(address)
            expect(page.locator('#tab-' + key)).to_have_attribute('aria-pressed', 'true')
            audit(page)
    # A slower previous tab response must never replace the newly selected tab.
    held = []
    pattern = re.compile(r'.*/api/admin/subscribers\?tab=pending$')
    page.route(pattern, lambda route: held.append(route))
    page.locator('#tab-pending').click()
    page.wait_for_timeout(100)
    assert len(held) == 1
    page.locator('#tab-confirmed').click()
    expect(page.locator('#rows')).to_contain_text('confirmed@example.test')
    held.pop().continue_()
    page.wait_for_timeout(200)
    expect(page.locator('#rows')).not_to_contain_text('pending@example.test')
    page.unroute(pattern)
    page.locator('#tab-suspect').click()
    expect(page.locator('#rows')).to_contain_text('Phrase ou contenu non conforme')
    expect(page.locator('#rows')).to_contain_text('I wanted to know your price')
    expect(page.locator('#rows')).to_contain_text('<script>synthetic</script>')
    assert page.locator('#rows script').count() == 0
    assert page.locator('#rows a').count() == 0, 'no suspect export'
    assert context.request.get(BASE + '/__qa/mail').json() == []
    screenshot = os.environ.get('TRIAGE_SCREENSHOT')
    if screenshot:
        page.screenshot(path=screenshot, full_page=True)
    # Keyboard buttons remain native, usable in order and at narrow widths.
    for width in [390, 320]:
        page.set_viewport_size({'width': width, 'height': 900})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.set_viewport_size({'width': 1280, 'height': 900})
    page.locator('#tab-pending').focus()
    page.keyboard.press('Enter')
    expect(page.locator('#rows')).to_contain_text('pending@example.test')
    page.locator('#tab-suspect').click()
    page.get_by_role('button', name='C’est un vrai — review@example.test', exact=True).click()
    expect(page.locator('#tab-pending')).to_have_attribute('aria-pressed', 'true')
    expect(page.locator('#rows')).to_contain_text('review@example.test')
    expect(page.locator('#rows')).to_contain_text('Confirmation envoyée.')
    expect(page.locator('#tab-pending')).to_have_text('En attente (2)')
    expect(page.locator('#tab-suspect')).to_have_text('Bots / suspects (1)')
    audit(page)
    mails = context.request.get(BASE + '/__qa/mail').json()
    assert len(mails) == 1
    assert mails[0]['to'] == 'review@example.test'
    page.locator('#tab-suspect').click()
    page.on('dialog', lambda dialog: dialog.accept())
    page.get_by_role('button', name='Supprimer delete@example.test', exact=True).click()
    expect(page.locator('#tab-suspect')).to_have_text('Bots / suspects (0)')
    expect(page.locator('#rows')).to_contain_text('Aucune inscription.')
    with page.expect_download() as download_info:
        page.get_by_role('link', name='Exporter les Vrais (CSV)', exact=True).click()
    csv = Path(download_info.value.path()).read_text()
    assert 'confirmed@example.test' in csv
    assert 'pending@example.test' not in csv and 'review@example.test' not in csv and 'delete@example.test' not in csv
    confirmation = re.search(r'http://127\.0\.0\.1:3819/confirmation\?token=[\w-]+', mails[0]['text']).group(0)
    page.goto(confirmation)
    page.get_by_role('button', name='Confirmer mon inscription à HyprL', exact=True).click()
    page.goto(BASE + '/admin')
    page.wait_for_load_state('networkidle')
    expect(page.locator('#tab-confirmed')).to_have_text('Vrais (2)')
    expect(page.locator('#rows')).to_contain_text('review@example.test')
    assert len(context.request.get(BASE + '/__qa/mail').json()) == 1
    assert page.evaluate('localStorage.length + sessionStorage.length') == 0
    assert not external, external
    assert not errors, errors
    context.close()
    browser.close()
print(json.dumps({'axe_checks': checks, 'tabs_counts_csv': True, 'import_move_confirm_delete': True,
                  'synthetic_confirmation_emails': 1, 'external_requests': 0, 'page_errors': 0}))
