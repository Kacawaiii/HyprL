# HyprL Go-Live Checklist (1 Soirée)

**Objectif :** Vendre les 30 premiers accès Beta en toute sécurité.

**Durée estimée :** 2-3 heures
**Date cible :** _________

---

## 🚀 GO-LIVE NOW (Quick Start)

**Run smoke test first:**

```bash
./scripts/ops/go_live_smoke.sh
```

**Then execute these 5 steps in order:**

### 1. Create Stripe Payment Links

```bash
# Via Stripe Dashboard:
# → Products → Create Product "HyprL Beta Early Access" (€19/mo)
# → Create Payment Link
# → Copy link: https://buy.stripe.com/...

# Edit apps/landing/index.html line 210
# Replace: STRIPE_LINK_BETA
# With: https://buy.stripe.com/...
```

See detailed instructions: [Phase 1](#phase-1--stripe-products--payment-links-15-min)

### 2. Generate First Track Record

```bash
# Source credentials (local only, never commit)
source .env.broker.alpaca

# Take snapshot
python scripts/ops/alpaca_track_record_snapshot.py \
  --paper \
  --out-dir docs/reports/track_record

# Generate report
python scripts/ops/make_track_record_report.py \
  --in-dir docs/reports/track_record \
  --out-dir docs/reports/track_record

# Verify
ls -lh docs/reports/track_record/TRACK_RECORD_latest.md
cat docs/reports/track_record/track_record_latest.json | jq .
```

See detailed instructions: [docs/TRACK_RECORD_OPS.md](TRACK_RECORD_OPS.md#first-real-run-minimal---no-secrets-committed)

### 3. Deploy Landing Page (Netlify)

```bash
cd apps/landing
netlify deploy --prod

# Output: https://hyprl-landing.netlify.app
```

**Verification:**
- Visit landing page
- Test waitlist form submission
- Verify redirect to `/thanks.html`
- Check Netlify Dashboard → Forms

### 4. Deploy Track Record Portal (Streamlit Cloud)

**Prerequisites:**
- Commit track record artifacts to GitHub
- Configure GitHub Secrets (see [docs/OPS.md](OPS.md))

```bash
# Commit track record
git add docs/reports/track_record/TRACK_RECORD_latest.md
git add docs/reports/track_record/track_record_latest.json
git commit -m "Add initial track record"
git push
```

**Deploy:**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. New app → Select repository: `Kacawaiii/HyprL`
3. **Main file path:** `apps/track_record/streamlit_app.py`
4. Deploy

**Note:** Streamlit Community Cloud installs dependencies from `requirements.txt` located in the same directory as the main app file (`apps/track_record/requirements.txt`).

**Verification:**
- Visit `https://[your-app].streamlit.app`
- Test without key (public view)
- Test with key (after creating entitlement)

### 5. Test End-to-End Entitlement Flow

```bash
# 1. Create test Stripe subscription (use test card 4242 4242 4242 4242)

# 2. Export customers CSV from Stripe Dashboard
# → Customers → Export

# 3. Import to entitlements DB
python scripts/ops/stripe_entitlements.py import \
  --csv ~/Downloads/stripe_customers.csv

# 4. Export allowlist
python scripts/ops/stripe_entitlements.py export

# Output:
# ✓ Exported N active keys to apps/track_record/entitlements_allowlist.txt
# ✓ Audit log saved to docs/reports/entitlements/ENTITLEMENTS_YYYY-MM-DD.json

# 5. Get access key
python scripts/ops/stripe_entitlements.py list --status active

# 6. Test portal access
# Visit: https://[your-app].streamlit.app?key=<ACCESS_KEY>
```

See detailed instructions: [Phase 4](#phase-4--test-end-to-end-30-min)

---

**✅ Production Checklist:**

- [ ] Smoke test passed (`./scripts/ops/go_live_smoke.sh`)
- [ ] Stripe payment link replaced in `apps/landing/index.html`
- [ ] Track record generated (no NaN values)
- [ ] Landing deployed on Netlify
- [ ] Portal deployed on Streamlit Cloud
- [ ] GitHub Secrets configured (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`)
- [ ] End-to-end entitlement flow tested
- [ ] Daily GitHub Action enabled (`.github/workflows/track-record-daily.yml`)

**Post-launch:**
- Monitor Netlify Forms for waitlist signups
- Monitor Stripe Dashboard for payments
- Run `python scripts/ops/stripe_entitlements.py sync` weekly
- Check track record updates daily via GitHub Actions

---

## Phase 1 : Stripe Products + Payment Links (15 min)

### 1.1 Créer le Product Beta

**Via Stripe Dashboard :**
1. Aller sur [dashboard.stripe.com](https://dashboard.stripe.com)
2. Products → Create product
3. Remplir :
   ```
   Name: HyprL Beta Early Access
   Description: Live track record dashboard + Discord community + Weekly updates
   Pricing: €19.00 EUR
   Billing period: Monthly
   ```
4. **Copier le Price ID** (commence par `price_...`)

**Via Stripe CLI (alternatif) :**
```bash
# Créer product
stripe products create \
  --name "HyprL Beta Early Access" \
  --description "Live track record dashboard + Discord community + Weekly updates"

# Créer price (remplacer PRODUCT_ID)
stripe prices create \
  --product PRODUCT_ID \
  --currency eur \
  --unit-amount 1900 \
  --recurring[interval]=month

# Output: price_1ABC... → COPIER CE ID
```

### 1.2 Créer Payment Link

**Via Stripe Dashboard :**
1. Payment links → Create payment link
2. Sélectionner le product "HyprL Beta Early Access"
3. Options :
   - ✅ Collect customer email
   - ✅ Allow promotion codes
   - ❌ Require billing address (optionnel)
4. Success URL : `https://hyprl.netlify.app/thanks.html`
5. **Copier le Payment Link** (https://buy.stripe.com/...)

**Via CLI (alternatif) :**
```bash
stripe payment_links create \
  --line-items[0][price]=price_1ABC... \
  --line-items[0][quantity]=1 \
  --after_completion[type]=hosted_confirmation \
  --after_completion[hosted_confirmation][custom_message]="Vous recevrez votre clé d'accès par email dans 48h."

# Output: https://buy.stripe.com/test_... → COPIER CE LIEN
```

### 1.3 Test Checkout (Happy Path)

```bash
# 1. Aller sur le payment link
open https://buy.stripe.com/test_...

# 2. Remplir avec carte test
Email: [email protected]
Card: 4242 4242 4242 4242
Expiry: 12/34
CVC: 123

# 3. Valider → doit rediriger vers /thanks.html

# 4. Vérifier dans Stripe Dashboard
# Customers → Voir [email protected]
# Subscriptions → Status "active"
```

### 1.4 Remplacer Placeholder dans Landing

```bash
# Éditer apps/landing/index.html ligne 210
# Remplacer:
STRIPE_LINK_BETA

# Par:
https://buy.stripe.com/test_...  # (ou live link si prod)
```

**Vérification :**
```bash
grep -n "https://buy.stripe.com" apps/landing/index.html
# Doit afficher: 210:    <a href="https://buy.stripe.com/test_..." class="btn-primary btn-block">Subscribe Now</a>
```

---

## Phase 2 : Track Record Automatisé (30 min)

### 2.1 Vérifier Alpaca API Keys

```bash
# Créer/vérifier .env.broker.alpaca
cat > .env.broker.alpaca <<EOF
APCA_API_KEY_ID=your_paper_key_id
APCA_API_SECRET_KEY=your_paper_secret_key
APCA_API_BASE_URL=https://paper-api.alpaca.markets
EOF

# Tester connexion
source .env.broker.alpaca
python -c "
import os
from alpaca.trading.client import TradingClient
client = TradingClient(os.environ['APCA_API_KEY_ID'], os.environ['APCA_API_SECRET_KEY'], paper=True)
account = client.get_account()
print(f'✓ Alpaca OK - Equity: \${account.equity}')
"
```

### 2.2 Premier Snapshot

```bash
source .env.broker.alpaca

python scripts/ops/alpaca_track_record_snapshot.py \
  --paper \
  --out-dir docs/reports/track_record

# Expected output:
# ✓ Snapshot saved: docs/reports/track_record/alpaca_snapshot_2025-01-XX...json
```

### 2.3 Premier Report

```bash
python scripts/ops/make_track_record_report.py \
  --in-dir docs/reports/track_record \
  --out-dir docs/reports/track_record

# Expected output:
# ✓ Report generated: docs/reports/track_record/TRACK_RECORD_latest.md
# ✓ JSON exported: docs/reports/track_record/track_record_latest.json

# Vérifier
cat docs/reports/track_record/TRACK_RECORD_latest.md
cat docs/reports/track_record/track_record_latest.json | jq .
```

### 2.4 Automation Daily (Cron)

**Option A : Crontab (Linux/Mac)**

```bash
# Créer script runner
cat > scripts/ops/daily_track_record.sh <<'EOF'
#!/bin/bash
set -e
cd /path/to/HyprL  # REMPLACER PAR CHEMIN ABSOLU
source .env.broker.alpaca
.venv/bin/python scripts/ops/alpaca_track_record_snapshot.py --paper --out-dir docs/reports/track_record
.venv/bin/python scripts/ops/make_track_record_report.py --in-dir docs/reports/track_record --out-dir docs/reports/track_record
echo "[$(date)] ✓ Track record updated" >> live/logs/daily_track_record.log
EOF

chmod +x scripts/ops/daily_track_record.sh

# Test manuel
./scripts/ops/daily_track_record.sh

# Ajouter au crontab (18h00 daily, lundi-vendredi)
crontab -e
# Ajouter ligne:
0 18 * * 1-5 /path/to/HyprL/scripts/ops/daily_track_record.sh
```

**Option B : GitHub Actions (alternatif)**

Créer `.github/workflows/daily-track-record.yml` :

```yaml
name: Daily Track Record Update

on:
  schedule:
    - cron: '0 18 * * 1-5'  # 18h UTC, Mon-Fri
  workflow_dispatch:  # Manual trigger

jobs:
  update-track-record:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install deps
        run: pip install -r requirements.txt
      - name: Snapshot
        env:
          APCA_API_KEY_ID: ${{ secrets.APCA_API_KEY_ID }}
          APCA_API_SECRET_KEY: ${{ secrets.APCA_API_SECRET_KEY }}
          APCA_API_BASE_URL: https://paper-api.alpaca.markets
        run: |
          python scripts/ops/alpaca_track_record_snapshot.py \
            --paper --out-dir docs/reports/track_record
      - name: Generate report
        run: |
          python scripts/ops/make_track_record_report.py \
            --in-dir docs/reports/track_record \
            --out-dir docs/reports/track_record
      - name: Commit and push
        run: |
          git config user.name "HyprL Bot"
          git config user.email "[email protected]"
          git add docs/reports/track_record/
          git commit -m "Update track record $(date +%Y-%m-%d)" || exit 0
          git push
```

**Configuration GitHub Secrets :**
```
Settings → Secrets and variables → Actions → New repository secret
Name: APCA_API_KEY_ID
Value: PKxxx...

Name: APCA_API_SECRET_KEY
Value: xxx...
```

---

## Phase 3 : Déploiement Public (30 min)

### 3.1 Deploy Landing (Netlify)

**Option A : Netlify CLI**

```bash
# Installer Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Init (première fois)
cd apps/landing
netlify init
# → Choose: Create & configure a new site
# → Team: Your team
# → Site name: hyprl-landing (ou custom)
# → Build command: (laisser vide)
# → Publish directory: . (current)

# Deploy
netlify deploy --prod

# Output: Site URL: https://hyprl-landing.netlify.app
```

**Option B : Netlify Dashboard (UI)**

1. Aller sur [app.netlify.com](https://app.netlify.com)
2. Add new site → Import from Git
3. OU : Drag & drop le dossier `apps/landing/`
4. Deploy settings:
   - Build command: (vide)
   - Publish directory: `/`
5. Deploy

**Vérification :**
```bash
# Tester form submission
open https://hyprl-landing.netlify.app
# → Remplir beta form → doit rediriger vers /thanks.html
# → Vérifier Netlify Dashboard → Forms
```

### 3.2 Deploy Track Record Portal (Streamlit Cloud)

**Prérequis :** Commit + push reports to GitHub

```bash
# Commit track record reports
git add docs/reports/track_record/TRACK_RECORD_latest.md
git add docs/reports/track_record/track_record_latest.json
git commit -m "Add initial track record"
git push origin main  # ou votre branch
```

**Deployment Streamlit Cloud :**

1. Aller sur [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. New app
4. Settings :
   ```
   Repository: Kacawaiii/HyprL
   Branch: main (ou votre branch)
   Main file path: apps/track_record/streamlit_app.py
   App URL: hyprl-track-record (ou custom)
   ```
5. **Advanced settings** (optionnel) :
   - Python version: 3.11
   - Requirements file: requirements.txt (auto-detect)
6. Deploy

**Custom domain (optionnel) :**
```
Settings → General → Custom subdomain
hyprl → https://hyprl.streamlit.app
```

**Vérification :**
```bash
# Tester accès public
open https://hyprl.streamlit.app

# Tester avec clé (après création entitlement)
open https://hyprl.streamlit.app?key=YOUR_TEST_KEY
```

---

## Phase 4 : Test End-to-End (30 min)

### 4.1 Happy Path Complet

**Scénario : Premier client Beta**

```bash
# 1. Client visite landing
open https://hyprl-landing.netlify.app

# 2. Clique "Subscribe Now" (Beta €19/mo)
# → Redirigé vers Stripe Checkout

# 3. Remplit carte test
Email: [email protected]
Card: 4242 4242 4242 4242

# 4. Paiement réussi → redirection /thanks.html

# 5. Opérateur : Export Stripe CSV
# Stripe Dashboard → Customers → Export → CSV
# Télécharger stripe_customers_2025-01-XX.csv

# 6. Import entitlements
python scripts/ops/stripe_entitlements.py import \
  --csv ~/Downloads/stripe_customers_2025-01-XX.csv

# Output:
# ✓ Entitlement created for [email protected]
#   Access Key: abc123def456...
#   Tier: beta

# 7. Export allowlist
python scripts/ops/stripe_entitlements.py export

# Output:
# ✓ Exported 1 active keys to apps/track_record/entitlements_allowlist.txt
# ✓ Audit log saved to docs/reports/entitlements/ENTITLEMENTS_2025-01-XX.json

# 8. Envoyer email au client
python scripts/ops/stripe_entitlements.py list --status active

# Copier access key et envoyer email:
---
Subject: Your HyprL Beta Access Key

Hi Test User,

Welcome to HyprL Beta! Your access key is:

abc123def456...

Access the track record dashboard at:
https://hyprl.streamlit.app?key=abc123def456...

Questions? Reply to this email.

Thanks,
HyprL Team
---

# 9. Client teste accès
open https://hyprl.streamlit.app?key=abc123def456...

# ✓ Dashboard charge avec métriques
# ✓ Accès private sections
# ✓ Export CSV disponible
```

### 4.2 Checklist Validation

- [ ] **Paiement Stripe** : Checkout fonctionne, subscription créée
- [ ] **CSV Export** : Customer exporté avec email + subscription_id
- [ ] **Import Entitlements** : Clé générée et stockée en DB
- [ ] **Export Allowlist** : `entitlements_allowlist.txt` créé
- [ ] **Auth Portal** : Clé valide donne accès, clé invalide refuse
- [ ] **Email Template** : Copié et prêt à envoyer
- [ ] **Track Record** : Mis à jour daily (cron/GH Actions)
- [ ] **Landing Public** : Accessible sur Netlify
- [ ] **Portal Public** : Accessible sur Streamlit Cloud

---

## Phase 5 : Monitoring + Support (15 min)

### 5.1 Monitoring Simple

**Track Record Health Check :**

```bash
# Créer script de vérification
cat > scripts/ops/check_track_record_health.sh <<'EOF'
#!/bin/bash
LATEST_JSON="docs/reports/track_record/track_record_latest.json"

if [ ! -f "$LATEST_JSON" ]; then
  echo "✗ ERROR: track_record_latest.json missing"
  exit 1
fi

# Check last update (doit être < 48h)
LAST_UPDATE=$(jq -r '.generated_at' "$LATEST_JSON")
echo "Last update: $LAST_UPDATE"

# Check no NaN values
if jq . "$LATEST_JSON" | grep -q "NaN"; then
  echo "✗ ERROR: NaN values detected"
  exit 1
fi

echo "✓ Track record healthy"
EOF

chmod +x scripts/ops/check_track_record_health.sh

# Tester
./scripts/ops/check_track_record_health.sh
```

**Email Alert (optionnel) :**

```bash
# Ajouter à cron (daily check à 19h)
0 19 * * 1-5 /path/to/HyprL/scripts/ops/check_track_record_health.sh || echo "Track record health check failed" | mail -s "HyprL Alert" [email protected]
```

### 5.2 Procédures Support

**Reset Access Key :**

```bash
# 1. Client demande reset
# 2. Désactiver ancienne clé
python scripts/ops/stripe_entitlements.py verify --key OLD_KEY
# → Si trouvé, noter l'email

# 3. Générer nouvelle clé
python scripts/ops/stripe_entitlements.py generate \
  --email [email protected] \
  --tier beta

# 4. Export allowlist
python scripts/ops/stripe_entitlements.py export

# 5. Envoyer nouveau email avec nouvelle clé
```

**Cancel Subscription :**

```bash
# 1. Client annule dans Stripe Customer Portal
# OU manuellement dans Dashboard: Subscriptions → Cancel

# 2. Sync entitlements
python scripts/ops/stripe_entitlements.py sync

# 3. Export allowlist (exclut canceled)
python scripts/ops/stripe_entitlements.py export

# 4. Redémarrer portal (Streamlit Cloud auto-restart, ou manual)
```

---

## Phase 6 : Cadre Business (Hors scope technique)

### 6.1 Statut Légal

**Options France :**

- **Micro-entreprise** (le plus simple pour démarrer)
  - Plafond : 77,700 € CA/an (services BIC)
  - TVA : franchise en base (pas de TVA si < seuils)
  - Déclaration : mensuelle ou trimestrielle
  - **Recommandé pour Beta (< 1000 €/mois)**

- **EURL/SASU** (si croissance rapide)
  - Comptabilité + expert-comptable
  - TVA collectée/déductible
  - Plus de flexibilité fiscale

**Démarches micro-entreprise :**
1. URSSAF auto-entrepreneur (en ligne, gratuit)
2. Choisir activité : "Conseil en systèmes et logiciels informatiques" (code NAF 6202A)
3. Option fiscale : Versement libératoire si éligible
4. Obtenir SIRET (2-3 semaines)

### 6.2 TVA + Facturation

**Si micro-entreprise (franchise TVA) :**

- Prix TTC = Prix HT (pas de TVA)
- Mentions obligatoires facture :
  ```
  "TVA non applicable, article 293 B du CGI"
  ```

**Si assujetti TVA (EURL/SASU) :**

- TVA standard France : 20%
- Prix affiché : €19 TTC → €15.83 HT + €3.17 TVA
- Stripe gère la collecte, vous reversez

**Facturation automatique Stripe :**

Stripe → Settings → Customer emails → Enable automatic receipts

Ou utiliser Stripe Tax + Invoicing pour factures conformes.

### 6.3 Mentions Légales Landing

**À ajouter dans footer (apps/landing/index.html) :**

```html
<p class="footer-legal">
  HyprL — [SIRET si micro-entreprise] — [email protected]<br>
  Siège social : [Adresse] — France<br>
  TVA non applicable, article 293 B du CGI
</p>
```

---

## Checklist Go-Live Final

### Pré-Launch (J-7)

- [ ] Track record : 7 jours de données paper (daily cron OK)
- [ ] Métriques stables (no NaN, equity curve propre)
- [ ] Stripe products créés (Beta €19/mo)
- [ ] Payment link testé (happy path end-to-end)
- [ ] Landing déployé (Netlify)
- [ ] Portal déployé (Streamlit Cloud)
- [ ] Auth entitlements testé (allowlist fonctionne)
- [ ] Email template prêt
- [ ] Statut légal clarifié (micro-entreprise ou autre)

### Launch Day (J0)

- [ ] Remplacer `STRIPE_LINK_BETA` par lien live
- [ ] Annoncer sur Discord/Twitter/LinkedIn (si applicable)
- [ ] Monitorer Netlify Forms (submissions waitlist)
- [ ] Monitorer Stripe Dashboard (paiements)
- [ ] Répondre emails < 24h

### Post-Launch (J+1 → J+30)

- [ ] Onboarding clients Beta (export CSV → import → send key)
- [ ] Daily track record update (cron vérifié)
- [ ] Collecter feedback Discord
- [ ] Itérer sur bugs/UX
- [ ] Préparer tier Starter (après 30j track record stable)

---

## Commandes Rapides (Copier-Coller)

```bash
# Daily snapshot + report
source .env.broker.alpaca && \
python scripts/ops/alpaca_track_record_snapshot.py --paper --out-dir docs/reports/track_record && \
python scripts/ops/make_track_record_report.py --in-dir docs/reports/track_record --out-dir docs/reports/track_record

# Onboarding client
python scripts/ops/stripe_entitlements.py import --csv ~/Downloads/stripe_customers.csv && \
python scripts/ops/stripe_entitlements.py export && \
python scripts/ops/stripe_entitlements.py list --status active

# Health check
./scripts/ops/check_track_record_health.sh

# Deploy landing
cd apps/landing && netlify deploy --prod

# Deploy portal (via git push)
git add docs/reports/track_record/ && \
git commit -m "Update track record" && \
git push
```

---

**Durée totale estimée : 2-3 heures**

**Résultat : Produit 100% vendable pour les 30 premiers Beta users.**

Questions ? Bloqué sur une étape ? Dis-moi où tu en es et je t'aide à débloquer.
