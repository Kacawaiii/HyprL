# HyprL Stripe Setup — Monetization Guide

**Objective:** Set up Stripe payment processing for HyprL subscriptions with automatic entitlement management.

---

## 1. Stripe Account Setup

### Create Stripe Account

1. Go to [stripe.com/register](https://stripe.com/register)
2. Complete business information:
   - **Business type:** Individual or Company
   - **Country:** France
   - **Business name:** HyprL
   - **Product description:** "Algorithmic trading signal intelligence and analytics platform"

### Activate Payments

1. Complete identity verification (ID + bank details)
2. Add bank account for payouts
3. Enable payment methods:
   - ✅ Cards (Visa, Mastercard, Amex)
   - ✅ SEPA Direct Debit (EU)
   - ✅ Apple Pay / Google Pay (optional)

---

## 2. Create Products

### Beta Early Access

```
Name: HyprL Beta Early Access
Description: Live track record dashboard, Discord community, weekly updates
Price: €19/month
Billing: Recurring monthly
Trial: None
```

**Stripe CLI:**
```bash
stripe products create \
  --name "HyprL Beta Early Access" \
  --description "Live track record dashboard, Discord community, weekly updates"

stripe prices create \
  --product <PRODUCT_ID> \
  --currency eur \
  --unit-amount 1900 \
  --recurring='{"interval":"month"}'
```

---

### Starter

```
Name: HyprL Starter
Description: Dashboard, email alerts, monthly reports, full trade history
Price: €29/month
Billing: Recurring monthly
Trial: 7 days (optional)
```

**Stripe CLI:**
```bash
stripe products create \
  --name "HyprL Starter" \
  --description "Dashboard, email alerts, monthly reports, full trade history"

stripe prices create \
  --product <PRODUCT_ID> \
  --currency eur \
  --unit-amount 2900 \
  --recurring='{"interval":"month","trial_period_days":7}'
```

---

### Pro

```
Name: HyprL Pro
Description: Real-time API access, webhooks, priority support, advanced analytics
Price: €79/month
Billing: Recurring monthly
Trial: 7 days (optional)
```

**Stripe CLI:**
```bash
stripe products create \
  --name "HyprL Pro" \
  --description "Real-time API access, webhooks, priority support, advanced analytics"

stripe prices create \
  --product <PRODUCT_ID> \
  --currency eur \
  --unit-amount 7900 \
  --recurring='{"interval":"month","trial_period_days":7}'
```

---

## 3. Payment Page Setup

### Option A: Stripe Checkout (Recommended for MVP)

**Pros:** No frontend code, Stripe-hosted, PCI compliant

```python
import stripe
stripe.api_key = "sk_test_..."

checkout_session = stripe.checkout.Session.create(
    payment_method_types=['card'],
    line_items=[{
        'price': 'price_...',  # Beta price ID
        'quantity': 1,
    }],
    mode='subscription',
    success_url='https://hyprl.io/success?session_id={CHECKOUT_SESSION_ID}',
    cancel_url='https://hyprl.io/cancel',
    customer_email='[email protected]',
)

print(checkout_session.url)  # Redirect user to this URL
```

**HTML Link:**
```html
<a href="https://buy.stripe.com/test_..." class="btn-primary">
    Subscribe to Beta
</a>
```

Generate link via Stripe Dashboard: **Products → Create payment link**

---

### Option B: Embedded Checkout (Advanced)

For custom branding, embed Stripe Checkout in your site:

```html
<script src="https://js.stripe.com/v3/"></script>
<button id="checkout-button">Subscribe</button>

<script>
const stripe = Stripe('pk_test_...');
document.getElementById('checkout-button').addEventListener('click', async () => {
    const {error} = await stripe.redirectToCheckout({
        lineItems: [{price: 'price_...', quantity: 1}],
        mode: 'subscription',
        successUrl: 'https://hyprl.io/success',
        cancelUrl: 'https://hyprl.io/cancel',
    });
    if (error) console.error(error);
});
</script>
```

---

## 4. Webhook Integration

### Setup Webhook Endpoint

Stripe sends events (subscription created, payment succeeded, etc.) to your server.

**Endpoint URL:** `https://hyprl.io/api/stripe/webhook`

**Events to listen:**
- `checkout.session.completed` → New subscription
- `customer.subscription.updated` → Subscription changed
- `customer.subscription.deleted` → Cancellation
- `invoice.payment_succeeded` → Successful payment
- `invoice.payment_failed` → Failed payment

**Python Example:**
```python
from flask import Flask, request
import stripe

app = Flask(__name__)
stripe.api_key = "sk_live_..."
endpoint_secret = "whsec_..."

@app.route('/api/stripe/webhook', methods=['POST'])
def webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError:
        return "Invalid payload", 400
    except stripe.error.SignatureVerificationError:
        return "Invalid signature", 400

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        customer_email = session['customer_email']
        subscription_id = session['subscription']

        # Generate access key
        import secrets
        access_key = secrets.token_urlsafe(32)

        # Store in database
        add_entitlement(customer_email, access_key, subscription_id)

        # Send welcome email
        send_welcome_email(customer_email, access_key)

    elif event['type'] == 'customer.subscription.deleted':
        subscription_id = event['data']['object']['id']
        remove_entitlement(subscription_id)

    return "Success", 200
```

---

## 5. Entitlement Management

### Discord ID Capture + Role Sync

- Add a Stripe custom field named "Discord ID" on the payment link.
- Store it in the entitlements database (`discord_id` column).
- Sync roles with:
  `python scripts/ops/stripe_entitlements.py discord-sync --enforce-single-tier --prune-inactive`

Required env vars:
- `DISCORD_BOT_TOKEN`
- `DISCORD_GUILD_ID`
- `DISCORD_ROLE_BETA`
- `DISCORD_ROLE_STARTER`
- `DISCORD_ROLE_PRO`

### Database Schema

```sql
CREATE TABLE entitlements (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    access_key VARCHAR(64) UNIQUE NOT NULL,
    discord_id VARCHAR(64),
    subscription_id VARCHAR(255) UNIQUE NOT NULL,
    tier VARCHAR(50) NOT NULL,  -- 'beta', 'starter', 'pro'
    status VARCHAR(50) NOT NULL,  -- 'active', 'canceled', 'past_due'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_access_key ON entitlements(access_key);
CREATE INDEX idx_subscription_id ON entitlements(subscription_id);
```

### Python Script

See `scripts/ops/stripe_entitlements.py` for full implementation.

**Key functions:**
```python
def generate_access_key():
    return secrets.token_urlsafe(32)

def add_entitlement(email, access_key, subscription_id, tier='beta'):
    # Insert into database
    pass

def verify_access_key(key):
    # Check if key is valid and subscription active
    return True/False

def sync_stripe_subscriptions():
    # Periodic job: sync Stripe status → local DB
    subscriptions = stripe.Subscription.list(limit=100)
    for sub in subscriptions:
        update_entitlement_status(sub.id, sub.status)
```

---

## 6. Testing

### Test Mode

Use Stripe test keys:
```
Publishable: pk_test_...
Secret: sk_test_...
```

**Test card numbers:**
- Success: `4242 4242 4242 4242`
- Decline: `4000 0000 0000 0002`
- 3D Secure: `4000 0027 6000 3184`

**Expiry:** Any future date (e.g., 12/34)
**CVC:** Any 3 digits

### Test Checklist

- [ ] Create checkout session
- [ ] Complete payment with test card
- [ ] Verify webhook received
- [ ] Access key generated and stored
- [ ] User can access dashboard with key
- [ ] Subscription cancellation works
- [ ] Failed payment handling

---

## 7. Go Live

### Switch to Production

1. Complete Stripe account activation
2. Replace test keys with live keys:
   ```bash
   # .env.stripe
   STRIPE_PUBLIC_KEY=pk_live_...
   STRIPE_SECRET_KEY=sk_live_...
   STRIPE_WEBHOOK_SECRET=whsec_...
   ```
3. Update webhook endpoint to production URL
4. Test with real (small) payment
5. Monitor Stripe Dashboard for issues

### Security

- **Never commit keys to git**
- Use environment variables
- Verify webhook signatures
- Use HTTPS only
- Implement rate limiting

---

## 8. Customer Management

### Stripe Customer Portal

Enable self-service for users:
- Update payment method
- View invoices
- Cancel subscription

**Setup:**
Stripe Dashboard → Settings → Billing → Customer portal

**Link:**
```python
portal_session = stripe.billing_portal.Session.create(
    customer='cus_...',
    return_url='https://hyprl.io/account',
)
print(portal_session.url)  # Redirect user here
```

---

## 9. Pricing Strategy

### Beta Launch (Now)

- **€19/month** (first 30 users)
- **50% off first month** for first 10
- Goal: 20-30 paying beta users = **€400-600/mo**

### Public Launch (Q1 2026)

- Starter: **€29/month**
- Pro: **€79/month**
- Goal: 100 users = **€3,000-8,000/mo**

### Scale (Q2 2026+)

- Add annual plans (2 months free)
- Team pricing (5+ seats, €199/mo)
- White-label (€2,000-5,000/mo)

---

## 10. Analytics & Reporting

### Stripe Dashboard

Monitor:
- MRR (Monthly Recurring Revenue)
- Churn rate
- LTV (Lifetime Value)
- Failed payments

### Export Data

```bash
stripe balance_transactions list --limit 100 > transactions.json
stripe customers list --limit 100 > customers.json
```

### Webhooks to Track

- New subscriber count
- Churn count
- Revenue per day
- Failed payment recovery

---

## 11. Operator Flow (V0 - Manual Process)

For V0 launch without full webhook automation, follow this manual operator flow:

### Step 1: Export Customers from Stripe

```bash
# Export from Stripe Dashboard
# Navigate to: Customers → Export → CSV
# Download file as: stripe_customers_YYYY-MM-DD.csv
```

Or use Stripe CLI:
```bash
stripe customers list --limit 100 > customers.json
# Convert to CSV manually or use jq/python script
```

### Step 2: Import to Entitlements Database

```bash
# Import CSV (expected columns: Email, Subscription ID, Status, Product)
python scripts/ops/stripe_entitlements.py import --csv stripe_customers_2025-01-15.csv
```

Expected CSV format:
```csv
Email,Subscription ID,Status,Product
[email protected],sub_1234567890,active,HyprL Beta Early Access
[email protected],sub_0987654321,active,HyprL Starter
```

### Step 3: Export to Allowlist

```bash
# Generate allowlist file + audit log
python scripts/ops/stripe_entitlements.py export
```

This creates:
- `apps/track_record/entitlements_allowlist.txt` (one key per line)
- `docs/reports/entitlements/ENTITLEMENTS_YYYY-MM-DD.json` (audit log)

### Step 4: Send Access Keys to Customers

```bash
# List all entitlements
python scripts/ops/stripe_entitlements.py list --status active

# Copy keys and send via email template:
---
Subject: Your HyprL Beta Access Key

Hi {NAME},

Welcome to HyprL Beta! Your access key is:

{ACCESS_KEY}

Access the track record dashboard at:
https://hyprl.io/track-record?key={ACCESS_KEY}

Questions? Reply to this email.

Thanks,
HyprL Team
---
```

### Step 5: Key Rotation (Weekly Recommended)

```bash
# 1. Sync with Stripe to update statuses
python scripts/ops/stripe_entitlements.py sync

# 2. Export fresh allowlist (excludes canceled)
python scripts/ops/stripe_entitlements.py export

# 3. Restart track record app (picks up new allowlist)
# See docs/TRACK_RECORD_OPS.md for deployment instructions
```

### Security Best Practices

- ✅ **Never commit** `entitlements_allowlist.txt` to git (add to `.gitignore`)
- ✅ **Never log** full access keys (only prefixes: `abc12345...`)
- ✅ **Rotate keys** if leaked (generate new, invalidate old)
- ✅ **Audit regularly** using `docs/reports/entitlements/*.json` logs

### Manual Entitlement Generation

For special cases (free beta, partners, testing):

```bash
# Generate single entitlement
python scripts/ops/stripe_entitlements.py generate \
  --email [email protected] \
  --tier beta

# Verify a key
python scripts/ops/stripe_entitlements.py verify \
  --key abc123def456...
```

---

## Next Steps

1. ✅ Create Stripe account
2. ✅ Set up products (Beta, Starter, Pro)
3. ✅ Generate payment links
4. ✅ Implement webhook handler (optional for V0)
5. ✅ Create entitlements database
6. ✅ Test with test cards
7. ⏳ **Run operator flow** (import → export → send keys)
8. ⏳ Go live with Beta
9. ⏳ Monitor first 10 customers
10. ⏳ Iterate based on feedback

**Estimated setup time:** 2-3 hours (manual flow) / 4-6 hours (with webhooks)

---

## Resources

- [Stripe Docs](https://stripe.com/docs)
- [Stripe Dashboard](https://dashboard.stripe.com/)
- [Stripe CLI](https://stripe.com/docs/stripe-cli)
- [Webhook Testing](https://stripe.com/docs/webhooks/test)
- [Customer Portal](https://stripe.com/docs/billing/subscriptions/customer-portal)

**Questions?** Check `scripts/ops/stripe_entitlements.py` for implementation examples.
