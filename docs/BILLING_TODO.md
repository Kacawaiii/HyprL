# Billing TODO (Stripe)

## Checklist

- [ ] Create Stripe product + pricing for Beta Basic / Beta Pro.
- [ ] Set up checkout page (manual link ok for v0).
- [ ] Create webhook endpoint (payment success).
- [ ] On webhook:
  - [ ] Grant Discord role
  - [ ] Issue API key
- [ ] On cancel/failure:
  - [ ] Revoke Discord role
  - [ ] Revoke API key
- [ ] Log all entitlements changes.
- [ ] Add audit trail for billing events.

## Notes

- Keep manual onboarding until webhook path is tested.
- Never store raw card data; Stripe handles payment tokens.
