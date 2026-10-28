# Growth v1 browser evidence

All screenshots use the isolated `test/growth-browser-server.js`, synthetic
DEMO events and an explicitly synthetic claim. No production store or recipient
is read. These images are documentation, never served as product results.

Reproduce from the site directory after `npm ci` and installing the QA requirements:

```sh
PYTHONPATH=.qa-deps python -m pip install --target .qa-deps playwright==1.56.0
PYTHONPATH=.qa-deps python "$SKILLS_LIB/webapp-testing/scripts/with_server.py" \
  --server 'node test/growth-browser-server.js' --port 3818 -- \
  python test/growth_browser_audit.py
```

The audit blocks every non-loopback request, checks empty visitor storage,
JSON-LD/canonicals, light/dark pages, a 320 px viewport, keyboard menu, the
unticked optional checkbox, native signup, double opt-in and withdrawal.
Thirteen WCAG A/AA axe checks are automated evidence, not a complete RGAA audit.

Nine captures: calendar light/dark/mobile; claim index and article; SEC guide;
referral rules; unticked newsletter form; confirmed newsletter signup.
