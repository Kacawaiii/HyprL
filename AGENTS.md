# Rules for coding agents on HyprL

These rules bind every agent (Claude Code, Codex, any other) working on this repository, locally or on
the shared server. When a rule and a task disagree, the rule wins: stop and report instead.

## Never

- **Commit a secret.** The repository is public. No tokens, keys, passwords, `.env` content, contact
  e-mails, private paths, raw captured bodies, stores, logs or fixtures. Commit only code, synthetic
  data, and digests/URLs/counts as evidence.
- **Send a real request to an official source** (Federal Reserve, SEC EDGAR, any capture target)
  without an operator authorization file naming the scope, budget and expiry. Offline work and the
  synthetic providers are always fine.
- **Weaken a rule to make a check pass.** Specs (`docs/artifacts/*_spec_v1.json`) are authoritative; a
  change needs a new revision, its canonical hash and the code bindings (`spec.py`), and stores bound to
  the old hash stay rejected (no implicit migration).
- **Touch holdout data, train models, or trade** (paper or live) unless a task explicitly authorizes it.
- **Rewrite history**: no force push, no rebase of a pushed branch, never merge `main`.
- **Touch archives or frozen code**: closed pilots, their stores and their detached worktrees.

## Git

- Start from the current `origin/feat/phase5`; work on your own branch `agents/<agent>/<topic>`.
- Small coherent commits, message body explaining the why, trailer `Co-Authored-By: <agent> <...>`.
- Push your branch only, fast-forward. Integration into `feat/phase5` is fast-forward only, after the full
  checks below pass on the integrated result.

## Checks before any push

```
python -m pytest tests/crypto/test_fomc_*.py tests/crypto/test_sources_primitives.py \
  tests/crypto/test_edgar_slice.py tests/crypto/test_app_api*.py tests/crypto/test_ops_server.py \
  tests/crypto/test_equity_api.py tests/api -q
cd apps/web && npm ci && npm run typecheck && npm run lint && npx vitest run      # when apps/web changed
```

A check that cannot run (missing dependency, blocked network or loopback) is reported as BLOCKED with
its cause, never as passed.

## Where things are

- Official event sources: `scripts/trading_lab/sources/` (shared store, causal availability, HTTP clock,
  limiter), `scripts/trading_lab/fomc/`, `scripts/trading_lab/edgar/`.
- Registries (state, proofs, limits): `docs/FOMC_V1_OFFLINE_SLICE.md`, `docs/EDGAR_V1_OFFLINE_SLICE.md`.
- Read-only application API: `scripts/trading_lab/app_api/`; cockpit: `apps/web/`.

## Reporting

End every task with a short report: what changed (commits), what was verified (commands and results),
what is blocked or unproven, and the next step. Say "not done" plainly when it is not done.
