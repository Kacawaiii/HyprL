# Execution v3 half-size tiers

Execution spec v3 and execution preregistration revision 7 are bound by canonical
hash before any v3 order. The original paper grant is immutable. Historical
weekday research revision 5, weekend research revision 6, and full-size execution
identities remain unchanged. New decisions record an additional execution
preregistration hash; older decisions can only exercise v3 through a dry replay.

For equities, select KEEP before DOWNGRADE when agreeing views coexist. Within
the selected tier, use the original analyst probability closest to 0.50, with an
analyst name tie-break. KEEP retains the v2 formula and identity. DOWNGRADE uses
`half_size_downgrade_v1`. Compute the v2 quantity with its existing reference,
reserve, cohort fractions and cap scale, floor whole shares, then halve and floor
again. Zero shares produce no intent. Reviewer-adjusted probabilities continue
to belong to their original research scores.

For both crypto populations, two directional KEEP analysts with a qualifying
consensus retain the full-size identity (`alpaca_open_entry_v1` on weekdays,
`weekend_crypto_v1` on closed days). Exactly one directional KEEP analyst, with
the other abstaining or missing, uses `crypto_single_kept_half_v1`. DOWNGRADE does
not contribute to a crypto order. The 0.55 threshold, long/flat restriction and
1e-8 entry flooring remain. Any issued UP/DOWN conflict vetoes all tiers,
including a rejected or downgraded opposite view.

`paper-report` exposes `execution_scores`, grouped by account, decision
population, tier and horizon. Fill, pending, unfilled and remaining-lot counts
are distinct; realized lot P&L before broker fees comes from cumulative exit
observations. Repeated observations do not add the same P&L twice. Shadow
research and the primary consensus hypothesis retain their original scores.

`paper-rebind` requires the approved decision, exact previous v2 binding and
unchanged grant. It refuses all open lots, nonterminal intents, broker positions
or open orders in either AI account. No open lot is covered by this transition.
The appended event names the previous binding/spec/grant digests and new
execution and research registrations. Historical v1/v2 transitions and the
original weekend registration are verified and preserved. Rebinding issues
GETs only to the two AI paper accounts; 2EQN remains outside this transition.

The offline replay module accepts `--research-root`, `--authorization`,
`--paper-authorization`, and `--day`. It opens recorded research read-only,
starts a disposable flat-account ledger, and plans at the first instant after
all predictions were durably recorded. It acquires no source, model or broker
data and submits no orders. Fixed equity is 100000 per account; results are
counterfactual sizing evidence, not realized performance.

The committed evidence artifact contains only digests, identities and counts.
The 2026-10-09 DEGRADED decision would produce these 1d DOWNGRADE lots:

| Asset | Direction | Whole shares |
| --- | --- | ---: |
| CVX | sell | 1 |
| NVDA | buy | 1 |
| XLE | sell | 5 |
| XLK | buy | 1 |
| XOM | sell | 2 |

The remaining five selected lots floor to zero (AVGO/1d, AMD/1d, XOM/5d,
CVX/5d, XLE/5d). Crypto produces zero orders. On 2026-10-10 both analysts
abstained on every crypto horizon, so v3 also produces zero orders.
