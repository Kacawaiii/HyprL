# Alpaca execution v2

From the October 9, 2026 decision, `ia_actions` executes the operator-approved
`REVIEWED_ANALYST_UNION`. Either analyst's issued UP or DOWN view can contribute
when the reviewer says KEEP. Probability controls position size through the
existing formula; share flooring, caps and timing still apply. DOWNGRADE, REJECT, ABSTAIN and MISSING do not contribute. An issued UP versus
DOWN conflict on the same asset and horizon vetoes execution, even when one of
those views was rejected. Agreeing eligible views produce one lot, sized using
the probability closest to 0.50; analyst name resolves ties deterministically.

Consensus remains recorded and scored with the existing primary shadow rules.
The execution variants `alpaca_open_entry_v2` and `alpaca_after_hours_v2` remain
exploratory. After-hours research predictions retain their existing label
variant; paper lot outcomes identify the execution-v2 population separately.
The equity union admits every directional KEEP view; crypto retains the 0.55
consensus threshold. Sizing, account caps, short permissions, kill switch, order accounting,
idempotency, causal availability and timing retain the v1 rules. Entry crypto
quantities still floor to eight decimals; v2 allocates partial fills in the
observed nine-decimal broker units. Previous-spec intents retain eight-decimal
fill allocation.

The authoritative artifacts are `trader_alpaca_execution_spec_v2.json` and
preregistration revision 5 in `trader_agent_preregistration_v2.json`. Their
canonical hashes are bound in code. The previous artifacts remain unchanged.
Old decisions are ineligible for new production execution.

## Crypto amendment

The versioned `trader_crypto_universe_amendment_v1.json` artifact freezes
SOL-USD, AVAX-USD, LINK-USD, DOGE-USD and LTC-USD. Each candidate was verified as
active and tradable on Alpaca paper, absent from the protection contracts, with
Coinbase public daily candles and the exact one-minute label anchor. The
liquidity screen uses mean daily close times volume across 20 completed candles
of at least $5 million. This is a selection screen, not a measured spread or
execution-cost claim. BTC-USD and ETH-USD remain excluded; the other Alpaca pairs
are outside this scoped amendment and were not given candle captures.

Install the frozen amendment beside the private grants as
`trader-agent-crypto-amendment-v1.json`. The loader verifies its hash, original
grant identity and current protection registry before creating an effective
authorization. It never modifies `trader-agent-v1.json` or
`trader-alpaca-paper-v1.json`. The analyst context and source whitelist use this
effective universe. A trader-scoped candle parser preserves Coinbase's native
six fields; the shared BTC/ETH MarketBar v1 contract is unchanged.

The Coinbase daily cap rises from 20 to 40: five daily feature calls plus up to
35 minute calls when Monday's label job observes weekend maturities (seven
distinct entry/exit anchors per asset). The label job shares its cache across
all analyst, reviewer and consensus predictions. Missed jobs or retries receive
no extra allowance; excess labels stay pending. Dispatch history is retained
across the authorization amendment.

## Explicit ledger rebinding and replay

Rebind only during a deployment window with the trader timers quiescent:

```sh
python -m scripts.trading_lab.trader_agent.cli paper-rebind \
  --authorization "$TRADER_GRANT" --paper-authorization "$PAPER_GRANT" \
  --runtime "$TRADER_RUNTIME" --operator-decision "$OPERATOR_DECISION"
```

The exact approved operator decision must match its frozen canonical hash.
Under the paper owner lock, verify the existing chain, refuse any open lot or
nonterminal intent, verify both broker account suffixes, and GET positions and
open orders. Both AI accounts must be empty. Append a new binding carrying the
previous binding digest, grant/spec identities and the new spec,
preregistration, base/effective grant and amendment hashes. Historical events,
order caps, account peaks and halt state remain intact. Without this explicit
transition, old stores stay rejected.

The October 8 run can be replayed without an order:

```sh
python -m scripts.trading_lab.trader_agent.cli paper-execute --dry-run \
  --paper-replay-at 2026-10-08T12:16:00Z --paper-account ia_actions \
  --authorization "$TRADER_GRANT" --paper-authorization "$PAPER_GRANT" \
  --runtime "$TRADER_RUNTIME"
```

Replay requires dry-run execution. Planning uses the historical decision time;
GET authorization uses the actual observation time. Broker mutations and order
reservations remain disabled. This replay would plan XOM, CVX and XLE; it does
not turn a missed opening auction into an order or a scored fill.

Account suffix 2EQN is reported as `claude_book`, remains GET-only, and is outside
all AI variants, combined equity and baselines. Its original private grant key
and historical journal events remain unchanged. No discretionary book exists
inside either AI account.
