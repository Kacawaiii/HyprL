/** Synthetic snapshots with the shape the export job writes. No real name, price or account is used. */
import type { PaperSnapshot, RadarHome } from '../api/radarTypes';

const HASH = 'ab'.repeat(32);
const view = (analyst: string, horizon: string, v: string, p: number, verdict = 'KEEP') => ({
  analyst, horizon, view: v, p_outperform: p, verdict, reason: `${analyst} reasoning`, falsifier: 'A falsifier.',
  review_note: null, catalysts: [],
});

export const radarHome: RadarHome = {
  schema: 'cockpit-radar-home-v1',
  generated_at: '2026-10-10T12:00:00Z',
  radar: {
    date: '2026-10-10', slot: 'morning', cutoff: '2026-10-10T11:32:02Z', status: 'PARTIAL', report_hash: HASH,
    total_events: 120, shown_events: 2, sources_by_status: { LIVE: 40, DEAD: 3 },
    limitations: ['Prices are closes; partial-day volume is not extrapolated.'],
  },
  trader: {
    runs: 2, labels: 1, latest_run: 'trader:2026-10-10:synthetic', latest_run_at: '2026-10-10T12:04:00Z',
    models: { analyst_claude: { model: 'opus', cli_version: '1', reported_version: 'synthetic-claude-1' } },
    preregistration_hash: HASH, predictive_quality: {}, hypothesis_state: 'NOT_CONFIRMED',
  },
  regime: {
    SPY: { symbol: 'SPY', last: 700, at: '2026-10-09T20:00:00Z', status: 'OBSERVED', returns_pct: { '1d': 0.2 } },
    BTC: { symbol: 'BTC/USD', last: 80000, at: '2026-10-10T11:31:00Z', status: 'OBSERVED', returns_pct: { '1d': -1.4 } },
  },
  events: [
    {
      id: 'e1'.padEnd(64, '0'), rank: 1, headline: 'Synthetic oil supply shock', link: 'https://news.example.invalid/oil',
      source: 'examplewire', published_at: '2026-10-10T09:00:00Z', available_at: '2026-10-10T09:05:00Z', novelty: 'new',
      themes: ['energy'], countries: ['Exampleland'],
      badges: {
        event_importance: { value: 82, components: { source_support: 30, independent_publishers: 2 } },
        evidence_strength: { value: 65, status: 'corroborated_reporting', publishers: 2, primary: false },
        model_conviction: { label: 'faible', by_analyst: { analyst_claude: 0.4, analyst_gpt: 0.1 }, basis: 'abs(p - 0.5) * 2' },
        predictive_quality: {
          by_analyst: { 'analyst_claude/1d': { issued: 20, realized: 8, hit_rate: 0.5, brier: 0.26, climatology_brier: 0.25, ic: 0.02, mean_unit_pnl_after_costs: 0, days: 4 } },
          basis: 'scorecard',
        },
        after_cost_performance: {
          event_labels: { n: 2, mean_net_unit_pnl: 0.004 }, paper_accounts: [], basis: 'labelled net_unit_pnl',
        },
      },
      what_changed: {
        expectations: 'Consensus unknown.', changed_expectations: 'Reinforces the supply-risk narrative.',
        summary: 'A synthetic shock.', impact: 'Producers may benefit; airlines pay more for fuel.',
        horizon: 'Days to weeks.', priced_in: 'Partly priced in.', invalidation: 'A durable ceasefire.', source: 'llm_scenario',
      },
      assets: [
        {
          symbol: 'XLE', role: 'sector_etf', mechanism: 'Higher crude supports producers.', direction: 'up',
          priced_in: { return_pct: 1.5, move_atr: 0.8, baseline_at: '2026-10-10T09:00:00Z', price_at: '2026-10-10T11:00:00Z', note: 'observed movement' },
          anticipation: {
            state: 'COVERED',
            latest: {
              run_id: 'trader:2026-10-10:synthetic', at: '2026-10-10T12:04:00Z',
              views: [
                view('analyst_claude', '1d', 'UP', 0.7), view('analyst_gpt', '1d', 'DOWN', 0.4),
                view('reviewer_claude', '1d', 'UP', 0.65, 'DOWNGRADE'), view('consensus', '1d', 'UP', 0.55),
                view('analyst_claude', '5d', 'UP', 0.62),
              ],
            },
            timeline: [
              { run_id: 'r1', at: '2026-10-09T12:00:00Z', by_analyst: { analyst_claude: { view: 'UP', p_outperform: 0.6, verdict: 'KEEP', horizon: '1d' }, analyst_gpt: { view: 'DOWN', p_outperform: 0.45, verdict: 'KEEP', horizon: '1d' } } },
              { run_id: 'r2', at: '2026-10-10T12:00:00Z', by_analyst: { analyst_claude: { view: 'UP', p_outperform: 0.7, verdict: 'KEEP', horizon: '1d' }, analyst_gpt: { view: 'DOWN', p_outperform: 0.4, verdict: 'KEEP', horizon: '1d' } } },
            ],
          },
          outcomes: [{
            model_id: 'analyst_claude', run_id: 'r1', horizon: '1d', view: 'UP', realized_at: '2026-10-09T20:00:00Z',
            available_at: '2026-10-09T21:30:00Z', raw_return: 0.01, spy_relative_return: 0.005, net_unit_pnl: 0.004, cost_roundtrip: 0.002,
          }],
        },
        {
          symbol: 'DAL', role: 'input_cost', mechanism: 'Fuel compresses airline margins.', direction: 'down', priced_in: null,
          anticipation: { state: 'NOT_COVERED', latest: null, timeline: [] }, outcomes: [],
        },
      ],
      stories: [{ publisher: 'examplewire', url: 'https://news.example.invalid/oil', headline: 'Synthetic oil supply shock', published_at: '2026-10-10T09:00:00Z', received_at: '2026-10-10T09:05:00Z', primary: false }],
      provenance: { priced_in_status: 'MEASURED', independence: 'editorial_group', retail_hype: 0 },
    },
    {
      id: 'e2'.padEnd(64, '0'), rank: 2, headline: 'Synthetic chip export rule', link: null, source: null,
      published_at: null, available_at: '2026-10-10T10:00:00Z', novelty: 'repeat', themes: [], countries: [],
      badges: {
        event_importance: { value: 40, components: null },
        evidence_strength: { value: 25, status: 'single_source', publishers: 1, primary: false },
        model_conviction: { label: null, by_analyst: null, basis: 'none' },
        predictive_quality: { by_analyst: null, basis: 'scorecard' },
        after_cost_performance: { event_labels: null, paper_accounts: [], basis: 'labelled net_unit_pnl' },
      },
      what_changed: {
        expectations: 'Consensus unknown.', changed_expectations: null, summary: null, impact: null, horizon: null,
        priced_in: null, invalidation: null, source: 'rules_only',
      },
      assets: [], stories: [], provenance: { priced_in_status: 'UNKNOWN_PUBLICATION_TIME', independence: null, retail_hype: null },
    },
  ],
};

export const paperSnapshot: PaperSnapshot = {
  schema: 'cockpit-paper-v1', generated_at: '2026-10-10T17:10:00Z', report_at: '2026-10-10T17:00:00Z',
  accounts: [
    { account: 'ia_actions', suffix: 'AAAA', label: 'AI stocks', equity: 100000, pnl: 0, day_pnl: 0, peak: 100000, open_lots: 0,
      halted: false, return_since_start: 0, positions: [], journal: [] },
    { account: 'ia_crypto', suffix: 'BBBB', label: 'AI crypto', equity: 100500, pnl: 500, day_pnl: 20, peak: 100600, open_lots: 0,
      halted: false, return_since_start: 0.005, positions: [], journal: [] },
    {
      account: 'claude_book', suffix: 'CCCC', label: 'Claude book', equity: 107303.58, start_equity: 107292.42, cash: 88000,
      peak: 107303.58, return_since_start: 0.0001, tags: ['momo_v0', 'sleeve'],
      positions: [
        { symbol: 'BATUSD', asset_class: 'crypto', tag: 'momo_v0', qty: 19443.3, entry: 0.1378, last: 0.1397, unrealized_pl: 36.2,
          unrealized_pct: 0.0135, stop: 0.1124, target: 0.195, protection: 'policy', opened_at: '2026-10-10T17:07:55Z',
          reason: { event: 'breakout', mechanism: 'momentum', priced_in: 'no', scenario: 'trend', invalidation: 'close below stop' } },
        { symbol: 'BTCUSD', asset_class: 'crypto', tag: 'sleeve', qty: 0.04, entry: 82783.9, last: 82971.9, unrealized_pl: 8.2,
          unrealized_pct: 0.0022, stop: null, target: null, protection: 'none_defined', opened_at: null, reason: null },
      ],
      open_orders: [{ symbol: 'BTC/USD', side: 'buy', qty: 0.066, type: 'limit', limit: 80600, status: 'new', submitted: '2026-10-10T14:06', legs: [] }],
      journal: [{ at: '2026-10-10T17:07:55Z', action: 'buy_intent', symbol: 'BAT/USD', qty: 19492, limit: 0.13994, stop: 0.1124,
        target: 0.195, engine: 'momo_v0', reason: 'BAT breaks its 20-day high', mechanism: 'time-series momentum', invalidation: 'stop' }],
    },
  ],
  benchmarks: { SPY: { price: 778.5, return_since_paper_start: 0.0047, base_at: '2026-10-08T13:30:00Z' }, BTC: { price: 82000, at: '2026-10-10T11:31:00Z' } },
  curve: [
    { at: '2026-10-08T14:00:00Z', equity: { ia_actions: 100000, ia_crypto: 100000, claude_book: 107292 }, spy: 775, btc: 81000 },
    { at: '2026-10-09T14:00:00Z', equity: { ia_actions: 100000, ia_crypto: 100200, claude_book: 107350 }, spy: 777, btc: 82500 },
    { at: '2026-10-10T17:00:00Z', equity: { ia_actions: 100000, ia_crypto: 100500, claude_book: 107303 }, spy: 778.5, btc: 82000 },
  ],
  limitations: ['After-hours fills are optimistic.'], notes: ['paper only'],
};
