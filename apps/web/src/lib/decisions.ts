import type { DecisionChain, DecisionSource, Verification } from '../api/analysisTypes';
import type { DecisionView } from '../api/traderTypes';
import type { LedgerRow } from './trader';

const PRIMARY = ['sec.gov', 'federalreserve.gov', 'bls.gov', 'fcc.gov'];
export function verification(url: string | null, primary?: boolean): Verification {
  if (!url) return 'NON VERIFIE';
  try {
    const parsed = new URL(url);
    if (!['https:', 'http:'].includes(parsed.protocol) || parsed.username || parsed.password) return 'NON VERIFIE';
    return primary === true || primary !== false && PRIMARY.some((h) => parsed.hostname === h || parsed.hostname.endsWith(`.${h}`)) ? 'OFFICIEL' : 'FIL';
  } catch { return 'NON VERIFIE'; }
}
export function blankDecision(): DecisionChain {
  return { fact: null, sources: [], verification: 'NON VERIFIE', expectations: null, expectation_at: null,
    priced_in: null, scenario: null, entry_condition: null, invalidation: null, result: null };
}
export function traderDecision(view: DecisionView, row?: LedgerRow, entry?: string): DecisionChain {
  const raw = view.raw_view;
  const refs: DecisionSource[] = (raw?.catalysts ?? []).map((c) => ({ url: c.url, publisher: null,
    published_at: c.published_at, verification: verification(c.url) }));
  const label = row?.label;
  return { ...blankDecision(), fact: raw?.catalysts.map((c) => c.fact).join('; ') || null,
    sources: refs, verification: refs[0]?.verification ?? 'NON VERIFIE',
    priced_in: raw?.priced_in_assessment ?? null, scenario: raw?.confidence_reason ?? null,
    entry_condition: entry ?? (row ? `Entrée proposée ${row.prediction.payload.proposed_position.entry_at} · poids ${row.prediction.payload.proposed_position.weight}` : null),
    invalidation: raw?.falsifier ?? null,
    result: label ? { pnl_after_costs: null, r_after_costs: null, net_return: label.value.net_unit_pnl,
      costs: label.value.cost_roundtrip, at: label.available_at, basis: 'modelled_roundtrip_cost' } : null };
}
