import { DecisionCard } from '../../components/DecisionCard';
import type { RunSummary } from '../../api/traderTypes';
import type { LedgerRow } from '../../lib/trader';
import { analystLabel } from '../../lib/trader';
import { traderDecision } from '../../lib/decisions';

export function TraderDecisionCards({ run, rows, asset }: { run: RunSummary; rows: LedgerRow[]; asset: string | null }) {
  return <section className="card stack" aria-label="Décisions IA sourcées">
    <h2 className="card-title">Du fait à la décision</h2>
    {(run.decision?.views ?? []).filter((v) => !asset || v.asset === asset).map((v) => {
      const row = rows.find((r) => r.prediction.payload.signal.run_id === run.run_id &&
        r.prediction.payload.product === v.asset && r.prediction.payload.signal.view.analyst === v.analyst &&
        r.prediction.payload.signal.label_definition.horizon === v.horizon);
      return <details key={`${v.analyst}-${v.asset}-${v.horizon}`} className="decision-detail" open={v.analyst === 'consensus'}>
        <summary>{analystLabel(v.analyst)} · {v.asset} · {v.horizon} · {v.view} · reviewer {v.verdict}</summary>
        <DecisionCard decision={traderDecision(v, row, run.portfolios?.[v.horizon]?.entry)} />
      </details>;
    })}
  </section>;
}
