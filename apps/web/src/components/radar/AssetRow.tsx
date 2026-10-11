/** One concerned asset: the mechanism, what each model anticipates, how that moved, and what was realised. */
import type { RadarAsset } from '../../api/radarTypes';
import {
  analystLabel, horizonsOf, instant, pct, signedNumber, signedPct, viewWords, viewsForHorizon,
} from '../../lib/radar';
import { AnticipationTimeline } from './AnticipationTimeline';
import { DecisionCard } from '../DecisionCard';

const DIRECTION: Record<string, string> = {
  up: 'beneficiary', down: 'loser', uncertain: 'direction to confirm', mixed: 'mixed',
};

export function AssetRow({ asset, expert }: { asset: RadarAsset; expert: boolean }) {
  const latest = asset.anticipation.latest;
  const horizons = latest ? horizonsOf(latest.views) : [];
  const side = asset.direction ? DIRECTION[asset.direction] ?? asset.direction : null;
  return (
    <li className="asset" data-covered={asset.anticipation.state === 'COVERED'}>
      <div className="asset-head">
        <strong className="asset-symbol">{asset.symbol}</strong>
        {asset.role && <span className="chip">{asset.role.replace(/_/g, ' ')}</span>}
        {side && <span className="chip" data-direction={asset.direction}>{side}</span>}
      </div>
      {asset.mechanism && <p className="asset-mechanism">{asset.mechanism}</p>}

      {asset.priced_in && (
        <p className="muted">
          Already moved {signedNumber(asset.priced_in.return_pct)} %
          {asset.priced_in.move_atr !== null && <> ({signedNumber(asset.priced_in.move_atr)} ATR)</>}
          {' '}since publication{expert && <> · {instant(asset.priced_in.baseline_at)} → {instant(asset.priced_in.price_at)}</>}.
          {expert && asset.priced_in.note && <> {asset.priced_in.note}.</>}
        </p>
      )}

      {asset.anticipation.state === 'NOT_COVERED' || !latest ? (
        <p className="muted asset-uncovered">No model run covers this asset: no anticipation to show.</p>
      ) : (
        <div className="asset-models">
          {horizons.map((horizon) => (
            <table className="data models" key={horizon} aria-label={`Model views, ${horizon} horizon`}>
              <caption>{horizon} horizon · run {expert ? latest.run_id : instant(latest.at)}</caption>
              <thead>
                <tr><th>Model</th><th>Anticipates</th><th>P(outperform)</th><th>Review</th></tr>
              </thead>
              <tbody>
                {viewsForHorizon(latest.views, horizon).map((view) => (
                  <tr key={view.analyst} data-analyst={view.analyst}>
                    <td>{analystLabel(view.analyst)}</td>
                    <td>{viewWords(view.view)}</td>
                    <td>{pct(view.p_outperform)}</td>
                    <td>{view.verdict ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ))}
          {expert && viewsForHorizon(latest.views, '1d').filter((v) => v.reason).slice(0, 2).map((v) => (
            <p className="muted reason" key={v.analyst}>
              <strong>{analystLabel(v.analyst)}:</strong> {v.reason}{v.falsifier ? ` Falsifier: ${v.falsifier}` : ''}
            </p>
          ))}
          {latest.views.map((view) => {
            const outcome = asset.outcomes.find((o) => o.run_id === latest.run_id && o.horizon === view.horizon && o.model_id === view.analyst);
            const decision = view.decision;
            return <details className="decision-detail" key={`${view.analyst}-${view.horizon}`}>
              <summary>{analystLabel(view.analyst)} · {view.horizon} · {view.view} · reviewer {view.verdict ?? 'inconnu'} : chaîne de décision</summary>
              <DecisionCard decision={decision && outcome ? { ...decision, result: {
                pnl_after_costs: null, r_after_costs: null, net_return: outcome.net_unit_pnl, costs: outcome.cost_roundtrip,
                at: outcome.available_at, basis: 'modelled_roundtrip_cost',
              } } : decision} />
            </details>;
          })}
          <AnticipationTimeline timeline={asset.anticipation.timeline} expert={expert} />
        </div>
      )}

      {asset.outcomes.length > 0 ? (
        <table className="data outcomes" aria-label="Realised outcomes">
          <caption>Realised outcome</caption>
          <thead>
            <tr><th>Model</th><th>Horizon</th><th>Raw</th><th>vs SPY</th><th>After costs</th>{expert && <th>Available at</th>}</tr>
          </thead>
          <tbody>
            {asset.outcomes.map((o) => (
              <tr key={`${o.model_id}-${o.run_id}-${o.horizon}`}>
                <td>{analystLabel(o.model_id)}</td><td>{o.horizon}</td>
                <td>{signedPct(o.raw_return)}</td><td>{signedPct(o.spy_relative_return)}</td>
                <td>{signedPct(o.net_unit_pnl)}</td>
                {expert && <td>{instant(o.available_at)}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        asset.anticipation.state === 'COVERED' && <p className="muted">Not labelled yet: the horizon has not closed.</p>
      )}
    </li>
  );
}
