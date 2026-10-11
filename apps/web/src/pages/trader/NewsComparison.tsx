import { useState } from 'react';
import { Badge } from '../../components/States';
import type { NewsAbScorecard, NewsArmScore } from '../../api/traderTypes';
import { analystLabel } from '../../lib/trader';
import { pct } from './shared';

const brier = (v: number | null) => v === null ? 'n/a' : v.toFixed(3);
const cells = (score: NewsArmScore) => <>
  <td>{score.n}</td><td>{pct(score.hit_rate, 1)}</td><td>{brier(score.brier)}</td><td>{pct(score.abstention_rate, 1)}</td>
</>;

export function NewsComparison({ card }: { card: NewsAbScorecard }) {
  const choices = [...new Set(card.comparisons.map((r) => `${r.population}/${r.asset_class}/${r.horizon}`))];
  const [selected, setSelected] = useState<string | null>(null);
  const choice = selected && choices.includes(selected) ? selected : choices[0];
  const rows = card.comparisons.filter((r) => `${r.population}/${r.asset_class}/${r.horizon}` === choice);
  return <section className="card" aria-label="Base versus news arm">
    <div className="row" style={{ flexWrap: 'wrap', justifyContent: 'space-between' }}>
      <h2 className="card-title">Base vs news</h2><Badge tone="warn">NEWS · SHADOW ONLY</Badge>
      <label>Comparison <select className="control" value={choice ?? ''} onChange={(e) => setSelected(e.target.value)}>
        {choices.map((c) => <option key={c} value={c}>{c.replaceAll('/', ' · ')}</option>)}
      </select></label>
    </div>
    <p className="metric-sub">{card.complete_news_runs} completed news runs. Same frozen information time, universe and label windows.
      {' '}Only prospective paired sessions count; earlier base history is excluded.</p>
    <div className="table-scroll"><table className="data" aria-label="News A/B comparison">
      <caption className="metric-sub" style={{ textAlign: 'left' }}>Target: {rows[0]?.target ?? 'SPY_relative'}.
        {' '}n = realized directional views. Brier lower is better. Abstention is measured on realized views.</caption>
      <thead>
        <tr><th rowSpan={2}>Source</th><th colSpan={4}>Base</th><th colSpan={4}>News</th><th rowSpan={2}>Promotion</th></tr>
        <tr>{['Base', 'News'].flatMap((arm) => ['n', 'Hit rate', 'Brier', 'Abstention'].map((name) => <th key={`${arm}/${name}`}>{name}</th>))}</tr>
      </thead>
      <tbody>{rows.map((r) => <tr key={r.role}>
        <td>{analystLabel(r.role)}</td>{cells(r.base)}{cells(r.news)}
        <td>{r.state === 'TOO_EARLY' ? 'Too early' : r.state === 'ELIGIBLE_FOR_OPERATOR_REVIEW' ? 'Operator review required' :
          r.state === 'DESCRIPTIVE_ONLY_NO_SPY_TARGET' ? 'Descriptive only (no SPY target)' : 'Not eligible'}</td>
      </tr>)}</tbody>
    </table></div>
    <p className="metric-sub">Promotion requires ≥ {card.promotion_rule.minimum_directional_per_arm} realized directional views in each arm,
      a strictly higher SPY-relative hit rate and a strictly lower Brier for news, compared within each source, population and horizon.
      {' '}{card.promotion_rule.execution}</p>
  </section>;
}
