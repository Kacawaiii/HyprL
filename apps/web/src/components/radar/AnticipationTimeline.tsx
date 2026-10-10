/** How the models' probability of outperformance moved from one run to the next. */
import type { TimelinePoint } from '../../api/radarTypes';
import { ANALYST_ORDER, analystLabel, analystSeries, evolution, instant, pct } from '../../lib/radar';

const W = 220;
const H = 56;
const PAD = 6;

export function AnticipationTimeline({ timeline, expert }: { timeline: TimelinePoint[]; expert: boolean }) {
  const analysts = ANALYST_ORDER.filter((name) => analystSeries(timeline, name).length > 0);
  if (analysts.length === 0) return <p className="muted">No run has covered this asset yet.</p>;
  const x = (index: number, total: number) => (total <= 1 ? W / 2 : PAD + (index / (total - 1)) * (W - 2 * PAD));
  const y = (p: number) => H - PAD - p * (H - 2 * PAD);
  const runs = timeline.map((point) => point.run_id);
  return (
    <div className="anticipation-timeline">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Probability of outperformance by run"
        className="spark">
        <line x1={PAD} x2={W - PAD} y1={y(0.5)} y2={y(0.5)} className="spark-mid" />
        {analysts.map((name) => {
          const series = analystSeries(timeline, name);
          const points = series.map((s) => `${x(runs.indexOf(s.run_id), runs.length)},${y(s.p)}`);
          return (
            <g key={name} data-analyst={name} className="spark-series">
              {points.length > 1 && <polyline points={points.join(' ')} fill="none" />}
              {series.map((s) => (
                <circle key={s.run_id} cx={x(runs.indexOf(s.run_id), runs.length)} cy={y(s.p)} r={2.6}>
                  <title>{`${analystLabel(name)} ${pct(s.p)} at ${instant(s.at)}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
      <ul className="spark-legend">
        {analysts.map((name) => (
          <li key={name} data-analyst={name}>
            <span className="swatch" aria-hidden="true" />
            <strong>{analystLabel(name)}</strong> {evolution(analystSeries(timeline, name))}
          </li>
        ))}
      </ul>
      {expert && (
        <p className="muted">
          Runs: {timeline.map((point) => instant(point.at)).join(' → ')}. The dashed line is 50 % (no edge).
        </p>
      )}
    </div>
  );
}
