/** Several curves re-based to 100 at their own first point, so an account and two benchmarks share one scale. */
import type { RebasedSeries } from '../lib/radar';

const W = 640;
const H = 220;
const PAD = { left: 40, right: 12, top: 12, bottom: 22 };

export function RebasedChart({ series }: { series: RebasedSeries[] }) {
  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return <p className="muted">No equity history recorded yet: the curve starts with the first export.</p>;
  const times = all.map((p) => Date.parse(p.at));
  const t0 = Math.min(...times);
  const t1 = Math.max(...times);
  const values = all.map((p) => p.value).concat(100);
  const lo = Math.min(...values) - 0.2;
  const hi = Math.max(...values) + 0.2;
  const x = (at: string) => (t1 === t0 ? W / 2 : PAD.left + ((Date.parse(at) - t0) / (t1 - t0)) * (W - PAD.left - PAD.right));
  const y = (v: number) => H - PAD.bottom - ((v - lo) / (hi - lo)) * (H - PAD.top - PAD.bottom);
  return (
    <figure className="rebased">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Equity re-based to 100 against SPY and BTC">
        <line x1={PAD.left} x2={W - PAD.right} y1={y(100)} y2={y(100)} className="spark-mid" />
        <text x={4} y={y(100) + 4} className="axis">100</text>
        <text x={4} y={y(hi)+10} className="axis">{hi.toFixed(1)}</text>
        <text x={4} y={y(lo)} className="axis">{lo.toFixed(1)}</text>
        {series.map((s, index) => (
          <g key={s.key} className="rebased-series" data-series={s.key} data-slot={index % 6}>
            {s.points.length > 1
              ? <polyline fill="none" points={s.points.map((p) => `${x(p.at)},${y(p.value)}`).join(' ')} />
              : <circle cx={x(s.points[0]?.at ?? '')} cy={y(s.points[0]?.value ?? 100)} r={3.5} />}
          </g>
        ))}
      </svg>
      <figcaption>
        {Math.max(...series.map((s) => s.points.length)) < 2 && (
          <p className="muted">One recorded point so far: each export adds one, so the curves fill in over time.</p>
        )}
        <ul className="spark-legend">
          {series.map((s, index) => (
            <li key={s.key} data-slot={index % 6}>
              <span className="swatch" aria-hidden="true" />
              <strong>{s.label}</strong> {(s.points[s.points.length - 1]?.value ?? 100).toFixed(2)} ({s.points.length} point{s.points.length > 1 ? 's' : ''})
            </li>
          ))}
        </ul>
      </figcaption>
    </figure>
  );
}
