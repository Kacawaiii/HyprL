/**
 * Trader charts in plain SVG, each with a table alternative. One series per chart; direction is carried by
 * marker shape (triangle up / down, diamond for no view), never by colour alone. No animation.
 */
import type { AssetChart, ChartDecision } from '../lib/trader';
import type { CalibrationBin } from '../api/traderTypes';

const W = 640;
const H = 250;
const PAD = { left: 52, right: 14, top: 14, bottom: 26 };

function scale(values: number[], low: number, high: number) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return (v: number) => high - ((v - min) / span) * (high - low);
}

const day = (t: number) => new Date(t).toISOString().slice(0, 10);

function Marker({ d, x, y }: { d: ChartDecision; x: number; y: number }) {
  const r = 5;
  const shape = d.view === 'UP' ? `${x},${y - r} ${x - r},${y + r} ${x + r},${y + r}`
    : d.view === 'DOWN' ? `${x},${y + r} ${x - r},${y - r} ${x + r},${y - r}`
      : `${x},${y - r} ${x + r},${y} ${x},${y + r} ${x - r},${y}`;
  const result = d.realized ? 'realized' : 'pending';
  const text = `${day(d.at)} ${d.horizon} ${d.view} p=${d.p.toFixed(2)} — ${result}`;
  return (
    <polygon points={shape} tabIndex={0} aria-label={text}
      fill={d.realized ? 'var(--accent)' : 'var(--surface, transparent)'} stroke="var(--accent)" strokeWidth={1.5}>
      <title>{text}</title>
    </polygon>
  );
}

export function PriceChart({ chart }: { chart: AssetChart }) {
  const times = [...chart.line.map((p) => p.at), ...chart.decisions.map((d) => d.at), ...chart.outcomes.flatMap((o) => [o.from.at, o.to.at])];
  const prices = [...chart.line.map((p) => p.price), ...chart.outcomes.flatMap((o) => [o.from.price, o.to.price])];
  if (chart.line.length === 0 && chart.decisions.length === 0) {
    return <div className="state">No reference prices recorded for {chart.asset} yet.</div>;
  }
  const tMin = Math.min(...times);
  const tMax = Math.max(...times);
  const x = (t: number) => PAD.left + (tMax === tMin ? (W - PAD.left - PAD.right) / 2 : ((t - tMin) / (tMax - tMin)) * (W - PAD.left - PAD.right));
  const y = scale(prices, PAD.top, H - PAD.bottom);
  const pMin = Math.min(...prices);
  const pMax = Math.max(...prices);
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="group" aria-label={`${chart.asset}: reference prices, decisions and realized outcomes`}
        style={{ width: '100%', height: 'auto', maxHeight: 300 }}>
        <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} stroke="var(--border)" />
        <text x={PAD.left - 6} y={PAD.top + 4} textAnchor="end" fontSize="10" fill="var(--text-muted)">{pMax.toFixed(2)}</text>
        <text x={PAD.left - 6} y={H - PAD.bottom} textAnchor="end" fontSize="10" fill="var(--text-muted)">{pMin.toFixed(2)}</text>
        <text x={PAD.left} y={H - 8} fontSize="10" fill="var(--text-muted)">{day(tMin)}</text>
        <text x={W - PAD.right} y={H - 8} textAnchor="end" fontSize="10" fill="var(--text-muted)">{day(tMax)}</text>
        {chart.line.length > 1 && (
          <polyline fill="none" stroke="var(--text-muted)" strokeWidth={1.5}
            points={chart.line.map((p) => `${x(p.at)},${y(p.price)}`).join(' ')} />
        )}
        {chart.line.map((p) => <circle key={p.at} cx={x(p.at)} cy={y(p.price)} r={2.5} fill="var(--text-muted)"><title>{`${day(p.at)} reference close ${p.price.toFixed(2)}`}</title></circle>)}
        {chart.outcomes.map((o) => (
          <line key={`${o.from.at}-${o.horizon}`} x1={x(o.from.at)} y1={y(o.from.price)} x2={x(o.to.at)} y2={y(o.to.price)}
            stroke={o.net >= 0 ? 'var(--positive)' : 'var(--negative)'} strokeWidth={2} strokeDasharray={o.horizon === '5d' ? '5 3' : undefined}>
            <title>{`${o.horizon} outcome ${o.from.price.toFixed(2)} → ${o.to.price.toFixed(2)}; net of cost ${(o.net * 100).toFixed(3)} %`}</title>
          </line>
        ))}
        {chart.decisions.map((d) => <Marker key={`${d.at}-${d.horizon}`} d={d} x={x(d.at)} y={y(d.price) + (d.horizon === '5d' ? 12 : 0)} />)}
      </svg>
      <figcaption className="metric-sub">
        Grey: the last close known at each decision (reference price, not a full history). Triangles: consensus view at the decision
        (up / down; diamond = no view; hollow = label pending, filled = realized). Segments: realized entry → exit, green or red by
        result after costs, dashed for the 5-session horizon.
      </figcaption>
      <details>
        <summary>Table view</summary>
        <table className="data" aria-label={`${chart.asset} decisions and outcomes`}>
          <thead><tr><th>Day</th><th>Horizon</th><th>View</th><th>p</th><th>State</th></tr></thead>
          <tbody>
            {chart.decisions.map((d) => (
              <tr key={`${d.at}-${d.horizon}`}><td>{day(d.at)}</td><td>{d.horizon}</td><td>{d.view}</td><td>{d.p.toFixed(2)}</td>
                <td>{d.realized ? 'realized' : 'pending'}</td></tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}

const S = 220;
export function CalibrationChart({ bins, label }: { bins: CalibrationBin[]; label: string }) {
  const filled = bins.filter((b) => b.n > 0 && b.mean_p !== null && b.frequency !== null);
  const total = filled.reduce((sum, b) => sum + b.n, 0);
  const p = (v: number) => 30 + v * (S - 44);
  const q = (v: number) => S - 24 - v * (S - 44);
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${S} ${S}`} role="img" style={{ width: '100%', maxWidth: 320, height: 'auto' }}
        aria-label={`Calibration of ${label}: ${total} scored views in ${filled.length} bins`}>
        <rect x={p(0)} y={q(1)} width={p(1) - p(0)} height={q(0) - q(1)} fill="none" stroke="var(--border)" />
        <line x1={p(0)} y1={q(0)} x2={p(1)} y2={q(1)} stroke="var(--text-muted)" strokeDasharray="4 3" />
        <text x={p(0)} y={S - 8} fontSize="9" fill="var(--text-muted)">stated probability 0 → 1</text>
        <text x={2} y={q(1) + 8} fontSize="9" fill="var(--text-muted)">freq.</text>
        {filled.map((b) => (
          <g key={b.low}>
            <circle cx={p(b.mean_p as number)} cy={q(b.frequency as number)} r={5} fill="var(--accent)" stroke="var(--surface, #fff)" strokeWidth={2} tabIndex={0}
              aria-label={`bin ${b.low}–${Math.min(b.high, 1)}: stated ${(b.mean_p as number).toFixed(2)}, observed ${(b.frequency as number).toFixed(2)}, n=${b.n}`}>
              <title>{`stated ${(b.mean_p as number).toFixed(2)} · observed ${(b.frequency as number).toFixed(2)} · n=${b.n}`}</title>
            </circle>
            <text x={p(b.mean_p as number) + 7} y={q(b.frequency as number) + 3} fontSize="9" fill="var(--text-muted)">n={b.n}</text>
          </g>
        ))}
      </svg>
      <figcaption className="metric-sub">
        Dashed diagonal = perfectly calibrated. Each dot is one probability bin with its sample size. {total < 100
          ? `Only ${total} scored views: far too few to read as calibration. No calibration is claimed.` : 'No calibration is claimed without a registered method.'}
      </figcaption>
      <details>
        <summary>Table view</summary>
        <table className="data" aria-label="Calibration bins">
          <thead><tr><th>Bin</th><th>n</th><th>Mean stated p</th><th>Observed frequency</th></tr></thead>
          <tbody>{bins.map((b) => (
            <tr key={b.low}><td>{b.low}–{Math.min(b.high, 1)}</td><td>{b.n}</td><td>{b.mean_p === null ? '—' : b.mean_p.toFixed(3)}</td><td>{b.frequency === null ? '—' : b.frequency.toFixed(3)}</td></tr>
          ))}</tbody>
        </table>
      </details>
    </figure>
  );
}
