/**
 * One chart, five layers on one time axis: close price, the model's projection, decisions,
 * executions and events. SVG rather than canvas because every mark must be reachable by keyboard
 * and screen reader; the series is bounded (one period, at most MAX_PERIOD_BARS bars).
 */
import { useMemo, useRef, useState } from 'react';
import { layoutTimeline, timelineStops } from '../lib/timeline';
import type { TimelineInput } from '../lib/timeline';

const WIDTH = 900;
const HEIGHT = 300;

export function CockpitChart({ input, showProjection }: { input: TimelineInput; showProjection: boolean }) {
  const layout = useMemo(() => layoutTimeline(
    showProjection ? input : { ...input, projections: [] }, { width: WIDTH, height: HEIGHT }),
  [input, showProjection]);
  const stops = useMemo(() => timelineStops(layout), [layout]);
  const [cursor, setCursor] = useState<number | null>(null);
  const frame = useRef<HTMLDivElement>(null);
  const { left, right, top, bottom } = layout.box;

  function onKey(event: React.KeyboardEvent) {
    if (stops.length === 0) return;
    const keys: Record<string, (current: number | null) => number> = {
      ArrowRight: (current) => Math.min(stops.length - 1, (current ?? -1) + 1),
      ArrowLeft: (current) => Math.max(0, (current ?? stops.length) - 1),
      Home: () => 0,
      End: () => stops.length - 1,
    };
    const move = keys[event.key];
    if (!move) return;
    event.preventDefault();
    setCursor(move);
  }

  const path = layout.price.map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(' ');
  const focused = cursor === null ? null : stops[cursor];
  return (
    <div ref={frame} className="cockpit-chart">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" tabIndex={0} onKeyDown={onKey}
        aria-label={`Timeline: ${layout.price.length} closes, ${layout.decisions.length} decisions, ${layout.fills.length} executions, ${layout.events.length} events. Arrow keys step through marks.`}
        aria-describedby="cockpit-chart-readout"
      >
        <line x1={left} x2={left} y1={top} y2={bottom} className="axis" />
        <line x1={left} x2={right} y1={bottom} y2={bottom} className="axis" />
        <text x={4} y={top + 8} className="tick">{layout.high.toFixed(0)}</text>
        <text x={4} y={bottom} className="tick">{layout.low.toFixed(0)}</text>
        <text x={left} y={HEIGHT - 6} className="tick">{input.period.from.slice(0, 10)}</text>
        <text x={right} y={HEIGHT - 6} textAnchor="end" className="tick">{input.period.to.slice(0, 10)}</text>
        {layout.events.map(({ x, item }, index) => (
          <line key={`e${index}`} x1={x} x2={x} y1={top} y2={bottom} className="mark-event">
            <title>{`${item.kind}: ${item.label}`}</title>
          </line>
        ))}
        <path d={path} className="mark-price" fill="none" />
        {layout.projections.map(({ from, to, projection }) => (
          <g key={projection.decisionAt} className="mark-projection">
            <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} />
            <circle cx={from.x} cy={from.y} r={2} />
            <circle cx={to.x} cy={to.y} r={2.5} className="projection-end" />
            <title>{`${projection.decisionAt}: ${projection.formula}; reference ${projection.referencePrice}`}</title>
          </g>
        ))}
        {layout.decisions.map(({ item, x, y }) => (
          <path key={item.timestamp} className={`mark-decision ${item.direction.toLowerCase()}`}
            d={item.direction === 'LONG' ? `M${x} ${y - 7} l4 8 h-8 z`
              : item.direction === 'SHORT' ? `M${x} ${y + 7} l4 -8 h-8 z`
                : `M${x - 4} ${y} h8`}>
            <title>{`${item.direction} · strength ${item.strength}`}</title>
          </path>
        ))}
        {layout.fills.map(({ item, x, y }, index) => (
          <rect key={`f${index}`} className="mark-fill" x={x - 3} y={y - 3} width={6} height={6}
            transform={`rotate(45 ${x} ${y})`}>
            <title>{`Execution ${item.side} ${item.quantity_delta}`}</title>
          </rect>
        ))}
      </svg>
      <ul className="legend" aria-label="Chart legend">
        <li><span className="swatch price" />Close</li>
        {showProjection && <li><span className="swatch projection" />Model projection (reference → projected)</li>}
        <li><span className="swatch decision" />Decision (▲ long, ▼ short, – flat)</li>
        <li><span className="swatch fill" />Execution (paper replay)</li>
        <li><span className="swatch event" />Event (declared time)</li>
      </ul>
      <p id="cockpit-chart-readout" className="metric-sub" role="status" aria-live="polite">
        {focused ? `${focused.at.slice(0, 16).replace('T', ' ')} · ${focused.text}` : `${stops.length} marks. Focus the chart and use the arrow keys.`}
      </p>
    </div>
  );
}
