import type { MarketObservation, MarketOverlay, OverlayKind } from '../api/analysisTypes';
import { instant, signedPct } from '../lib/radar';
import { overlayLabels } from '../lib/marketOverlays';
import { useEffect, useRef, useState } from 'react';

const KINDS: OverlayKind[] = ['event', 'decision', 'book_entry', 'book_exit', 'outcome'];
const COLOR: Record<OverlayKind, string> = { event: 'var(--warning)', decision: 'var(--accent)', book_entry: 'var(--text)',
  book_exit: 'var(--text-muted)', outcome: 'var(--positive)' };
const H = 370, LEFT = 66, RIGHT = 18, TOP = 18, PRICE_END = 198;
export function MarketOverlayChart({ asset, prices, overlays, selected, onSelect }: {
  asset: string; prices: MarketObservation[]; overlays: MarketOverlay[]; selected: string | null; onSelect: (id: string) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [W, setWidth] = useState(900);
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const update = () => setWidth(Math.max(300, svg.getBoundingClientRect().width || 900));
    update();
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(update);
      observer.observe(svg);
      return () => observer.disconnect();
    }
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, [prices.length, overlays.length]);
  const dates = [...prices.map((p) => Date.parse(p.at)), ...overlays.map((o) => Date.parse(o.at!))];
  if (dates.length === 0) return <p className="muted">Aucun prix ni événement observé pour cette sélection.</p>;
  const first = Math.min(...dates), last = Math.max(...dates);
  const x = (at: string) => LEFT + (last === first ? .5 : (Date.parse(at) - first) / (last - first)) * (W - LEFT - RIGHT);
  const levels = [...prices.map((p) => p.price), ...overlays.flatMap((o) => o.stop !== null && o.stop !== undefined ? [o.stop] : [])];
  const low = Math.min(...levels), high = Math.max(...levels);
  const y = (value: number) => PRICE_END - (high === low ? .5 : (value - low) / (high - low)) * (PRICE_END - TOP);
  return <figure className="observation-chart">
    <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} role="group" aria-label={`${asset} : prix et événements de décision`}>
      {[TOP, PRICE_END].map((yy, i) => <g key={yy}>
        <line x1={LEFT} x2={W - RIGHT} y1={yy} y2={yy} stroke="var(--border)" />
        {prices.length > 0 && <text x={LEFT - 8} y={yy + 4} textAnchor="end">{(i ? low : high).toLocaleString('en-US', { maximumFractionDigits: 2 })}</text>}
      </g>)}
      {prices.length === 0 && <text x={LEFT} y={90}>Aucun prix observé</text>}
      {prices.length > 1 && <polyline points={prices.map((p) => `${x(p.at)},${y(p.price)}`).join(' ')} fill="none"
        stroke="var(--text-muted)" strokeDasharray="4 4" strokeWidth={1.5} />}
      {prices.map((p, i) => <circle key={`${p.at}-${i}`} cx={x(p.at)} cy={y(p.price)} r={3} fill="var(--text)">
        <title>{instant(p.at)} · {p.price} · {p.basis}</title></circle>)}
      {overlays.filter((o) => o.stop !== null && o.stop !== undefined).map((o) => <line key={`stop-${o.id}`} x1={x(o.at!)} x2={W - RIGHT}
        y1={y(o.stop!)} y2={y(o.stop!)} stroke="var(--negative)" strokeDasharray="6 4"><title>Stop prévu Claude : {o.stop}; exécution non attestée</title></line>)}
      {KINDS.map((kind, index) => <g key={kind}>
        <line x1={LEFT} x2={W - RIGHT} y1={226 + index * 24} y2={226 + index * 24} stroke="var(--border)" />
        <text x={LEFT - 8} y={230 + index * 24} textAnchor="end">{['Radar', 'IA', 'Entrée', 'Sortie', 'Net'][index]}</text>
      </g>)}
      {overlays.map((o, index) => {
        const xx = x(o.at!), yy = 226 + KINDS.indexOf(o.kind) * 24;
        const label = `${instant(o.at)} · ${overlayLabels[o.kind]} · ${o.analyst ?? ''} ${o.label} ${o.verdict ?? ''}${o.kind === 'outcome' ? ` · net ${signedPct(o.net_return)}` : ''}`;
        return <g key={o.id}>
          {selected === o.id && <line x1={xx} x2={xx} y1={TOP} y2={yy} stroke="var(--accent)" strokeDasharray="2 4" />}
          <path d={o.kind === 'decision' ? o.label === 'DOWN' ? `M ${xx} ${yy + 5} l 5 -10 h -10 Z`
            : o.label === 'ABSTAIN' ? `M ${xx} ${yy - 5} l 5 5 l -5 5 l -5 -5 Z` : `M ${xx} ${yy - 5} l 5 10 h -10 Z` : o.kind === 'event'
            ? `M ${xx} ${yy - 5} l 5 5 l -5 5 l -5 -5 Z` : `M ${xx - 4} ${yy - 4} h 8 v 8 h -8 Z`}
            fill={o.net_return !== undefined && o.net_return !== null && o.net_return < 0 ? 'var(--negative)' : COLOR[o.kind]}
            stroke={selected === o.id ? 'var(--text)' : 'var(--surface)'} strokeWidth={selected === o.id ? 2 : 1}
            role="button" tabIndex={0} aria-label={label} aria-pressed={selected === o.id}
            onClick={() => onSelect(o.id)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(o.id); } }}>
            <title>{label} · #{index + 1}</title>
          </path>
        </g>;
      })}
      <text x={LEFT} y={H - 8}>{new Date(first).toISOString().slice(0, 10)}</text>
      <text x={W - RIGHT} y={H - 8} textAnchor="end">{new Date(last).toISOString().slice(0, 10)}</text>
    </svg>
    <figcaption className="metric-sub">Points : prix observés. Pointillés : liaison visuelle entre observations espacées.
      Losange : radar reçu. Triangle : décision IA. Carrés : intentions/sorties Claude et résultats. Ligne rouge : stop prévu.
      Les événements occupent leur propre ligne; leur hauteur n’est pas un prix d’exécution.</figcaption>
  </figure>;
}
