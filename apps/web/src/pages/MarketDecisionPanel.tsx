import { useState } from 'react';
import { apiClient } from '../api/client';
import type { OverlayKind } from '../api/analysisTypes';
import { AnalysisState } from '../components/AnalysisState';
import { DecisionCard } from '../components/DecisionCard';
import { MarketOverlayChart } from '../components/MarketOverlayChart';
import { marketAssets, marketSelection, overlayLabels } from '../lib/marketOverlays';
import { analystLabel, instant, signedPct } from '../lib/radar';
import { useQuery } from '../state/useQuery';

const KINDS = Object.keys(overlayLabels) as OverlayKind[];
export function MarketDecisionPanel() {
  const query = useQuery('radar-analysis', (signal) => apiClient.getRadarAnalysis(signal), { staleMs: 60_000 });
  const [chosen, setChosen] = useState('');
  const [horizon, setHorizon] = useState('1d');
  const [days, setDays] = useState(7);
  const [analyst, setAnalyst] = useState('all');
  const [kinds, setKinds] = useState<OverlayKind[]>(KINDS);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const data = query.data;
  if (!data) return <section className="card"><h2 className="card-title">Marchés et décisions</h2>
    <AnalysisState status={query.status} error={query.error} retry={query.refetch} /></section>;
  const assets = marketAssets(data);
  const asset = chosen || (assets.includes('SPY') ? 'SPY' : assets[0]) || '';
  const filtered = marketSelection(data, asset, horizon, days, kinds);
  const selection = { ...filtered, overlays: filtered.overlays.filter((o) => analyst === 'all' || !o.analyst || o.analyst === analyst) };
  const analysts = [...new Set(data.overlays.filter((o) => o.kind === 'decision').map((o) => o.analyst).filter((a): a is string => Boolean(a)))];
  const selected = selection.overlays.find((o) => o.id === selectedId) ?? [...selection.overlays].reverse().find((o) => o.kind === 'decision') ?? selection.overlays.at(-1);
  return <section className="card stack" aria-label="Marchés et décisions">
    <h1 className="card-title">Marchés et décisions</h1>
    <p className="muted">Snapshot {instant(data.generated_at)}. Choisir un repère pour lire sa chaîne de décision.</p>
    <div className="observation-controls">
      <label>Actif <select className="control" value={asset} onChange={(e) => setChosen(e.target.value)}>
        {assets.map((a) => <option key={a}>{a}</option>)}</select></label>
      <label>Horizon <select className="control" value={horizon} onChange={(e) => setHorizon(e.target.value)}>
        <option value="1d">1 session</option><option value="5d">5 sessions</option><option value="all">Tous</option></select></label>
      <label>Période <select className="control" value={days} onChange={(e) => setDays(Number(e.target.value))}>
        <option value={7}>7 jours</option><option value={30}>30 jours</option><option value={0}>Tout l’historique</option></select></label>
      <label>Analyste <select className="control" value={analyst} onChange={(e) => setAnalyst(e.target.value)}>
        <option value="all">Tous</option>{analysts.map((a) => <option key={a} value={a}>{analystLabel(a)}</option>)}</select></label>
    </div>
    <ul className="overlay-legend" aria-label="Calques">{KINDS.map((kind) => <li key={kind}>
      <label><input type="checkbox" checked={kinds.includes(kind)} onChange={(e) => setKinds((v) => e.target.checked ? [...v, kind] : v.filter((k) => k !== kind))} /> {overlayLabels[kind]}</label>
    </li>)}</ul>
    <MarketOverlayChart asset={asset} prices={selection.prices} overlays={selection.overlays} selected={selected?.id ?? null} onSelect={setSelectedId} />
    {selected && <div aria-live="polite">
      <h3 className="card-title">{overlayLabels[selected.kind]} · {instant(selected.at)} · {selected.label}</h3>
      {selected.analyst && <p>{analystLabel(selected.analyst)} · {selected.horizon} · reviewer {selected.verdict ?? 'sans objet'}</p>}
      {selected.price_basis === 'intent_limit' && <p className="metric-sub">Prix limite prévu : {selected.price}. Stop prévu : {selected.stop ?? 'non défini'}.</p>}
      {selected.kind === 'outcome' ? <p>Résultat réalisé net / unité : {signedPct(selected.net_return)}. Coûts modélisés.</p> : <DecisionCard decision={selected.decision} />}
    </div>}
    <details><summary>{selection.overlays.length} repères : liste accessible</summary>
      <div className="overlay-list">{selection.overlays.map((o) => <button className="control" key={o.id} aria-pressed={selected?.id === o.id}
        onClick={() => setSelectedId(o.id)}>{instant(o.at)} · {overlayLabels[o.kind]} · {o.analyst ? analystLabel(o.analyst) : ''} {o.label} {o.verdict ?? ''}</button>)}</div>
    </details>
    <p className="metric-sub">{data.limitations[0]}</p>
  </section>;
}
