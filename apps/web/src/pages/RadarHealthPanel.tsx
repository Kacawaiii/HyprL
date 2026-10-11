import { apiClient } from '../api/client';
import { AnalysisState } from '../components/AnalysisState';
import { Badge } from '../components/States';
import { instant } from '../lib/radar';
import { useQuery } from '../state/useQuery';

function stale(at: string | null, now: string) {
  return !at || !Number.isFinite(Date.parse(at)) || Date.parse(now) - Date.parse(at) > 6 * 3600_000;
}
export function RadarHealthPanel() {
  const query = useQuery('radar-analysis', (signal) => apiClient.getRadarAnalysis(signal), { staleMs: 60_000 });
  const data = query.data;
  if (!data) return <section className="card"><h2 className="card-title">Santé du radar et des unités</h2>
    <AnalysisState status={query.status} error={query.error} retry={query.refetch} /></section>;
  const system = data.system;
  return <section className="card stack" aria-label="Santé du radar et des unités">
    <h1 className="card-title">Santé du radar et des unités</h1>
    <p className="muted health-date">Export {instant(data.generated_at)} · rapport radar {instant(system.radar_at)}.
      L’état LIVE décrit le dernier contrôle; au-delà de 6 heures il est signalé périmé.</p>
    <div className="table-scroll"><table className="data" aria-label="Sources radar">
      <thead><tr><th>Source</th><th>Dernier état</th><th>Fraîcheur</th><th>Motif</th><th>Contrôlée</th><th>Items</th></tr></thead>
      <tbody>{system.sources.map((s) => <tr key={s.source}><td>{s.source}</td>
        <td><Badge tone={s.status === 'LIVE' ? 'ok' : 'warn'}>{s.status}</Badge></td>
        <td>{stale(s.checked_at, new Date().toISOString()) ? 'Périmé / inconnu' : 'Récent'}</td>
        <td>{s.reason ?? (s.status === 'LIVE' ? '—' : 'Motif non consigné')}</td><td>{instant(s.checked_at)}</td><td>{s.items ?? '—'}</td></tr>)}</tbody>
    </table></div>
    <div className="grid grid-2">
      <div><h2 className="card-title">Budgets radar utilisés · {system.budget_date}</h2>
        <dl className="kv-list">{Object.entries(system.budgets_used).map(([name, count]) => <div key={name}><dt>{name}</dt><dd>{count}</dd></div>)}</dl>
        {Object.keys(system.budgets_used).length === 0 && <p className="muted">Aucune réservation ce jour dans le ledger.</p>}
        <p className="metric-sub">Les plafonds d’autorisation privés ne sont pas publiés.</p></div>
      <div><h2 className="card-title">Trader</h2>
        <p>{system.trader.health?.state ?? 'Santé inconnue'} · {instant(system.trader.health?.at)}</p>
        <p>Dernier label : {system.trader.last_label?.state ?? 'inconnu'} · {instant(system.trader.last_label?.at)}</p>
        <p>Pause : {system.trader.paused === undefined ? 'inconnue' : system.trader.paused ? 'oui' : 'non'}</p>
        <p className="metric-sub">Appels réservés : {Object.entries(system.trader.health?.budget_counts ?? {}).map(([k, v]) => `${k} ${v}`).join(', ') || 'aucun dans le dernier contrôle'}.</p>
      </div>
    </div>
    <h2 className="card-title">Unités trader, radar, timers Claude book et export</h2>
    <div className="table-scroll"><table className="data" aria-label="Unités observées">
      <thead><tr><th>Unité</th><th>État</th><th>Résultat / code sortie</th><th>Dernier run</th><th>Prochain run</th></tr></thead>
      <tbody>{system.units.map((u) => <tr key={u.unit}><td>{u.unit}</td><td>{u.state ?? 'UNKNOWN'}</td><td>{u.result || '—'} / {u.exit_status ?? '—'}</td>
        <td>{u.last_run || 'inconnu'}</td><td>{u.next_run || 'sans objet / inconnu'}</td></tr>)}</tbody>
    </table></div>
    {system.units.length === 0 && <p className="muted">Statut des unités indisponible.</p>}
    <h2 className="card-title">Échecs récents</h2>
    {system.trader.failures?.length ? <ul>{system.trader.failures.slice(-15).reverse().map((f, i) => <li key={`${f.at}-${i}`}>{f.code} · {instant(f.at)}</li>)}</ul>
      : <p className="muted">Aucun échec dans les observations exportées; les logs bruts ne sont pas servis.</p>}
  </section>;
}
