import { apiClient } from '../api/client';
import { ErrorState, LoadingState } from '../components/States';
import { AI_TABS, PageTabs } from '../components/Simple';
import { useQuery } from '../state/useQuery';

const money = (n: number) => n.toLocaleString('fr-FR', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });

export function AgentV2Page() {
  const query = useQuery('agent-v2-summary', (signal) => apiClient.getAgentV2Summary(signal));
  const data = query.data;
  return <div className="stack">
    <PageTabs label="IA" links={AI_TABS} />
    <section className="card">
      <h1>Agent v2 · Shadow</h1>
      <p>Décisions prospectives, exécutions simulées avec frais. Registre indépendant de v1.</p>
      <p>Aucun ordre courtier. Le passage au papier dépend du résultat du replay et d’une décision de l’opérateur.</p>
      <button className="control" onClick={query.refetch}>Actualiser</button>
    </section>
    {query.status === 'loading' && <LoadingState />}
    {query.error && <ErrorState error={query.error} onRetry={query.refetch} />}
    {data && <>
      <section className="card" aria-label="État du shadow">
        <h2>{data.mode === 'SYNTHETIC' ? 'Données synthétiques · test technique' : 'Suivi shadow prospectif'}</h2>
        <p>{data.state === 'NO_QUOTES' ? 'Aucune quote reçue : suivi non démarré.' : data.state === 'STALE' ? 'Dernière quote périmée : valorisation à vérifier.' : 'Dernière valorisation disponible.'}</p>
        {data.last_mark && <dl>
          <div className="kv"><dt>Valorisation simulée</dt><dd>{money(data.last_mark.equity)}</dd></div>
          <div className="kv"><dt>Risque aux stops</dt><dd>{data.last_mark.heat === null ? 'Indisponible' : `${(data.last_mark.heat * 100).toFixed(2)} % / 4 %`}</dd></div>
          <div className="kv"><dt>Gouverneur</dt><dd>{data.last_mark.governor}</dd></div>
          <div className="kv"><dt>Dernière observation</dt><dd><time>{data.last_mark.at}</time></dd></div>
        </dl>}
        <p>{data.pending} plans en attente d’une quote postérieure à l’approbation.</p>
        {data.last_blocked && <p role="status">Dernier blocage : {data.last_blocked.reason} · {data.last_blocked.at}</p>}
      </section>
      <section className="card">
        <h2>Scorecards par couche</h2>
        <p>Résultats simulés, après frais. Aucune efficacité établie.</p>
        <div style={{ overflowX: 'auto' }}><table>
          <thead><tr><th>Couche</th><th>Clôturées</th><th>Gagnantes / total</th><th>Cluster-jours</th><th>P&amp;L net</th><th>Espérance R</th></tr></thead>
          <tbody>{(['core', 'news'] as const).map((layer) => {
            const row = data.layers[layer];
            return <tr key={layer}><th>{layer === 'core' ? 'Systématique' : 'Événements primaires'}</th>
              <td>{row.closed}</td><td>{row.wins} / {row.closed}</td><td>{row.cluster_days}</td>
              <td>{row.closed ? money(row.pnl) : 'Pas de résultat'}</td><td>{row.expectancy_r === null ? 'Non mesurée' : row.expectancy_r.toFixed(3)}</td></tr>;
          })}</tbody>
        </table></div>
      </section>
      <section className="card"><h2>Veto du superviseur</h2>
        <p>{data.supervisor_vetoes.count} veto. Effet sur les résultats : non mesuré.</p>
        <ul>{Object.entries(data.supervisor_vetoes.codes).map(([code, count]) => <li key={code}>{code} : {count}</li>)}</ul>
      </section>
      <section className="card"><h2>Positions simulées</h2>
        {data.positions.length === 0 ? <p>Aucune position.</p> : <div style={{ overflowX: 'auto' }}><table>
          <thead><tr><th>Instrument</th><th>Playbook</th><th>Cluster</th><th>Quantité</th><th>Entrée</th><th>Stop</th></tr></thead>
          <tbody>{data.positions.map((p) => <tr key={p.symbol}><th>{p.symbol}</th><td>{p.playbook}</td><td>{p.cluster}</td><td>{p.qty}</td><td>{money(p.entry)}</td><td>{money(p.stop)}</td></tr>)}</tbody>
        </table></div>}
      </section>
      <details className="card"><summary>Configuration figée et intégrité</summary>
        <p>Code : <code>{data.code_sha}</code></p><p>Préenregistrement : <code>{data.preregistration_hash}</code></p>
        <p>Spécification : <code>{data.spec_hash}</code></p><p>Registre : {data.events} événements · <code>{data.ledger_hash}</code></p>
      </details>
    </>}
  </div>;
}
