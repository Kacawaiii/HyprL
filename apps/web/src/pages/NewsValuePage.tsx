import { useState } from 'react';
import { apiClient } from '../api/client';
import { AnalysisState } from '../components/AnalysisState';
import { DecisionCard } from '../components/DecisionCard';
import { analystLabel, instant, pct, signedNumber, signedPct } from '../lib/radar';
import { useQuery } from '../state/useQuery';
import type { NewsGroup } from '../api/analysisTypes';

const GROUPS: Record<string, string> = { news: 'Actualité', technical: 'Technique', unclassified: 'Non classé' };
function NewsRows({ rows }: { rows: NewsGroup[] }) {
  return <div className="table-scroll"><table className="data" aria-label="Actualité versus technique après coûts">
    <thead><tr><th>Tag / groupe</th><th>Intentions</th><th>Clôtures liées</th><th>n net connu</th><th>Hit rate net</th><th>R moyen net</th><th>n R</th><th>P&amp;L net USD</th><th>Lecture</th></tr></thead>
    <tbody>{rows.map((g) => <tr key={g.group}><td>{GROUPS[g.group] ?? g.group}</td><td>{g.count}</td><td>{g.closed}</td><td>{g.scored}</td>
      <td>{pct(g.hit_rate, 1)}</td><td>{signedNumber(g.average_r)}</td><td>{g.r_samples}</td><td>{signedNumber(g.pnl_after_costs)}</td>
      <td>{g.state === 'TOO_EARLY' ? 'Trop tôt' : 'Descriptif seulement'}</td></tr>)}</tbody>
  </table></div>;
}
export function NewsValuePage() {
  const query = useQuery('radar-analysis', (signal) => apiClient.getRadarAnalysis(signal), { staleMs: 60_000 });
  const [population, setPopulation] = useState('equity_etf');
  const [horizon, setHorizon] = useState('1d');
  const [variant, setVariant] = useState('primary');
  const data = query.data;
  if (!data) return <AnalysisState status={query.status} error={query.error} retry={query.refetch} />;
  const keys = Object.keys(data.news.ai_scores);
  const variants = ['primary', ...new Set(keys.map((key) => key.split('/')[0]!).filter((name) => name.includes(':')).map((name) => name.slice(0, name.lastIndexOf(':'))))];
  const target = population === 'crypto' ? 'raw' : 'SPY_relative';
  const names = ['analyst_claude', 'analyst_gpt', 'reviewer_claude', 'reviewer_gpt', 'consensus', 'always_up', 'momentum20', 'random', 'spy_relative_zero'];
  const cohort = data.book_trades;
  return <div className="stack news-comparison">
    <section className="card">
      <h1 className="radar-title">Est-ce que l’actualité aide ?</h1>
      <p className="muted">Comparer les décisions après coûts, avec leurs effectifs. Snapshot {instant(data.generated_at)}.</p>
      <p>Un petit échantillon ne permet pas de conclure. Le seuil d’affichage de {data.news.minimum_n} résultats est une précaution descriptive, pas une preuve statistique.</p>
    </section>
    <section className="card stack" aria-label="Claude book par tag">
      <h2 className="card-title">Claude book : actualité et technique</h2>
      <NewsRows rows={data.news.groups} />
      <p className="metric-sub">Hit rate = part des P&amp;L nets strictement positifs. R = P&amp;L net / risque initial consigné.
        Les intentions et les clôtures sans coûts connus restent hors des métriques. Les catégories viennent des tags du journal; aucun tag news n’est inféré d’un titre.</p>
      {data.news.by_tag && <><h3 className="card-title">Détail par tag</h3><NewsRows rows={data.news.by_tag} /></>}
      <p className="metric-sub">{Object.entries(data.news.tag_counts).map(([tag, n]) => `${tag} : ${n}`).join(' · ') || 'Aucun tag enregistré.'}</p>
    </section>
    <section className="card stack" aria-label="IA avec ou sans actualité">
      <h2 className="card-title">IA : avec ou sans contexte d’actualité</h2>
      <p>Trop tôt : comparaison appariée indisponible.</p><p className="muted">{data.news.note}</p>
      <p className="metric-sub">Avec actualité : n apparié inconnu. Sans actualité : n apparié inconnu. Les variantes de calendrier ou d’exécution ne prouvent pas l’apport de l’actualité.</p>
    </section>
    <section className="card stack" aria-label="IA contre baselines">
      <h2 className="card-title">Scorecards IA contre baselines</h2>
      <div className="observation-controls">
        <label>Population <select className="control" value={population} onChange={(e) => setPopulation(e.target.value)}>
          <option value="equity_etf">Actions / ETF</option><option value="crypto">Crypto</option></select></label>
        <label>Horizon <select className="control" value={horizon} onChange={(e) => setHorizon(e.target.value)}>
          <option value="1d">1 session</option><option value="5d">5 sessions</option></select></label>
        <label>Variante <select className="control" value={variant} onChange={(e) => setVariant(e.target.value)}>
          {variants.map((v) => <option key={v}>{v}</option>)}</select></label>
      </div>
      <div className="table-scroll"><table className="data" aria-label="Scorecards et baselines avec effectifs">
        <caption className="metric-sub">Cible {target}. Même population, horizon et variante; coûts modélisés. Aucune métrique R sans risque initial défini.</caption>
        <thead><tr><th>Source</th><th>Émises</th><th>Réalisées</th><th>n actives</th><th>Jours</th><th>Hit rate</th><th>Net moyen / unité</th><th>Lecture</th></tr></thead>
        <tbody>{names.map((name) => {
          const scoreName = name === 'random' ? 'random_seeded' : name;
          const key = `${variant === 'primary' ? '' : `${variant}:`}${scoreName}/${population}/${horizon}/${target}`;
          const score = data.news.ai_scores[key];
          return <tr key={name}><td>{analystLabel(name)}</td><td>{score?.issued ?? '—'}</td><td>{score?.realized ?? '—'}</td>
            <td>{score?.non_abstained ?? '—'}</td><td>{score?.days ?? '—'}</td><td>{pct(score?.hit_rate, 1)}</td>
            <td>{signedPct(score?.mean_unit_pnl_after_costs, 3)}</td><td>{!score ? 'Indisponible' : score.non_abstained < data.news.minimum_n || score.days < 30 ? 'Trop tôt' : 'Descriptif seulement'}</td></tr>;
        })}</tbody>
      </table></div>
      <p className="metric-sub">Les effectifs peuvent différer à cause de l’abstention. Cette table ne démontre ni causalité ni avantage. Les cohortes chevauchantes ne sont pas indépendantes.</p>
    </section>
    <section className="card stack"><h2 className="card-title">Décisions du journal</h2>
      {cohort.length === 0 && <p>Aucune intention consignée.</p>}
      {[...cohort].reverse().slice(0, 30).map((t) => <details className="decision-detail" key={t.id}>
        <summary>{t.symbol} · {t.tag} · {instant(t.at)} · {t.state}</summary><DecisionCard decision={t.decision} /></details>)}
    </section>
  </div>;
}
