import { useState } from 'react';
import { apiClient } from '../api/client';
import type { PaperReplayProduct } from '../api/types';
import { useQuery } from '../state/useQuery';
import { EmptyState, ErrorState, Hash, LoadingState } from '../components/States';
import { LineChart } from '../components/LineChart';
import { Num } from '../components/Num';
import type { NumKind } from '../lib/format';

function ReplayProduct({ data }: { data: PaperReplayProduct }) {
  const [cursors, setCursors] = useState<Array<string | undefined>>([undefined]);
  const cursor = cursors[cursors.length - 1];
  const equity = useQuery(`paper-replay-equity-${data.product}`, (signal) =>
    apiClient.getPaperReplayEquity(data.product, signal));
  const fills = useQuery(`paper-replay-fills-${data.product}-${cursor ?? 'first'}`, (signal) =>
    apiClient.getPaperReplayFills(data.product, cursor, signal));
  const quality = data.prediction_quality;
  const metrics = data.metrics;
  const values: Array<[string, string | number | null, NumKind | 'count']> = [
    ['Rank IC', quality.rank_ic, 'ratio'], ['MAE', quality.mae, 'ratio'], ['RMSE', quality.rmse, 'ratio'],
    ['Labels scored', quality.observations, 'count'], ['Predictions', quality.predictions, 'count'],
    ['Unscored predictions', quality.unscored_predictions, 'count'], ['Fills', data.counts.fills, 'count'],
    ['Initial equity (USD)', metrics.initial_equity, 'money'], ['Final equity (USD)', metrics.final_equity, 'money'],
    ['Net return', metrics.net_return, 'percent'], ['Max drawdown', metrics.max_drawdown, 'percent'],
    ['Sharpe annualisé', metrics.annualized_sharpe, 'ratio'], ['Fees (USD)', metrics.total_fees, 'money'],
    ['Slippage (USD)', metrics.total_slippage_cost, 'money'], ['Total costs (USD)', metrics.total_execution_cost, 'money'],
  ];
  return (
    <article className="card" aria-label={`Replay ${data.product}`}>
      <h3 className="card-title">{data.product}</h3>
      <dl style={{ margin: 0 }}>
        {values.map(([label, value, kind]) => (
          <div className="kv" key={label}><dt>{label}</dt><dd style={{ overflowWrap: 'anywhere' }}>{kind === 'count' ? (value ?? 'undefined') : <Num value={value} kind={kind} />}</dd></div>
        ))}
      </dl>
      <p className="muted">Signaux / cibles (LONG, FLAT, SHORT)</p>
      <dl>
        {['LONG', 'FLAT', 'SHORT'].map((side) => (
          <div className="kv" key={side}><dt>{side}</dt><dd>{data.counts.signals[side]} / {data.counts.targets[side]}</dd></div>
        ))}
      </dl>
      <p className="muted">
        {data.counts.bars} bars · {data.counts.gaps} gaps · {data.counts.expired_targets} expired targets ·{' '}
        {data.counts.warmup_without_prediction} warm-up bars · {data.counts.pending_terminal_targets} terminal target pending.
      </p>
      <p className="muted">{quality.scoring_policy}</p>
      {equity.status === 'loading' && <LoadingState label="Loading replay equity" />}
      {equity.error && <ErrorState error={equity.error} onRetry={equity.refetch} />}
      {equity.data && <>
        <LineChart label={`Replay equity ${data.product}`}
          points={equity.data.series.map((row) => ({ timestamp: row.available_at, value: Number(row.equity) }))}
          baseline={Number(metrics.initial_equity)} />
        <p className="metric-sub">{equity.data.page.total} points conservés sur {equity.data.metadata.source_count} · extrema conservés.</p>
      </>}
      <details>
        <summary>Fills du replay</summary>
        {fills.status === 'loading' && <LoadingState label="Loading replay fills" />}
        {fills.error && <ErrorState error={fills.error} onRetry={fills.refetch} />}
        {fills.data && <>
          <div style={{ overflowX: 'auto' }}>
            <table className="data">
              <thead><tr><th>Fill open</th><th>Observed at</th><th>Side</th><th>Quantity Δ</th><th>Fee</th><th>Slippage</th></tr></thead>
              <tbody>{fills.data.fills.map((fill) => <tr key={fill.timestamp}>
                <td>{fill.timestamp}</td><td>{fill.available_at}</td><td>{fill.side}</td>
                <td>{fill.quantity_delta}</td><td>{fill.fee}</td><td>{fill.slippage_cost}</td>
              </tr>)}</tbody>
            </table>
          </div>
          <p>{fills.data.page.returned} / {fills.data.page.total} fills · page {cursors.length}</p>
          <button disabled={cursors.length === 1} onClick={() => setCursors((old) => old.slice(0, -1))}>Précédent</button>{' '}
          <button disabled={!fills.data.page.has_more} onClick={() => {
            const next = fills.data?.page.next_cursor;
            if (next) setCursors((old) => [...old, next]);
          }}>Suivant</button>
        </>}
      </details>
      <details>
        <summary>Provenance du replay</summary>
        <dl>{Object.entries({ ...data.hashes, result_hash: data.result_hash }).map(([label, value]) => (
          <div className="kv" key={label}><dt>{label}</dt><dd style={{ overflowWrap: 'anywhere', minWidth: 0 }}><Hash value={value} chars={64} /></dd></div>
        ))}</dl>
      </details>
    </article>
  );
}

export function PaperReplaySection() {
  const replay = useQuery('paper-replay-v2', (signal) => apiClient.getPaperReplay(signal));
  const data = replay.data;
  return (
    <section className="stack" aria-label="Replay hors échantillon (v2)">
      <div className="card">
        <h2 className="card-title">Replay hors échantillon (v2)</h2>
        <p>Une seule fenêtre hors échantillon, non confirmatoire, non optimisée. Comptes indépendants du replay, séparés de la session shadow live.</p>
        {replay.status === 'loading' && <LoadingState label="Loading offline replay" />}
        {replay.error && <ErrorState error={replay.error} onRetry={replay.refetch} />}
        {data && !data.available && <EmptyState title="Replay indisponible" detail={data.reason} />}
        {data?.available && <>
          <p>{data.window?.start} → {data.window?.end}. Entraînement jusqu’au 2026-04-30T23:00:00Z.</p>
          <p>Déterminisme vérifié : {String(data.determinism?.verified)} · {data.determinism?.replay_count} replays identiques.</p>
          <p className="muted">Les valeurs ci-dessous sont celles du résultat figé. Rendement et drawdown sont affichés en pourcentage (valeur exacte au survol) ; Sharpe descriptif, annualisé sur 8760 heures.</p>
          <ul>{data.limitations?.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
        </>}
      </div>
      {data?.available && <div className="grid grid-2">
        {data.products.map((product) => <ReplayProduct key={product.product} data={product} />)}
      </div>}
    </section>
  );
}
