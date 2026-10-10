/**
 * The Radar home: the event, what it could change, the assets concerned, what our models anticipate and how
 * that anticipation moved. Beginner and Expert read the same snapshot; Expert adds provenance, availability
 * at T, versions and metrics. The page shows snapshot values verbatim and computes no score of its own.
 */
import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiClient, ApiError } from '../api/client';
import { carrySelection } from '../lib/cockpit';
import { useQuery } from '../state/useQuery';
import { useCockpit } from '../state/useCockpit';
import { EmptyState, ErrorState, Hash, LoadingState } from '../components/States';
import { EventCard } from '../components/radar/EventCard';
import { instant, signedNumber } from '../lib/radar';

const PAGE = 10;
const REGIME_ORDER = ['SPY', 'QQQ', 'VIX', 'BTC', 'ETH', 'Brent', 'gold', '10y_yield', 'dollar_index', 'EURUSD'];

export function RadarHomePage() {
  const { selection, params } = useCockpit();
  const carry = carrySelection(params);
  const expert = selection.mode === 'expert';
  const [covered, setCovered] = useState(false);
  const [shown, setShown] = useState(PAGE);
  const home = useQuery('radar-home', (signal) => apiClient.getRadarHome(signal), { staleMs: 60_000 });

  const events = useMemo(() => {
    const all = home.data?.events ?? [];
    return covered ? all.filter((event) => event.assets.some((a) => a.anticipation.state === 'COVERED')) : all;
  }, [home.data, covered]);

  if (home.status === 'loading') return <LoadingState label="Loading radar" />;
  if (home.status === 'error' && home.error) {
    const missing = home.error instanceof ApiError && [404, 503].includes(home.error.status);
    if (missing) {
      return (
        <EmptyState title="No radar snapshot is published yet"
          detail={<>Run <code>python -m scripts.radar.cockpit_export</code> and start the API with <code>--radar-root</code>.</>} />
      );
    }
    return <ErrorState error={home.error} onRetry={home.refetch} />;
  }
  const data = home.data;
  if (!data) return null;
  const regime = REGIME_ORDER.flatMap((key) => { const item = data.regime[key]; return item ? [item] : []; });

  return (
    <div className="stack radar-home">
      <section className="card radar-status" aria-label="Radar status">
        <div>
          <h1 className="radar-title">Radar {data.radar.date} · {data.radar.slot}</h1>
          <p className="muted">
            Data up to {instant(data.radar.cutoff)} · snapshot written {instant(data.generated_at)} ·{' '}
            {data.radar.shown_events} of {data.radar.total_events} events, ranked by importance ·{' '}
            {data.trader.runs} model runs, {data.trader.labels} realised labels
          </p>
          <p className="muted">
            No model run is a recommendation. A paper account is not real money, and the hypothesis state is{' '}
            <strong>{data.trader.hypothesis_state?.replace(/_/g, ' ').toLowerCase() ?? 'unknown'}</strong>.
          </p>
        </div>
        <ul className="regime" aria-label="Market regime">
          {regime.map((item) => (
            <li key={item.symbol}>
              <span>{item.symbol}</span>
              <span className="muted">{signedNumber(item.returns_pct['1d'])} % 1d</span>
            </li>
          ))}
        </ul>
        {expert && (
          <dl className="kv-list">
            <div><dt>Report</dt><dd><Hash value={data.radar.report_hash} chars={16} /></dd></div>
            <div><dt>Radar status</dt><dd>{data.radar.status}</dd></div>
            <div><dt>Sources</dt><dd>{Object.entries(data.radar.sources_by_status).map(([k, v]) => `${k} ${v}`).join(' · ')}</dd></div>
            <div><dt>Latest run</dt><dd>{data.trader.latest_run ?? '—'} · {instant(data.trader.latest_run_at)}</dd></div>
            <div><dt>Preregistration</dt><dd><Hash value={data.trader.preregistration_hash} chars={16} /></dd></div>
            <div>
              <dt>Model versions</dt>
              <dd>{Object.entries(data.trader.models).map(([k, v]) => `${k}: ${v.reported_version}`).join(' · ') || '—'}</dd>
            </div>
          </dl>
        )}
        {expert && data.radar.limitations.length > 0 && (
          <ul className="muted limitations">{data.radar.limitations.map((x) => <li key={x}>{x}</li>)}</ul>
        )}
      </section>

      <nav className="row journey" aria-label="Go deeper">
        {[['/events', 'Official events'], ['/markets', 'Markets'], ['/trader', 'Agent trader'], ['/paper', 'Paper accounts'],
          ['/lab', 'Lab'], ['/system', 'System']].map(([to, label]) => (
          <Link key={to} className="control" to={{ pathname: to, search: carry }}>{label}</Link>
        ))}
      </nav>

      <div className="row radar-filter">
        <label>
          <input type="checkbox" checked={covered} onChange={(e) => { setCovered(e.target.checked); setShown(PAGE); }} />{' '}
          Only events whose assets a model covers
        </label>
        <span className="muted">{events.length} events</span>
      </div>

      {events.length === 0
        ? <EmptyState title="No event matches" detail="Clear the filter to see every ranked event." />
        : events.slice(0, shown).map((event) => <EventCard key={event.id} event={event} expert={expert} />)}

      {events.length > shown && (
        <button className="control" onClick={() => setShown((value) => value + PAGE)}>
          Show {Math.min(PAGE, events.length - shown)} more
        </button>
      )}
    </div>
  );
}
