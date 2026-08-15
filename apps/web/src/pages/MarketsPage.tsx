import { useMemo, useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { CandleChart } from '../components/CandleChart';
import { DataTable, type Column } from '../components/DataTable';
import { ErrorState, Hash, LoadingState } from '../components/States';
import {
  InstrumentDetails, InstrumentSelector, findInstrument, useInstruments,
} from '../components/InstrumentSelector';
import { SessionCalendar } from '../components/SessionCalendar';
import { LocalResearchCorpus } from '../components/LocalResearchCorpus';
import type { Candle, Instrument } from '../api/types';

/** Windows are chosen in the UI; the server decides how many points come back. */
const WINDOWS = [
  { id: '7d', label: '7 days', hours: 24 * 7 },
  { id: '30d', label: '30 days', hours: 24 * 30 },
  { id: '90d', label: '90 days', hours: 24 * 90 },
  { id: 'all', label: 'Full corpus', hours: 0 },
] as const;

export function MarketsPage() {
  // Empty until the registry answers. Seeding this with a literal symbol
  // would be a hardcoded product list of length one, and it would be wrong
  // the moment the registry no longer starts with that market.
  const [selected, setSelected] = useState('');
  const [window, setWindow] = useState<(typeof WINDOWS)[number]['id']>('30d');

  const instruments = useInstruments();
  const tradable = useMemo(
    () => (instruments.data?.instruments ?? []).filter((item) => item.tradable),
    [instruments.data],
  );
  // The catalogue holds markets this build describes but does not trade. They
  // get their own section rather than being mixed into the picker, which
  // drives charts and backtests that cannot run for them.
  const reference = useMemo(
    () => (instruments.data?.instruments ?? []).filter((item) => !item.tradable),
    [instruments.data],
  );
  const [referenceId, setReferenceId] = useState('');
  const referenceInstrument =
    reference.find((item) => item.instrument_id === referenceId) ?? reference[0];

  const product = selected || tradable[0]?.legacy_product_id || '';
  const instrument = findInstrument(instruments.data?.instruments, product);

  const markets = useQuery('markets', (signal) => apiClient.getMarkets(signal));
  const entry = markets.data?.products.find((item) => item.product === product);

  // Asked for unconditionally, because "is it installed" is itself the answer
  // the panel renders. It is a local filesystem question; nothing is fetched
  // from a provider to answer it.
  const corpus = useQuery('research-corpus', (signal) =>
    apiClient.getResearchCorpus(signal),
  );

  // The window is derived from the corpus end, not from the wall clock: this
  // is historical data, and "now" has nothing to do with it.
  const range = useMemo(() => {
    if (!entry) return undefined;
    const selected = WINDOWS.find((item) => item.id === window)!;
    if (selected.hours === 0) return { start: undefined, end: undefined };
    const end = new Date(entry.last_open);
    const start = new Date(end.getTime() - selected.hours * 3600_000);
    return { start: start.toISOString(), end: end.toISOString() };
  }, [entry, window]);

  const chartKey = entry && product ? `chart:${product}:${window}` : null;
  const chart = useQuery(chartKey, (signal) =>
    apiClient.getChart(product, { start: range?.start, end: range?.end, maxPoints: 500 }, signal),
  );

  const tableKey = entry && product ? `candles:${product}:${window}` : null;
  const candles = useQuery(tableKey, (signal) =>
    apiClient.getCandles(product, { start: range?.start, end: range?.end, limit: 200 }, signal),
  );

  const columns: Column<Candle>[] = [
    { key: 'time', header: 'Opening (UTC)', render: (row) => row.bar_open_at.replace('T', ' ').slice(0, 16) },
    { key: 'open', header: 'Open', render: (row) => row.open },
    { key: 'high', header: 'High', render: (row) => row.high },
    { key: 'low', header: 'Low', render: (row) => row.low },
    { key: 'close', header: 'Close', render: (row) => row.close },
    { key: 'volume', header: 'Volume', render: (row) => Number(row.volume).toFixed(4) },
  ];

  if (markets.status === 'loading') return <LoadingState label="Loading markets" />;
  if (markets.status === 'error' && markets.error) {
    return <ErrorState error={markets.error} onRetry={markets.refetch} />;
  }

  return (
    <div className="stack">
      <div className="row">
        <InstrumentSelector
          id="product-select"
          label="Instrument"
          value={product}
          onChange={setSelected}
        />
        <label htmlFor="window-select" className="muted">Window</label>
        <select
          id="window-select"
          className="select"
          value={window}
          onChange={(event) => setWindow(event.target.value as typeof window)}
        >
          {WINDOWS.map((item) => (
            <option key={item.id} value={item.id}>{item.label}</option>
          ))}
        </select>
        <div className="topbar-spacer" />
        {entry && (
          <span className="metric-sub">
            {entry.rows.toLocaleString()} bars · {entry.missing_openings} gaps
          </span>
        )}
      </div>

      {instrument && (
        <section className="card">
          <h2 className="card-title">Instrument</h2>
          <InstrumentDetails instrument={instrument} />
        </section>
      )}

      {instrument && !entry && markets.status === 'success' && (
        <section className="card">
          <h2 className="card-title">No market history</h2>
          <p className="muted">
            {instrument.symbol} is a registered instrument, but the committed
            corpus holds no bars for it. Nothing is wrong with the app; there is
            simply no captured history to show.
          </p>
        </section>
      )}

      <section className="card">
        <h2 className="card-title">{product} · hourly</h2>
        {chart.status === 'loading' && <LoadingState label="Loading chart" />}
        {chart.status === 'error' && chart.error && (
          <ErrorState error={chart.error} onRetry={chart.refetch} />
        )}
        {chart.data && (
          <>
            <CandleChart candles={chart.data.series} />
            <p className="metric-sub" style={{ marginTop: 8 }}>
              {chart.data.metadata.returned_count} points from{' '}
              {chart.data.metadata.source_count.toLocaleString()} source bars
              {chart.data.metadata.aggregated
                ? ` · aggregated ${chart.data.metadata.bucket_size}×1h buckets (OHLC preserved, not native 1h candles)`
                : ' · native 1h candles'}
            </p>
          </>
        )}
      </section>

      {corpus.data && (
        <LocalResearchCorpus instruments={reference} corpus={corpus.data} />
      )}

      {reference.length > 0 && (
        <section className="card">
          <h2 className="card-title">
            Reference markets{' '}
            <span className="badge" data-tone="off">NOT TRADED</span>
          </h2>
          <p className="muted">
            {reference.length} markets this build can describe but does not
            trade. They exist so their identity, their provider and their real
            trading sessions can be inspected. There is no model, no signal, no
            backtest and no paper session behind any of them, and none of them
            appears in the picker above.
          </p>
          <div className="row">
            <label htmlFor="reference-select" className="muted">Market</label>
            <select
              id="reference-select"
              className="select"
              value={referenceInstrument?.instrument_id ?? ''}
              onChange={(event) => setReferenceId(event.target.value)}
            >
              {reference.map((item: Instrument) => (
                <option key={item.instrument_id} value={item.instrument_id}>
                  {item.symbol} — {item.display_name}
                </option>
              ))}
            </select>
          </div>
          {referenceInstrument && (
            <>
              <div style={{ marginTop: 16 }}>
                <InstrumentDetails instrument={referenceInstrument} />
              </div>
              <h3 className="card-title" style={{ marginTop: 24 }}>
                Trading sessions
              </h3>
              <SessionCalendar instrumentId={referenceInstrument.instrument_id} />
            </>
          )}
        </section>
      )}

      <section className="card">
        <h2 className="card-title">Candles</h2>
        {candles.status === 'loading' && <LoadingState label="Loading candles" />}
        {candles.status === 'error' && candles.error && (
          <ErrorState error={candles.error} onRetry={candles.refetch} />
        )}
        {candles.data && (
          <>
            <DataTable rows={candles.data.candles} columns={columns} height={380} />
            <p className="metric-sub" style={{ marginTop: 8 }}>
              First {candles.data.page.returned} of the window
              {candles.data.page.has_more ? ' · more available via cursor' : ''}
            </p>
          </>
        )}
      </section>

      {markets.data?.corpus_content_hash && (
        <p className="metric-sub">
          Corpus {markets.data.corpus_id} · <Hash value={markets.data.corpus_content_hash} />
        </p>
      )}
    </div>
  );
}
