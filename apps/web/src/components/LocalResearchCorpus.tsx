/**
 * The local equity research corpus, as a panel.
 *
 * Three states, and the difference between them is the whole point:
 *
 * - AVAILABLE: a verified local snapshot, charted, labelled for exactly what
 *   it is.
 * - NOT_INSTALLED: the machine has the code and the fingerprint but not the
 *   data. That is not an error, and it must not read like one -- nothing
 *   failed, nothing timed out, and no provider is offline, because nothing
 *   was ever contacted. There is deliberately no button here: a page load may
 *   not start a download from an unofficial source.
 * - INVALID / CORRUPT: files exist and do not bind. Nothing is drawn. Falling
 *   back to the source would silently replace audited bytes with unaudited
 *   ones, which is the one thing a research corpus may never do.
 *
 * Labels are chosen against a list of words this data has not earned: LIVE,
 * REALTIME, OFFICIAL, LICENSED, SPLIT-ADJUSTED, TOTAL RETURN. Every one of
 * them would be a claim the corpus cannot support, and a chart that looks
 * like a terminal is exactly where someone would assume otherwise.
 */
import { useMemo, useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { CandleChart } from './CandleChart';
import { Badge, EmptyState, ErrorState, Hash, LoadingState } from './States';
import type { Instrument, ResearchCorpusStatus } from '../api/types';

/** Sessions, not calendar days: the exchange decides which days exist. */
const WINDOWS = [
  { id: '60', label: '60 sessions', sessions: 60 },
  { id: '120', label: '120 sessions', sessions: 120 },
  { id: '250', label: '250 sessions', sessions: 250 },
  { id: 'max', label: 'Full local corpus', sessions: 0 },
] as const;

type WindowId = (typeof WINDOWS)[number]['id'];

interface Props {
  instruments: Instrument[];
  corpus: ResearchCorpusStatus;
}

function StatusChips({ corpus }: { corpus: ResearchCorpusStatus }) {
  return (
    <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
      <Badge tone="ok">LOCAL RESEARCH CORPUS</Badge>
      <Badge tone="off">HISTORICAL</Badge>
      <Badge tone="off">RAW</Badge>
      <Badge tone="off">REGULAR SESSION</Badge>
      <Badge tone="warn">UNOFFICIAL SOURCE</Badge>
      <Badge tone="off">{corpus.timeframe === '1d' ? '1D' : String(corpus.timeframe)}</Badge>
    </div>
  );
}

function SourceNote() {
  return (
    <p className="muted" style={{ marginTop: 8 }}>
      Unofficial research source. Local historical snapshot. Not a live market
      feed.
    </p>
  );
}

export function LocalResearchCorpus({ instruments, corpus }: Props) {
  // Tolerant of a partial payload. A cockpit panel that throws takes the whole
  // page down, and "the corpus status was missing a field" is not a reason to
  // lose the markets view.
  const reasons = corpus.reasons ?? [];
  const diagnostics = corpus.instruments ?? [];
  const available = instruments.filter(
    (item) => item.research?.local_corpus_available,
  );
  const [selected, setSelected] = useState('');
  const [windowId, setWindowId] = useState<WindowId>('120');

  const instrumentId = selected || available[0]?.instrument_id || '';
  const windowSpec = WINDOWS.find((item) => item.id === windowId)!;

  // The server bounds the page; this asks for a window, never for "all".
  const limit = windowSpec.sessions === 0 ? 1000 : windowSpec.sessions;
  const key = corpus.available && instrumentId
    ? `research-bars:${instrumentId}:${windowId}`
    : null;
  const bars = useQuery(key, (signal) =>
    apiClient.getResearchBars(instrumentId, { limit }, signal),
  );

  // The most recent window, taken from the end of the corpus rather than from
  // the wall clock. This is historical data; "today" is not part of it.
  const series = useMemo(() => {
    const rows = bars.data?.bars ?? [];
    if (windowSpec.sessions === 0) return rows;
    return rows.slice(-windowSpec.sessions);
  }, [bars.data, windowSpec.sessions]);

  if (corpus.status === 'NOT_INSTALLED') {
    return (
      <section className="card">
        <h2 className="card-title">
          Local equity research corpus{' '}
          <Badge tone="off">NOT AVAILABLE</Badge>
        </h2>
        <EmptyState
          title="Local Yahoo research corpus not available on this machine."
          detail={
            <>
              The corpus is local, gitignored research data. This build ships
              its fingerprint, not its bars. Nothing was requested over the
              network and nothing will be: it is captured by an explicit
              command-line run.
              {corpus.corpus_content_hash && (
                <>
                  {' '}Expected corpus{' '}
                  <Hash value={corpus.corpus_content_hash} />.
                </>
              )}
            </>
          }
        />
      </section>
    );
  }

  if (!corpus.available) {
    return (
      <section className="card">
        <h2 className="card-title">
          Local equity research corpus{' '}
          <Badge tone="warn">LOCAL CORPUS INVALID</Badge>
        </h2>
        <p className="muted">
          Local files exist but do not match the committed fingerprint, so no
          chart is drawn from them. Nothing is fetched to replace them — an
          unverified substitute would be worse than an empty panel. Re-run the
          local verification to see which artefact moved.
        </p>
        {reasons.length > 0 && (
          <ul className="muted" data-testid="corpus-reasons">
            {reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        )}
        {diagnostics.length > 0 && (
          <table className="table" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Instrument</th>
                <th>Present</th>
                <th>Rows</th>
                <th>Binds</th>
              </tr>
            </thead>
            <tbody>
              {diagnostics.map((item) => (
                <tr key={item.instrument_id}>
                  <td>{item.instrument_id}</td>
                  <td>{item.present ? 'yes' : 'no'}</td>
                  <td>{item.rows}</td>
                  <td>{item.content_hash_matches ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="muted" style={{ marginTop: 8 }}>
          The corpus is atomic: one artefact that does not bind invalidates all
          four.
        </p>
      </section>
    );
  }

  return (
    <section className="card">
      <h2 className="card-title">Local equity research corpus</h2>
      <StatusChips corpus={corpus} />
      <SourceNote />

      <div className="row" style={{ marginTop: 12 }}>
        <label htmlFor="research-instrument" className="muted">
          Research instrument
        </label>
        <select
          id="research-instrument"
          className="select"
          value={instrumentId}
          onChange={(event) => setSelected(event.target.value)}
        >
          {available.map((item) => (
            <option key={item.instrument_id} value={item.instrument_id}>
              {item.symbol} — {item.display_name}
            </option>
          ))}
        </select>
        <label htmlFor="research-window" className="muted">
          Research window
        </label>
        <select
          id="research-window"
          className="select"
          value={windowId}
          onChange={(event) => setWindowId(event.target.value as WindowId)}
        >
          {WINDOWS.map((item) => (
            <option key={item.id} value={item.id}>
              {item.label}
            </option>
          ))}
        </select>
        <div className="topbar-spacer" />
        <span className="metric-sub">
          {corpus.rows_total?.toLocaleString()} bars ·{' '}
          {corpus.expected_sessions} sessions per instrument · 0 gaps
        </span>
      </div>

      {bars.status === 'loading' && <LoadingState label="Loading local bars" />}
      {bars.status === 'error' && bars.error && (
        <ErrorState error={bars.error} onRetry={bars.refetch} />
      )}
      {bars.data && series.length > 0 && (
        <>
          <CandleChart candles={series} />
          <p className="metric-sub" style={{ marginTop: 8 }}>
            {series.length} daily sessions · {bars.data.metadata.provider} ·{' '}
            {bars.data.metadata.adjustment} ·{' '}
            {bars.data.metadata.source_timeframe} · session{' '}
            {bars.data.metadata.session}. Non-trading days are not gaps: the
            axis follows exchange sessions, and closed days are absent rather
            than drawn as zero.
          </p>
        </>
      )}

      <p className="metric-sub" style={{ marginTop: 8 }}>
        Corpus {corpus.corpus_id} · <Hash value={corpus.corpus_content_hash} /> ·
        spec <Hash value={corpus.corpus_spec_hash} /> · calendar{' '}
        <Hash value={corpus.calendar_spec_hash} />
      </p>
      <p className="muted">
        Research only. No prediction, no backtest, no paper session and no
        order exists for these instruments, and none of them is tradable in
        this build.
      </p>
    </section>
  );
}
