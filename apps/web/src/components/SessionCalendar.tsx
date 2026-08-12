/** Real trading sessions for one instrument, as the server computed them.
 *
 *  Every number here arrives finished: which days are sessions, when each
 *  opens and closes in UTC, how long it runs, how many bars it holds. None of
 *  it is derived in the browser.
 *
 *  That is not fussiness about layering. The session rules are holidays, early
 *  closes and two daylight-saving shifts a year, and a second implementation
 *  of them in TypeScript would agree with Python right up until one of those
 *  moved -- at which point the page would show a schedule the backend does not
 *  believe in, and nothing would report an error. So the browser formats and
 *  never computes.
 *
 *  A market with no session boundaries is not given a fabricated row per day.
 *  It says it is continuous, which is the true answer. */
import { useMemo, useState } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import { EmptyState, ErrorState, LoadingState } from './States';
import type { TradingSessionView } from '../api/types';

/** Windows are chosen here; the server enforces its own ceiling regardless. */
const WINDOWS = [
  { id: '2w', label: 'Next 2 weeks', days: 14 },
  { id: '1m', label: 'Next month', days: 30 },
  { id: '3m', label: 'Next quarter', days: 90 },
] as const;

function day(offset: number): string {
  const moment = new Date();
  moment.setUTCDate(moment.getUTCDate() + offset);
  return moment.toISOString().slice(0, 10);
}

function clock(value: string): string {
  return value.slice(11, 16);
}

function hours(seconds: number): string {
  const whole = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return minutes ? `${whole}h${String(minutes).padStart(2, '0')}` : `${whole}h`;
}

export function SessionCalendar({ instrumentId }: { instrumentId: string }) {
  const [window, setWindow] = useState<(typeof WINDOWS)[number]['id']>('1m');
  const span = WINDOWS.find((item) => item.id === window)!;

  const range = useMemo(
    () => ({ start: day(0), end: day(span.days) }),
    [span.days],
  );

  const sessions = useQuery(
    `sessions:${instrumentId}:${window}`,
    (signal) => apiClient.getInstrumentSessions(instrumentId, range, signal),
    { staleMs: 300_000 },
  );

  if (sessions.status === 'loading') {
    return <LoadingState label="Loading sessions" />;
  }
  if (sessions.status === 'error' && sessions.error) {
    return <ErrorState error={sessions.error} onRetry={sessions.refetch} />;
  }
  if (!sessions.data) return null;

  const data = sessions.data;

  if (data.continuous) {
    return (
      <EmptyState
        title="Continuous market"
        detail={
          <>
            {data.calendar.description}. There are no sessions to list: this
            market has no open, no close and no holidays, so a row per day
            would be invented rather than reported.
          </>
        }
      />
    );
  }

  const early = data.sessions.filter((item) => item.early_close);

  return (
    <div className="stack">
      <div className="row">
        <label htmlFor="session-window" className="muted">Window</label>
        <select
          id="session-window"
          className="select"
          value={window}
          onChange={(event) => setWindow(event.target.value as typeof window)}
        >
          {WINDOWS.map((item) => (
            <option key={item.id} value={item.id}>{item.label}</option>
          ))}
        </select>
        <div className="topbar-spacer" />
        <span className="metric-sub">
          {data.session_count} sessions · {early.length} early close
          {early.length === 1 ? '' : 's'} · {data.timeframe} grid
        </span>
      </div>

      <p className="muted">
        Weekends and holidays are absent because no bar is expected on them.
        They are not gaps in the data, and an early close simply holds fewer
        bars than a full session rather than being padded to match one.
      </p>

      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>Session</th>
              <th>Open (UTC)</th>
              <th>Close (UTC)</th>
              <th>Length</th>
              <th>Bars</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {data.sessions.map((session: TradingSessionView) => (
              <tr key={session.session_date}>
                <td>{session.session_date}</td>
                <td>{clock(session.open_at)}</td>
                <td>{clock(session.close_at)}</td>
                <td>{hours(session.duration_seconds)}</td>
                <td>{session.expected_bars}</td>
                <td>
                  {session.early_close ? (
                    <span className="badge" data-tone="warn">EARLY CLOSE</span>
                  ) : (
                    <span className="muted">regular</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.sessions.length === 0 && (
        <EmptyState
          title="No sessions in this window"
          detail="The market is shut for the whole of the selected range."
        />
      )}
    </div>
  );
}
