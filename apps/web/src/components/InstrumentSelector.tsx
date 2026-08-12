/** One market picker, shared by every page that needs one.
 *
 *  Not a `<select>` copied into Markets, Research, Backtests and Paper. Four
 *  copies means four places to add an asset class, and the fourth is always
 *  the one nobody updates -- so a page quietly keeps offering a market the
 *  registry no longer has, or misses one it gained.
 *
 *  The option list comes from `/api/v1/instruments`. The frontend holds no
 *  product list of its own: an array of symbol literals is a second registry,
 *  and a second registry is a disagreement waiting for a third instrument.
 *
 *  Options are grouped by asset class through <optgroup>, so equities land
 *  under their own heading.
 *
 *  But only tradable ones are offered. The catalogue describes six markets
 *  and this build trades two; the equities exist so their sessions and
 *  provider can be inspected, and offering AAPL in a picker that drives a
 *  backtest would promise a run that cannot happen. `tradableOnly` defaults
 *  to true for exactly that reason -- a caller has to ask for the wider list. */
import { useMemo } from 'react';
import { apiClient } from '../api/client';
import { useQuery } from '../state/useQuery';
import type { Instrument } from '../api/types';

const GROUP_LABELS: Record<string, string> = {
  CRYPTO: 'Crypto',
  EQUITY: 'Equities',
  ETF: 'ETFs',
  INDEX: 'Indices',
  FX: 'FX',
};

export function useInstruments() {
  return useQuery('instruments', (signal) => apiClient.getInstruments(signal), {
    // The registry is fixed at build time; re-fetching it per navigation is
    // a request that can only ever return the same bytes.
    staleMs: 300_000,
  });
}

/** Look up one instrument by either identity, without a second fetch. */
export function findInstrument(
  instruments: Instrument[] | undefined,
  value: string,
): Instrument | undefined {
  if (!instruments) return undefined;
  return instruments.find(
    (item) => item.instrument_id === value || item.legacy_product_id === value,
  );
}

export function InstrumentSelector({
  value,
  onChange,
  id = 'instrument-select',
  label = 'Instrument',
  /** Address instruments by the id the rest of the API already uses. */
  by = 'legacy',
  /** Offer only what this build can actually trade. */
  tradableOnly = true,
}: {
  value: string;
  onChange: (next: string) => void;
  id?: string;
  label?: string;
  by?: 'legacy' | 'canonical';
  tradableOnly?: boolean;
}) {
  const { data, status } = useInstruments();

  const groups = useMemo(() => {
    const all = data?.asset_classes ?? [];
    if (!tradableOnly) return all;
    // Filtered, then emptied groups dropped: an "Equities" heading with no
    // options under it reads as a loading failure rather than as a boundary.
    return all
      .map((group) => ({
        ...group,
        instruments: group.instruments.filter((item) => item.tradable),
      }))
      .filter((group) => group.instruments.length > 0);
  }, [data, tradableOnly]);

  const key = (item: Instrument) =>
    by === 'canonical' ? item.instrument_id : (item.legacy_product_id ?? item.instrument_id);

  if (status === 'loading') {
    return (
      <>
        <label htmlFor={id} className="muted">{label}</label>
        <select id={id} className="select" disabled aria-busy="true">
          <option>Loading…</option>
        </select>
      </>
    );
  }

  if (status === 'error' || groups.length === 0) {
    // No invented fallback list. If the registry cannot be read, the honest
    // state is "no instruments", not two guesses that might be wrong.
    return (
      <>
        <label htmlFor={id} className="muted">{label}</label>
        <select id={id} className="select" disabled>
          <option>No instruments available</option>
        </select>
      </>
    );
  }

  return (
    <>
      <label htmlFor={id} className="muted">{label}</label>
      <select
        id={id}
        className="select"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {groups.map((group) => (
          <optgroup
            key={group.asset_class}
            label={GROUP_LABELS[group.asset_class] ?? group.asset_class}
          >
            {group.instruments.map((item) => (
              <option key={item.instrument_id} value={key(item)}>
                {item.symbol}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </>
  );
}

/** The identity panel: what this market is, and where its bars come from.
 *
 *  The calendar block is server-computed. Nothing here counts bars in a
 *  session or works out an annualisation factor; those are the session rules,
 *  they live in Python, and a browser reimplementing them would disagree with
 *  the backend the first time a holiday moved. */
export function InstrumentDetails({ instrument }: { instrument: Instrument }) {
  const providers = useQuery('providers', (signal) => apiClient.getProviders(signal), {
    staleMs: 300_000,
  });
  const detail = useQuery(
    `instrument:${instrument.instrument_id}`,
    (signal) => apiClient.getInstrument(instrument.instrument_id, signal),
    { staleMs: 300_000 },
  );
  const served = providers.data?.providers.filter((provider) =>
    instrument.providers.includes(provider.provider_id),
  );
  const calendar = detail.data?.calendar;

  return (
    <dl style={{ margin: 0 }}>
      <div className="kv"><dt>Instrument</dt><dd>{instrument.display_name}</dd></div>
      <div className="kv"><dt>Identifier</dt><dd><code>{instrument.instrument_id}</code></dd></div>
      <div className="kv"><dt>Asset class</dt><dd>{GROUP_LABELS[instrument.asset_class] ?? instrument.asset_class}</dd></div>
      <div className="kv">
        <dt>Venue</dt>
        <dd>
          {instrument.venue}{' '}
          <span className="muted">
            (where it trades, not where the bars come from)
          </span>
        </dd>
      </div>
      <div className="kv"><dt>Base</dt><dd>{instrument.base_asset}</dd></div>
      <div className="kv"><dt>Quote</dt><dd>{instrument.quote_asset}</dd></div>
      <div className="kv"><dt>Exchange time</dt><dd>{instrument.timezone}</dd></div>
      <div className="kv">
        <dt>Calendar</dt>
        <dd>
          <code>{instrument.trading_calendar}</code>
          {calendar?.available === false && (
            <> <span className="badge" data-tone="off">NOT INSTALLED</span></>
          )}
          {calendar?.description && (
            <div className="muted">{calendar.description}</div>
          )}
        </dd>
      </div>
      {calendar?.available && (
        <div className="kv">
          <dt>Sessions</dt>
          <dd>
            {calendar.bars_per_day === null ? (
              <span className="muted">
                {calendar.timeframe_note ??
                  `no whole number of ${calendar.timeframe} bars fits a session`}
              </span>
            ) : (
              <>
                {calendar.bars_per_day} × {calendar.timeframe} per session ·{' '}
                {calendar.annualization_periods?.toLocaleString()} periods a year
                <div className="muted">
                  The factor a Sharpe ratio is scaled by, from the calendar
                  rather than from a constant.
                </div>
              </>
            )}
          </dd>
        </div>
      )}
      {calendar?.spec && (
        <div className="kv">
          <dt>Calendar source</dt>
          <dd>
            <span className="muted">
              {calendar.spec.calendar_provider}{' '}
              {calendar.spec.calendar_provider_version} ·{' '}
              {calendar.spec.calendar_name} · {calendar.spec.session_type} session
            </span>
          </dd>
        </div>
      )}
      <div className="kv"><dt>Timeframes</dt><dd>{instrument.native_timeframes.join(', ')}</dd></div>
      <div className="kv">
        <dt>Status</dt>
        <dd>
          {instrument.tradable ? (
            <span className="badge" data-tone="ok">TRADABLE</span>
          ) : (
            <>
              <span className="badge" data-tone="off">REFERENCE ONLY</span>{' '}
              <span className="muted">
                described, not traded: no model, no signal, no paper session
              </span>
            </>
          )}
        </dd>
      </div>
      <div className="kv">
        <dt>Provider</dt>
        <dd>
          {served && served.length > 0
            ? served.map((provider) => provider.display_name).join(', ')
            : instrument.providers.join(', ') || '—'}
        </dd>
      </div>
      {served?.map((provider) => (
        <div className="kv" key={provider.provider_id}>
          <dt>Data</dt>
          <dd>
            <span
              className="badge"
              data-tone={provider.capabilities.authenticated ? 'warn' : 'ok'}
            >
              {provider.capabilities.authenticated
                ? 'KEYED MARKET DATA'
                : 'PUBLIC MARKET DATA'}
            </span>{' '}
            <span className="badge" data-tone="off">
              {provider.capabilities.data_freshness.replace(/_/g, ' ')}
            </span>{' '}
            <span className="muted">
              {provider.capabilities.historical_bars ? 'historical bars' : ''}
              {provider.capabilities.latest_closed_bar ? ' · latest closed bar' : ''}
              {provider.capabilities.corporate_actions ? ' · corporate actions' : ''}
              {provider.capabilities.realtime_ticks ? '' : ' · no ticks'}
              {provider.capabilities.order_book ? '' : ' · no order book'}
              {' · no account data'}
            </span>
          </dd>
        </div>
      ))}
    </dl>
  );
}
