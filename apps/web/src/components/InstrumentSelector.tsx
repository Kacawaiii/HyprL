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
 *  Options are grouped by asset class through <optgroup>, so the day
 *  equities arrive they land under their own heading with no change here.
 *  Today there is exactly one group, Crypto, holding the two real markets --
 *  no placeholder equity, no "coming soon" row. */
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
}: {
  value: string;
  onChange: (next: string) => void;
  id?: string;
  label?: string;
  by?: 'legacy' | 'canonical';
}) {
  const { data, status } = useInstruments();

  const groups = useMemo(() => data?.asset_classes ?? [], [data]);
  const key = (item: Instrument) =>
    by === 'canonical' ? item.instrument_id : item.legacy_product_id;

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

/** The identity panel: what this market is, and where its bars come from. */
export function InstrumentDetails({ instrument }: { instrument: Instrument }) {
  const providers = useQuery('providers', (signal) => apiClient.getProviders(signal), {
    staleMs: 300_000,
  });
  const served = providers.data?.providers.filter((provider) =>
    instrument.providers.includes(provider.provider_id),
  );

  return (
    <dl style={{ margin: 0 }}>
      <div className="kv"><dt>Instrument</dt><dd>{instrument.display_name}</dd></div>
      <div className="kv"><dt>Identifier</dt><dd><code>{instrument.instrument_id}</code></dd></div>
      <div className="kv"><dt>Asset class</dt><dd>{GROUP_LABELS[instrument.asset_class] ?? instrument.asset_class}</dd></div>
      <div className="kv"><dt>Venue</dt><dd>{instrument.venue}</dd></div>
      <div className="kv"><dt>Base</dt><dd>{instrument.base_asset}</dd></div>
      <div className="kv"><dt>Quote</dt><dd>{instrument.quote_asset}</dd></div>
      <div className="kv"><dt>Calendar</dt><dd>{instrument.trading_calendar}</dd></div>
      <div className="kv"><dt>Timeframes</dt><dd>{instrument.native_timeframes.join(', ')}</dd></div>
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
            <span className="badge" data-tone="ok">PUBLIC MARKET DATA</span>{' '}
            <span className="muted">
              {provider.capabilities.historical_bars ? 'historical bars' : ''}
              {provider.capabilities.latest_closed_bar ? ' · latest closed bar' : ''}
              {provider.capabilities.realtime_ticks ? '' : ' · no ticks'}
              {provider.capabilities.order_book ? '' : ' · no order book'}
            </span>
          </dd>
        </div>
      ))}
    </dl>
  );
}
