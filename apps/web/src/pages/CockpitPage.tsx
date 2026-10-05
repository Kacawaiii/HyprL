/** The cockpit: one selection (product, period, model) read in two ways. The selection lives in the
 *  URL, so switching Beginner/Expert -- or navigating to a detail page and back -- keeps it. */
import { carrySelection } from '../lib/cockpit';
import { useCockpit } from '../state/useCockpit';
import { ErrorState, LoadingState } from '../components/States';
import { BeginnerView } from './cockpit/BeginnerView';
import { ExpertView } from './cockpit/ExpertView';
import { useCockpitData } from './cockpit/useCockpitData';

export function CockpitPage() {
  const { selection, update, params } = useCockpit();
  const data = useCockpitData(selection);
  const products = data.overview.data?.products.map((item) => item.product) ?? [];
  const model = data.model;
  const startValue = selection.start ?? data.period?.from.slice(0, 10) ?? '';
  const endValue = selection.end
    ?? (data.period ? new Date(Date.parse(data.period.to) - 1000).toISOString().slice(0, 10) : '');

  if (data.overview.status === 'loading') return <LoadingState label="Loading cockpit" />;
  if (data.overview.status === 'error' && data.overview.error) {
    return <ErrorState error={data.overview.error} onRetry={data.overview.refetch} />;
  }
  const preset = (days: number) => {
    const last = data.period ? new Date(Date.parse(data.period.to) - 1000) : null;
    if (!last) return;
    const first = new Date(last.getTime() - (days - 1) * 86_400_000);
    update({ start: first.toISOString().slice(0, 10), end: last.toISOString().slice(0, 10) });
  };

  return (
    <div className="stack">
      <form className="card cockpit-selection" aria-label="Selection" onSubmit={(event) => event.preventDefault()}>
        <label>Product{' '}
          <select className="control" value={data.product ?? ''} onChange={(event) => update({ product: event.target.value })}>
            {products.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <label>Model{' '}
          <select className="control" value={selection.model ?? model?.id ?? ''} onChange={(event) => update({ model: event.target.value })}>
            {model ? <option value={model.id}>{model.id} (frozen reference)</option> : <option value="">unavailable</option>}
          </select>
        </label>
        <label>From{' '}
          <input className="control" type="date" value={startValue}
            onChange={(event) => event.target.value && update({ start: event.target.value, end: selection.end ?? endValue })} />
        </label>
        <label>To{' '}
          <input className="control" type="date" value={endValue}
            onChange={(event) => event.target.value && update({ end: event.target.value, start: selection.start ?? startValue })} />
        </label>
        <span className="row" role="group" aria-label="Period presets">
          <button type="button" className="control" onClick={() => preset(1)}>1 day</button>
          <button type="button" className="control" onClick={() => preset(7)}>7 days</button>
          <button type="button" className="control" onClick={() => preset(30)}>30 days</button>
        </span>
        <span className="muted">
          {data.period?.source === 'default' ? 'Default period: latest week of the signal run.' : 'Selected period.'}
        </span>
      </form>
      {selection.mode === 'expert'
        ? <ExpertView data={data} carry={carrySelection(params)} />
        : <BeginnerView data={data} carry={carrySelection(params)} />}
    </div>
  );
}
