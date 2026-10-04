/** The point-in-time read of an event store: an instant (ISO-8601 with its offset) and, optionally, a
 *  commit horizon. The browser checks only that the horizon is a whole number; the server validates the
 *  read and decides what was available. */
import { useState } from 'react';
import { Badge } from './States';

export interface Read {
  asOf: string;
  horizon?: number;
}

export function readKey(read: Read): string {
  return `${read.asOf}|${read.horizon ?? ''}`;
}

export function ReadForm({ initialAsOf, onRead }: { initialAsOf: string; onRead: (read: Read) => void }) {
  const [asOf, setAsOf] = useState(initialAsOf);
  const [horizon, setHorizon] = useState('');
  const [invalid, setInvalid] = useState<string | null>(null);

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = horizon.trim();
    if (text !== '' && !/^\d+$/.test(text)) {
      setInvalid('The horizon is a commit sequence number (a whole number).');
      return;
    }
    setInvalid(null);
    onRead({ asOf: asOf.trim(), horizon: text === '' ? undefined : Number(text) });
  }

  return (
    <section className="card">
      <h2 className="card-title">Read the store at an instant</h2>
      <form onSubmit={submit} className="kv" aria-label="Point-in-time read">
        <label>
          As of (ISO-8601 with offset){' '}
          <input aria-label="As of" value={asOf} onChange={(event) => setAsOf(event.target.value)} size={34} />
        </label>
        <label>
          Horizon (optional){' '}
          <input aria-label="Horizon" value={horizon} onChange={(event) => setHorizon(event.target.value)} size={8} />
        </label>
        <button className="control" type="submit">Read</button>
      </form>
      {invalid && <p role="alert"><Badge tone="warn">{invalid}</Badge></p>}
    </section>
  );
}

export interface TimelineRead {
  asOf: string;
  fomcHorizon?: number;
  edgarHorizon?: number;
}

export function TimelineReadForm({ initialAsOf, onRead }: {
  initialAsOf: string; onRead: (read: TimelineRead) => void;
}) {
  const [asOf, setAsOf] = useState(initialAsOf);
  const [fomcHorizon, setFomcHorizon] = useState('');
  const [edgarHorizon, setEdgarHorizon] = useState('');
  const [invalid, setInvalid] = useState<string | null>(null);

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const horizons = [fomcHorizon.trim(), edgarHorizon.trim()];
    if (horizons.some((value) => value !== '' && (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value))))) {
      setInvalid('Each horizon must be a commit sequence number (a whole number).');
      return;
    }
    setInvalid(null);
    onRead({ asOf: asOf.trim(),
      fomcHorizon: horizons[0] === '' ? undefined : Number(horizons[0]),
      edgarHorizon: horizons[1] === '' ? undefined : Number(horizons[1]),
    });
  }

  return (
    <section className="card">
      <h2 className="card-title">Read official events at an instant</h2>
      <form onSubmit={submit} className="kv" aria-label="Timeline point-in-time read">
        <label>
          As of (ISO-8601 with offset){' '}
          <input aria-label="As of" value={asOf} onChange={(event) => setAsOf(event.target.value)} size={34} />
        </label>
        <label>
          FOMC horizon (optional){' '}
          <input aria-label="FOMC horizon" value={fomcHorizon} onChange={(event) => setFomcHorizon(event.target.value)} size={8} />
        </label>
        <label>
          EDGAR horizon (optional){' '}
          <input aria-label="EDGAR horizon" value={edgarHorizon} onChange={(event) => setEdgarHorizon(event.target.value)} size={8} />
        </label>
        <button className="control" type="submit">Read</button>
      </form>
      {invalid && <p role="alert"><Badge tone="warn">{invalid}</Badge></p>}
    </section>
  );
}
