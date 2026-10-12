/**
 * Beginner view of the official sources: what the Fed and the SEC published, in plain words, read from the same
 * read-only stores the trader uses. The stores are frozen captures; the page says so and when they were last written.
 */
import { apiClient, ApiError } from '../api/client';
import type { EdgarFilingSummary, FomcItemSummary } from '../api/types';
import { ShowMore } from '../components/Simple';
import { LoadingState } from '../components/States';
import { parisDay, parisTime } from '../lib/fr';
import { useQuery } from '../state/useQuery';

const FORMS: Record<string, string> = {
  '8-K': 'événement important', '10-K': 'rapport annuel', '10-Q': 'rapport trimestriel', '4': 'achat/vente d’un dirigeant',
  'DEF 14A': 'convocation à l’assemblée', 'S-1': 'introduction en bourse',
};

function unavailable(error: Error | undefined, name: string) {
  const missing = error instanceof ApiError && [404, 503].includes(error.status);
  return <p className="muted">{missing ? `${name} : source non branchée sur ce serveur.` : `${name} : lecture impossible (${error?.message ?? 'erreur'}).`}</p>;
}

function Fed() {
  const status = useQuery('simple-fomc-status', (signal) => apiClient.getFomcStatus(signal));
  const asOf = status.data?.status === 'AVAILABLE' ? status.data.suggested_as_of ?? null : null;
  const read = useQuery(asOf ? `simple-fomc-read:${asOf}` : null,
    (signal) => apiClient.getFomcSnapshot(asOf ?? '', undefined, signal, { limit: 200 }));
  if (status.status === 'loading' || (asOf && read.status === 'loading')) return <LoadingState label="Lecture de la Fed" />;
  if (status.error) return unavailable(status.error, 'Fed');
  if (!asOf) return <p className="muted">Fed : aucune capture enregistrée.</p>;
  if (read.error || !read.data) return unavailable(read.error, 'Fed');
  const items = [...read.data.items].sort((a, b) => (b.official_statement_date ?? '').localeCompare(a.official_statement_date ?? ''));
  return (
    <>
      <p className="panel-hint">Archive figée, dernière capture le {parisDay(asOf)} à {parisTime(asOf)} (Paris) · {items.length} communiqués.</p>
      <ShowMore<FomcItemSummary> items={items} label="Communiqués de la Fed" render={(item) => (
        <li key={item.sid} className="row-line">
          <span>{item.official_statement_date ? parisDay(item.official_statement_date) : 'date inconnue'}</span>
          <span className="grow">{item.canonical_source_url
            ? <a href={item.canonical_source_url} target="_blank" rel="noreferrer noopener">{item.title ?? 'Communiqué'}</a>
            : item.title ?? 'Communiqué'}</span>
          {item.declared_release_at && <span className="muted">publié à {parisTime(item.declared_release_at)}</span>}
        </li>
      )} />
    </>
  );
}

function Sec() {
  const status = useQuery('simple-edgar-status', (signal) => apiClient.getEdgarStatus(signal));
  const asOf = status.data?.status === 'AVAILABLE' ? status.data.suggested_as_of ?? null : null;
  const read = useQuery(asOf ? `simple-edgar-read:${asOf}` : null,
    (signal) => apiClient.getEdgarSnapshot(asOf ?? '', undefined, signal, { limit: 200 }));
  if (status.status === 'loading' || (asOf && read.status === 'loading')) return <LoadingState label="Lecture de la SEC" />;
  if (status.error) return unavailable(status.error, 'SEC');
  if (!asOf) return <p className="muted">SEC : aucune capture enregistrée.</p>;
  if (read.error || !read.data) return unavailable(read.error, 'SEC');
  const filings = [...read.data.filings].sort((a, b) => b.filing_date.localeCompare(a.filing_date));
  const companies = [...new Set(filings.map((f) => f.entity_name).filter(Boolean))];
  return (
    <>
      <p className="panel-hint">Essai figé sur {companies.length || read.data.watchlist?.length || 0} sociétés ({companies.join(', ') || 'noms inconnus'}),
        dernière capture le {parisDay(asOf)} · {filings.length} dépôts.</p>
      <ShowMore<EdgarFilingSummary> items={filings} label="Dépôts à la SEC" render={(f) => (
        <li key={f.accession_number} className="row-line">
          <span>{parisDay(f.filing_date)} {f.filing_date.slice(0, 4)}</span>
          <span className="grow">{f.entity_name ?? 'Société inconnue'} : {FORMS[f.form] ?? 'dépôt'}</span>
          <span className="muted">formulaire {f.form}</span>
        </li>
      )} />
    </>
  );
}

export function OfficialSimple() {
  return (
    <div className="grid grid-2">
      <section className="panel" aria-label="Fed"><h2>Fed : décisions de taux (FOMC)</h2><Fed /></section>
      <section className="panel" aria-label="SEC"><h2>SEC : dépôts des sociétés</h2><Sec /></section>
    </div>
  );
}
