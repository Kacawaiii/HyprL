import { BOOK_LABEL } from '../lib/publicAgentNames';
/**
 * "Aujourd'hui": the one-screen home. Three paper accounts, the market in five numbers, the system light, the next
 * scheduled events, the five news that matter and what the AIs decided. Everything comes from the radar and paper
 * snapshots verbatim; a news line opens a drawer with its full decision chain and its five scores.
 */
import { useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiClient, ApiError } from '../api/client';
import type { AiLine, CalendarEntry, PaperAccount, PaperSnapshot, RadarEvent, RadarHome } from '../api/radarTypes';
import { NewsDrawer, NewsLine } from '../components/radar/NewsLine';
import { EmptyState, ErrorState, LoadingState } from '../components/States';
import { carrySelection } from '../lib/cockpit';
import { ARROW, DIRECTION_WORD, LEVEL_WORD, dollars, parisDay, parisLongDay, parisTime, pctFr, pointsFr, since, tone } from '../lib/fr';
import { useCockpit } from '../state/useCockpit';
import { useQuery } from '../state/useQuery';

const ACCOUNT_NAME: Record<string, string> = { ia_actions: 'IA actions', ia_crypto: 'IA crypto', claude_book: BOOK_LABEL };
const ACCOUNT_SHORT: Record<string, string> = { ia_actions: 'actions', ia_crypto: 'crypto' };
const MARKETS: Array<[string, string]> = [['SPY', 'S&P 500'], ['QQQ', 'Nasdaq'], ['BTC', 'Bitcoin'], ['Brent', 'Pétrole'], ['VIX', 'Peur (VIX)']];
// The VIX measures fear: a rise is bad news for stocks, so its colour is inverted.
const INVERTED = new Set(['VIX']);
const STATUS: Record<AiLine['status'], string> = {
  executed: 'exécuté', sent: 'envoyé', planned: 'prévu', blocked: 'bloqué', skipped: 'écarté', expired: 'expiré',
};
const STATUS_HELP: Record<AiLine['status'], string> = {
  executed: 'Le courtier (compte papier) a exécuté l’ordre.',
  sent: 'L’ordre est chez le courtier, pas encore exécuté.',
  planned: 'Retenu par la règle d’exécution ; aucun ordre envoyé pour l’instant.',
  blocked: 'Bloqué par une sécurité du compte (pause ou palier arrêté).',
  skipped: 'Retenu par la règle, mais écarté par les contrôles avant envoi.',
  expired: 'Ordre envoyé mais jamais exécuté : il a expiré.',
};
const LIGHT: Record<string, string> = { green: 'Tout va bien', orange: 'À surveiller', red: 'Problème', unknown: 'Non vérifié' };

function dayChange(account: PaperAccount): number | null {
  if (account.day_pnl === null || account.day_pnl === undefined || account.equity === null) return null;
  const base = account.account === 'claude_book' ? account.last_equity : account.equity - account.day_pnl;
  return base ? account.day_pnl / base : null;
}

function positions(account: PaperAccount): number {
  return account.account === 'claude_book' ? account.positions.length : account.open_lots ?? account.positions.length;
}

function AccountTile({ account, spy }: { account: PaperAccount; spy: PaperSnapshot['benchmarks']['SPY'] }) {
  const day = dayChange(account);
  const gap = account.return_since_start !== null && spy.return_since_paper_start !== null
    ? account.return_since_start - spy.return_since_paper_start : null;
  const count = positions(account);
  return (
    <article className="tile" aria-label={ACCOUNT_NAME[account.account] ?? account.label}>
      <div className="tile-top">
        <h2 className="tile-name">{ACCOUNT_NAME[account.account] ?? account.label}</h2>
        <p className="tile-value">{dollars(account.equity)}</p>
      </div>
      <dl className="tile-facts">
        <div><dt>Aujourd’hui</dt><dd data-tone={tone(day)}>{pctFr(day)}</dd></div>
        <div><dt>Depuis le départ</dt><dd data-tone={tone(account.return_since_start)}>{pctFr(account.return_since_start)}</dd></div>
        <div>
          <dt>vs S&amp;P 500</dt>
          <dd data-tone={tone(gap)} title={spy.base_at ? `S&P 500 depuis le ${parisDay(spy.base_at)} : ${pctFr(spy.return_since_paper_start)}` : undefined}>
            {gap === null ? '—' : `${pctFr(gap)}`}
          </dd>
        </div>
        <div><dt>Positions</dt><dd>{count === 0 ? 'aucune' : count}</dd></div>
      </dl>
    </article>
  );
}

function Agenda({ entries }: { entries: CalendarEntry[] }) {
  if (entries.length === 0) return <p className="muted">Aucun rendez-vous daté dans le calendrier.</p>;
  return (
    <ol className="agenda" aria-label="Prochains rendez-vous">
      {entries.slice(0, 3).map((e) => (
        <li key={`${e.date}-${e.title}`}>
          <span className="agenda-when">
            <span>{parisDay(e.at ?? e.date)}</span>
            <span className="agenda-time">{e.at ? parisTime(e.at) : 'heure à confirmer'}</span>
          </span>
          <span className="agenda-what">{e.title}</span>
          <span className="level" data-level={e.verification}>{LEVEL_WORD[e.verification] ?? e.verification}</span>
        </li>
      ))}
    </ol>
  );
}

function AiLines({ block, carry }: { block: RadarHome['ai_today']; carry: string }) {
  const lines = block?.lines ?? [];
  if (!block?.run_id) return <p className="muted">Aucune décision des IA enregistrée.</p>;
  if (lines.length === 0) return <p className="muted">Les IA n’ont retenu aucune position ce jour-là.</p>;
  return (
    <>
      <ul className="ai-lines" aria-label="Décisions des IA">
        {lines.slice(0, 5).map((l) => (
          <li key={`${l.account}-${l.asset}-${l.horizon}-${l.slot}`}>
            <span className="ai-asset">{l.asset}</span>
            <span className="ai-dir" data-direction={l.direction}>{ARROW[l.direction] ?? '?'} {DIRECTION_WORD[l.direction] ?? l.direction}</span>
            <span className="ai-size">{l.size === 'full' ? 'taille normale' : 'demi-taille'} · {ACCOUNT_SHORT[l.account]} · {l.horizon === '1d' ? '1 jour' : '5 jours'}</span>
            <span className="status-chip" data-status={l.status} title={STATUS_HELP[l.status]}>{STATUS[l.status]}</span>
          </li>
        ))}
      </ul>
      {lines.length > 5 && <Link className="more" to={{ pathname: '/trader', search: carry }}>Voir les {lines.length} décisions</Link>}
    </>
  );
}

export function TodayPage() {
  const { params } = useCockpit();
  const carry = carrySelection(params);
  const home = useQuery('radar-home', (signal) => apiClient.getRadarHome(signal), { staleMs: 60_000 });
  const paper = useQuery('radar-paper', (signal) => apiClient.getRadarPaper(signal), { staleMs: 60_000 });
  const [open, setOpen] = useState<RadarEvent | null>(null);
  const close = useCallback(() => setOpen(null), []);

  if (home.status === 'loading') return <LoadingState label="Chargement de la journée" />;
  if (home.status === 'error' && home.error) {
    if (home.error instanceof ApiError && [404, 503].includes(home.error.status)) {
      return <EmptyState title="Le résumé du jour n’est pas encore publié"
        detail="L’export des actualités et des comptes n’a pas encore écrit son instantané sur ce serveur." />;
    }
    return <ErrorState error={home.error} onRetry={home.refetch} />;
  }
  const data = home.data;
  if (!data) return null;
  const light = data.system;
  const problems = light?.problems.length ?? 0;
  const session = data.ai_today?.session;
  const todayParis = new Date().toLocaleDateString('fr-CA', { timeZone: 'Europe/Paris' });
  const paperData = paper.data;
  const events = [...data.events].sort((a, b) => a.rank - b.rank).slice(0, 5);

  return (
    <div className="today">
      <header className="today-head">
        <div>
          <h1 className="today-title">Aujourd’hui, {parisLongDay(new Date())}</h1>
          <p className="muted today-sub">Comptes papier : argent fictif, rien de réel n’est engagé. Actualités mises à jour {since(data.radar.cutoff)}.</p>
        </div>
        <Link className="light" data-level={light?.level ?? 'unknown'} to={{ pathname: '/system', search: carry }}
          aria-label={`État du système : ${light ? LIGHT[light.level] : 'inconnu'}, ${problems} problème${problems > 1 ? 's' : ''}`}>
          <span className="light-dot" aria-hidden="true" />
          <span>{!light || light.level === 'unknown' ? 'État non vérifié'
            : problems === 0 ? 'Aucun problème' : `${problems} problème${problems > 1 ? 's' : ''}`}</span>
        </Link>
      </header>

      <ul className="market-strip" aria-label="Marchés aujourd’hui">
        {MARKETS.map(([key, name]) => {
          const value = data.regime[key]?.returns_pct['1d'];
          const shown = tone(value === null || value === undefined ? value : INVERTED.has(key) ? -value : value);
          return <li key={key}><span>{name}</span><strong data-tone={shown}>{pointsFr(value)}</strong></li>;
        })}
      </ul>

      <section className="tiles" aria-label="Comptes">
        {paperData
          ? paperData.accounts.map((a) => <AccountTile key={a.account} account={a} spy={paperData.benchmarks.SPY} />)
          : paper.status === 'loading' ? <LoadingState label="Chargement des comptes" />
            : <p className="muted">Comptes indisponibles pour le moment.</p>}
      </section>

      <div className="today-body">
        <section className="panel news-panel" aria-labelledby="news-title">
          <div className="panel-head">
            <h2 id="news-title">5 news qui comptent</h2>
            <Link className="more" to={{ pathname: '/radar', search: carry }}>Toutes les actualités</Link>
          </div>
          <p className="panel-hint">Importance sur 100 · actifs concernés et ce que les IA attendent sur 1 jour (↑ hausse, ↓ baisse, – pas d’avis, rien = pas analysé). Cliquez pour le détail.</p>
          <ul className="news-list" aria-label="5 news qui comptent">
            {events.map((e) => <NewsLine key={e.id} event={e} onOpen={setOpen} />)}
          </ul>
        </section>

        <div className="today-side">
          <section className="panel" aria-labelledby="agenda-title">
            <h2 id="agenda-title">Prochains rendez-vous</h2>
            <Agenda entries={data.calendar ?? []} />
          </section>
          <section className="panel" aria-labelledby="ai-title">
            <h2 id="ai-title">Ce que font les IA {session && session !== todayParis ? `(décisions du ${parisDay(session)})` : 'aujourd’hui'}</h2>
            <AiLines block={data.ai_today} carry={carry} />
          </section>
        </div>
      </div>
      {open && <NewsDrawer event={open} onClose={close} />}
    </div>
  );
}
