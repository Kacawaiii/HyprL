/** One news line (importance, title, assets and the AIs' expected direction) and its detail drawer. */
import type { RadarEvent } from '../../api/radarTypes';
import { ARROW, DIRECTION_WORD, parisDay, parisTime } from '../../lib/fr';
import { Drawer } from '../Simple';
import { EventCard } from './EventCard';

/** The direction the AIs expect for an asset: the consensus view on 1 day, else the analysts when they agree. */
export function expectation(event: RadarEvent): Array<{ symbol: string; direction: string | null }> {
  return event.assets.slice(0, 3).map((asset) => {
    const views = asset.anticipation.latest?.views ?? [];
    const daily = views.filter((v) => v.horizon === '1d');
    const consensus = daily.find((v) => v.analyst === 'consensus');
    if (consensus) return { symbol: asset.symbol, direction: consensus.view };
    const calls = new Set(daily.filter((v) => v.analyst.startsWith('analyst_')).map((v) => v.view));
    return { symbol: asset.symbol, direction: calls.size === 1 ? [...calls][0] ?? null : calls.size > 1 ? 'MIXED' : null };
  });
}

export function NewsLine({ event, onOpen }: { event: RadarEvent; onOpen: (event: RadarEvent) => void }) {
  const importance = event.badges.event_importance.value;
  const calls = expectation(event);
  return (
    <li>
      <button className="news-line" onClick={() => onOpen(event)}>
        <span className="news-score" title="Importance de l’événement sur 100">{importance === null ? '—' : Math.round(importance)}</span>
        <span className="news-title">{event.headline}</span>
        <span className="news-assets">
          {calls.length === 0 ? <span className="muted">aucun actif</span> : calls.map((c) => (
            <span key={c.symbol} className="asset-call" data-direction={c.direction ?? 'NONE'}
              title={c.direction === null ? 'Pas encore analysé par les IA' : c.direction === 'MIXED' ? 'Les IA ne sont pas d’accord'
                : `Les IA attendent : ${DIRECTION_WORD[c.direction] ?? c.direction}`}>
              {c.symbol}{c.direction === null ? '' : ` ${c.direction === 'MIXED' ? '↕' : ARROW[c.direction] ?? '?'}`}
            </span>
          ))}
        </span>
      </button>
    </li>
  );
}

export function NewsDrawer({ event, onClose }: { event: RadarEvent; onClose: () => void }) {
  return (
    <Drawer title="Détail de la news" onClose={onClose}>
      <p className="muted">Vue {event.available_at ? `le ${parisDay(event.available_at)} à ${parisTime(event.available_at)} (Paris)` : 'à une date inconnue'}.
        Les 5 mesures sont séparées et ne s’additionnent jamais.</p>
      <EventCard event={event} expert={false} />
    </Drawer>
  );
}
