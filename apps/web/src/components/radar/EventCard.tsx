/** An event as one reading path: what happened, what it could change, who is concerned, what the models expect. */
import type { RadarEvent } from '../../api/radarTypes';
import { host, instant } from '../../lib/radar';
import { AssetRow } from './AssetRow';
import { ScoreBadges } from './ScoreBadges';

function BeginnerSummary({ event }: { event: RadarEvent }) {
  const change = event.what_changed;
  const watched = event.assets.slice(0, 3).map((asset) => asset.symbol).join(', ');
  const covered = event.assets.filter((asset) => asset.anticipation.state === 'COVERED');
  const realised = event.assets.flatMap((asset) => asset.outcomes);
  return (
    <dl className="beginner-path">
      <div><dt>What to watch</dt><dd>{watched || 'No asset identified yet'}</dd></div>
      <div><dt>Why</dt><dd>{change.impact ?? event.assets[0]?.mechanism ?? 'The radar has no impact scenario for this event.'}</dd></div>
      <div><dt>For how long</dt><dd>{change.horizon ?? 'Not stated'}</dd></div>
      <div>
        <dt>Risk</dt>
        <dd>{change.invalidation ?? 'No invalidation stated.'}{' '}
          {covered.length === 0 && 'No model run covers these assets, so nothing is anticipated.'}
        </dd>
      </div>
      <div>
        <dt>Result</dt>
        <dd>{realised.length > 0 ? `${realised.length} labelled outcome(s) below` : 'Not realised yet.'}</dd>
      </div>
    </dl>
  );
}

export function EventCard({ event, expert }: { event: RadarEvent; expert: boolean }) {
  const change = event.what_changed;
  const titleId = `event-${event.id.slice(0, 12)}`;
  return (
    <article className="radar-event card" aria-labelledby={titleId}>
      <header className="event-head">
        <span className="event-rank" aria-label={`Rank ${event.rank}`}>{event.rank}</span>
        <div>
          <h2 id={titleId} className="event-title">
            {event.link
              ? <a href={event.link} target="_blank" rel="noreferrer noopener">{event.headline}</a>
              : event.headline}
          </h2>
          <p className="muted event-meta">
            {event.source ?? host(event.link)} · seen {instant(event.available_at)}
            {event.published_at && <> · published {instant(event.published_at)}</>}
            {event.novelty && <> · {event.novelty}</>}
          </p>
        </div>
      </header>

      <ScoreBadges badges={event.badges} expert={expert} />

      {!expert && <BeginnerSummary event={event} />}

      <section aria-label="What changed versus expectations" className="event-section">
        <h3>What changed versus expectations</h3>
        <p>{change.changed_expectations ?? change.expectations ?? 'Nothing recorded.'}</p>
        {expert && (
          <>
            {change.summary && <p className="muted">{change.summary}</p>}
            {change.priced_in && <p className="muted"><strong>Priced in:</strong> {change.priced_in}</p>}
            {change.invalidation && <p className="muted"><strong>Invalidation:</strong> {change.invalidation}</p>}
            <p className="muted">Origin: {change.source === 'llm_scenario' ? 'model scenario (checked against the report numbers)' : 'rules only'}</p>
          </>
        )}
      </section>

      <section aria-label="Assets concerned" className="event-section">
        <h3>Assets concerned and what the models expect</h3>
        {event.assets.length === 0
          ? <p className="muted">No concerned asset was identified.</p>
          : <ul className="assets">{event.assets.map((asset) => <AssetRow key={asset.symbol} asset={asset} expert={expert} />)}</ul>}
      </section>

      {expert && (
        <details className="event-section provenance">
          <summary>Provenance and availability</summary>
          <dl className="kv-list">
            <div><dt>Event id</dt><dd className="hash" title={event.id}>{event.id.slice(0, 16)}…</dd></div>
            <div><dt>Available at T</dt><dd>{instant(event.available_at)}</dd></div>
            <div><dt>Priced-in status</dt><dd>{event.provenance.priced_in_status ?? '—'}</dd></div>
            <div><dt>Retail hype</dt><dd>{event.provenance.retail_hype ?? '—'}</dd></div>
            <div><dt>Independence</dt><dd>{event.provenance.independence ?? '—'}</dd></div>
            <div><dt>Themes</dt><dd>{[...event.themes, ...event.countries].join(', ') || '—'}</dd></div>
          </dl>
          <ul className="stories">
            {event.stories.map((story) => (
              <li key={story.url}>
                <a href={story.url} target="_blank" rel="noreferrer noopener">{story.headline ?? story.url}</a>
                <span className="muted"> · {story.publisher ?? host(story.url)} · received {instant(story.received_at)}
                  {story.primary && ' · primary'}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </article>
  );
}
