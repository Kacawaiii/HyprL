import type { Score } from '../lib/cockpit';

/** One score. Never merged with another: the definition, the period and the sample size travel with the number. */
export function ScoreCard({ score }: { score: Score }) {
  return (
    <article className="card score" aria-label={`${score.title} score`}>
      <h3 className="card-title">{score.title}</h3>
      <div className="metric">{score.value ?? 'not provided'}</div>
      {score.detail && <div className="metric-sub">{score.detail}</div>}
      <dl style={{ margin: '10px 0 0' }}>
        <div className="kv"><dt>Period</dt><dd>{score.period}</dd></div>
        <div className="kv"><dt>Sample</dt><dd>{score.sample === null ? 'unknown' : `${score.sample.toLocaleString()} ${score.sampleUnit}`}</dd></div>
      </dl>
      <details>
        <summary>Definition</summary>
        <p className="metric-sub">{score.definition}</p>
      </details>
    </article>
  );
}
