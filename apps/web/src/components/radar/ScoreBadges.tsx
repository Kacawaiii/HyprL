/** The five scores of an event, each in its own labelled box. They answer different questions and are never merged. */
import type { EventBadges } from '../../api/radarTypes';
import { analystLabel, pct, signedNumber } from '../../lib/radar';

function Score({ id, name, value, detail, note }: {
  id: string; name: string; value: string; detail?: string; note?: string;
}) {
  return (
    <div className="score" data-score={id} title={note}>
      <dt>{name}</dt>
      <dd><span className="score-value">{value}</span>{detail && <span className="score-detail">{detail}</span>}</dd>
    </div>
  );
}

export function ScoreBadges({ badges, expert }: { badges: EventBadges; expert: boolean }) {
  const importance = badges.event_importance;
  const evidence = badges.evidence_strength;
  const conviction = badges.model_conviction;
  const quality = badges.predictive_quality;
  const after = badges.after_cost_performance;

  const convictionByAnalyst = conviction.by_analyst
    ? Object.entries(conviction.by_analyst).map(([name, value]) => `${analystLabel(name)} ${pct(value)}`).join(' · ')
    : undefined;
  const qualityEntries = quality.by_analyst ? Object.entries(quality.by_analyst) : [];
  const qualityLead = qualityEntries.find(([key]) => key.startsWith('analyst_claude'))
    ?? qualityEntries[0];
  const afterLabels = after.event_labels;

  return (
    <dl className="scores" aria-label="Scores (five separate measures)">
      <Score id="importance" name="Event importance"
        value={importance.value === null ? '—' : `${Math.round(importance.value)}/100`}
        detail={expert && importance.components
          ? Object.entries(importance.components).map(([k, v]) => `${k.replace(/_/g, ' ')} ${v}`).join(' · ')
          : undefined}
        note="Radar rules: how much this event could matter. Not a forecast." />
      <Score id="evidence" name="Evidence strength"
        value={evidence.value === null ? '—' : `${Math.round(evidence.value)}/100`}
        detail={`${evidence.status?.replace(/_/g, ' ') ?? 'unknown'}${evidence.primary ? ' · primary source' : ''}`}
        note="How well the facts are sourced, independent of what they imply." />
      <Score id="conviction" name="Model conviction"
        value={conviction.label ?? (convictionByAnalyst ? 'by model' : 'none yet')}
        detail={expert || !conviction.label ? convictionByAnalyst : undefined}
        note={conviction.basis} />
      <Score id="quality" name="Predictive quality"
        value={qualityLead ? `hit ${pct(qualityLead[1].hit_rate)}` : 'not scored'}
        detail={qualityLead
          ? `${analystLabel(qualityLead[0].split('/')[0] ?? '')} · ${qualityLead[1].realized ?? 0} realised${
            expert && qualityLead[1].brier !== null ? ` · Brier ${qualityLead[1].brier?.toFixed(3)}` : ''}`
          : undefined}
        note={quality.basis} />
      <Score id="after-cost" name="After-cost performance"
        value={afterLabels ? signedNumber(afterLabels.mean_net_unit_pnl * 100, 2) + ' %' : 'not realised'}
        detail={afterLabels ? `${afterLabels.n} labelled` : 'no labelled outcome for these assets'}
        note={after.basis} />
    </dl>
  );
}
