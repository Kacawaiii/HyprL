import { PUBLIC_AGENT_NAMES } from '../lib/publicAgentNames';
import type { RoleLifecycle } from '../api/traderTypes';
import { analystLabel } from '../lib/trader';

export function TraderLifecycle({ roles }: { roles?: RoleLifecycle }) {
  const retired = Object.entries(roles ?? {}).filter(([, value]) => value.state === 'RETIRED');
  if (!retired.length) return null;
  return <p className="metric-sub" aria-label="Retired trader roles">
    {retired.map(([role]) => `${analystLabel(role)}: RETIRED`).join(' · ')}.
    {' '}Historical views and results are retained. Current decisions use the {PUBLIC_AGENT_NAMES.opus} analyst and reviewer.
  </p>;
}
