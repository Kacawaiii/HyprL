import type { RoleLifecycle } from '../api/traderTypes';
import { analystLabel } from '../lib/trader';

export function TraderLifecycle({ roles }: { roles?: RoleLifecycle }) {
  const retired = Object.entries(roles ?? {}).filter(([, value]) => value.state === 'RETIRED');
  if (!retired.length) return null;
  return <p className="metric-sub" aria-label="Retired trader roles">
    {retired.map(([role]) => `${analystLabel(role)}: RETIRED`).join(' · ')}.
    {' '}Historical views and results are retained. Current decisions use the Claude analyst and Claude reviewer.
  </p>;
}
