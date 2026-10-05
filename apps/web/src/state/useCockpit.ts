import { useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { parseSelection, writeSelection } from '../lib/cockpit';
import type { CockpitSelection } from '../lib/cockpit';

/** The cockpit selection, held in the URL: mode, product, period and model survive a switch and a reload. */
export function useCockpit() {
  const [params, setParams] = useSearchParams();
  const selection = useMemo(() => parseSelection(params), [params]);
  const update = useCallback((patch: Partial<CockpitSelection>) => {
    setParams((current) => writeSelection(current, patch), { replace: false });
  }, [setParams]);
  return { selection, update, params };
}
