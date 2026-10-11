import type { AnalysisSnapshot, OverlayKind } from '../api/analysisTypes';
export const overlayLabels: Record<OverlayKind, string> = {
  event: 'Événements radar', decision: 'Décisions IA', book_entry: 'Intentions Claude', book_exit: 'Sorties Claude', outcome: 'Résultats réalisés',
};
export const symbolKey = (asset: string) => asset.toUpperCase().replace(/[-/]/g, '');
export function marketAssets(data: AnalysisSnapshot) {
  const seen = new Set<string>();
  return [...Object.keys(data.prices), ...data.overlays.map((o) => o.asset)].filter((asset) => {
    const key = symbolKey(asset);
    if (!key || seen.has(key)) return false;
    seen.add(key); return true;
  }).sort();
}
/** Filter in display space only. Dates come from the export; no reconstructed bars. */
export function marketSelection(data: AnalysisSnapshot, asset: string, horizon: string, days: number, kinds: OverlayKind[]) {
  const key = symbolKey(asset);
  const observations = Object.entries(data.prices).filter(([a]) => symbolKey(a) === key).flatMap(([, p]) => p)
    .filter((p) => Number.isFinite(Date.parse(p.at)) && Number.isFinite(p.price))
    .sort((a, b) => a.at.localeCompare(b.at));
  const matching = data.overlays.filter((o) => symbolKey(o.asset) === key && o.at && Number.isFinite(Date.parse(o.at)) &&
    (horizon === 'all' || o.horizon === null || o.horizon === horizon));
  const end = Math.max(...observations.map((p) => Date.parse(p.at)), ...matching.map((o) => Date.parse(o.at!)));
  const start = days ? end - days * 86400_000 : -Infinity;
  return { prices: observations.filter((p) => Date.parse(p.at) >= start),
    overlays: matching.filter((o) => Date.parse(o.at!) >= start && kinds.includes(o.kind)).sort((a, b) => a.at!.localeCompare(b.at!)) };
}
