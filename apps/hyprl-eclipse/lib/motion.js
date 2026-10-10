/**
 * Reduced-motion preference with a visitor override. The site follows prefers-reduced-motion by default; a visitor
 * whose system asks for reduced motion can still choose the full experience ("Activer les animations"), remembered
 * in this browser. Every motion consumer (scene, smooth scroll, reveals, the motion button) reads this one source.
 */
const media = matchMedia('(prefers-reduced-motion: reduce)');
const KEY = 'hyprl-motion';
let override = false;
try { override = localStorage.getItem(KEY) === 'full'; } catch { /* storage blocked: default to the system preference */ }
const listeners = new Set();
export const motionPref = {
  get reduced() { return media.matches && !override; },
  get systemReduced() { return media.matches; },
  setOverride(value) {
    override = Boolean(value);
    try { override ? localStorage.setItem(KEY, 'full') : localStorage.removeItem(KEY); } catch { /* not persisted */ }
    for (const fn of listeners) fn();
  },
  onChange(fn) {
    listeners.add(fn); media.addEventListener('change', fn);
    return () => { listeners.delete(fn); media.removeEventListener('change', fn); };
  }
};
