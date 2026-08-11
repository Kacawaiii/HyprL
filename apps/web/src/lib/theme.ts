/** Theme preference, resolved before the first paint.
 *
 *  Kept out of SettingsPage on purpose: the shell needs `readTheme` on its
 *  very first render, and importing it from the page would pull that whole
 *  route into the entry chunk and undo its lazy split.
 *
 *  Stored locally rather than server-side. A theme is per-browser, and it has
 *  to apply before any request completes -- waiting on HTTP to decide a
 *  background colour means painting the wrong one first. */
const THEME_KEY = 'hyprl.theme';

export type ThemePreference = 'dark' | 'light' | 'system';

export function readTheme(): ThemePreference {
  const stored = localStorage.getItem(THEME_KEY);
  return stored === 'light' || stored === 'system' || stored === 'dark' ? stored : 'dark';
}

export function writeTheme(preference: ThemePreference): void {
  localStorage.setItem(THEME_KEY, preference);
}

export function applyTheme(preference: ThemePreference): void {
  const resolved =
    preference === 'system'
      ? window.matchMedia?.('(prefers-color-scheme: light)').matches
        ? 'light'
        : 'dark'
      : preference;
  document.documentElement.setAttribute('data-theme', resolved);
}
