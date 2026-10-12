/**
 * Plain French formatting for Beginner pages: amounts, signed percentages, Paris times and arrows.
 * Display only: every value is the snapshot's own; nothing here computes a score.
 */

const PARIS = 'Europe/Paris';

/** 107097.21 -> "107 097 $" (whole dollars; the detail pages keep cents). */
export function dollars(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${Math.round(value).toLocaleString('fr-FR').replace(/[\u00a0\u202f]/g, ' ')} $`;
}

/** A fraction (0.0042) as "+0,42 %"; zero has no sign. */
export function pctFr(fraction: number | null | undefined, digits = 2): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return '—';
  const value = fraction * 100;
  const text = Math.abs(value).toFixed(digits).replace('.', ',');
  if (Number(text.replace(',', '.')) === 0) return `${(0).toFixed(digits).replace('.', ',')} %`;
  return `${value > 0 ? '+' : '−'}${text} %`;
}

/** A value already in percent (1.15 for +1.15 %). */
export function pointsFr(percent: number | null | undefined, digits = 2): string {
  return pctFr(percent === null || percent === undefined ? percent : percent / 100, digits);
}

export function tone(value: number | null | undefined): 'up' | 'down' | 'flat' {
  if (value === null || value === undefined || !Number.isFinite(value) || value === 0) return 'flat';
  return value > 0 ? 'up' : 'down';
}

/** "mar. 13 oct." in Paris. */
export function parisDay(at: string): string {
  const date = new Date(at.length === 10 ? `${at}T12:00:00Z` : at);
  return date.toLocaleDateString('fr-FR', { timeZone: PARIS, weekday: 'short', day: 'numeric', month: 'short' });
}

/** "13:00" in Paris. */
export function parisTime(at: string): string {
  return new Date(at).toLocaleTimeString('fr-FR', { timeZone: PARIS, hour: '2-digit', minute: '2-digit' });
}

/** "lundi 12 octobre" for today's header. */
export function parisLongDay(at: Date): string {
  return at.toLocaleDateString('fr-FR', { timeZone: PARIS, weekday: 'long', day: 'numeric', month: 'long' });
}

/** "il y a 3 h" / "à 21:35" for a freshness note. */
export function since(at: string | null | undefined, now = Date.now()): string {
  if (!at || !Number.isFinite(Date.parse(at))) return 'date inconnue';
  const minutes = Math.round((now - Date.parse(at)) / 60_000);
  if (minutes < 60) return `il y a ${Math.max(minutes, 1)} min`;
  if (minutes < 48 * 60) return `il y a ${Math.round(minutes / 60)} h`;
  return `le ${parisDay(at)}`;
}

export const ARROW: Record<string, string> = { UP: '↑', DOWN: '↓', ABSTAIN: '–' };
export const DIRECTION_WORD: Record<string, string> = { UP: 'hausse', DOWN: 'baisse', ABSTAIN: 'pas d’avis' };

/** OFFICIEL / FIL / NON VERIFIE in short words a beginner reads. */
export const LEVEL_WORD: Record<string, string> = { OFFICIEL: 'officiel', FIL: 'presse', 'NON VERIFIE': 'non vérifié' };
