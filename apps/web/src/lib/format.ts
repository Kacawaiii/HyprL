/**
 * Display-only number formatting.
 *
 * The API serves exact decimal strings and those strings are the record. These
 * helpers only decide how many digits a person reads; nothing computed here is
 * ever sent back or used to decide anything. Callers keep the exact string in
 * a tooltip (see `Num`) and in the detailed views.
 */

const LOCALE = 'en-US';

export type NumKind = 'ratio' | 'money' | 'percent';

function parse(value: string | number | null | undefined): number | null {
  if (value === null || value === undefined || value === '') return null;
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function missing(value: string | number | null | undefined, fallback: string): string {
  return value === null || value === undefined || value === '' ? fallback : String(value);
}

/** A ratio or score at `digits` significant digits (default 4); 0 stays "0". */
export function formatRatio(value: string | number | null | undefined, digits = 4, fallback = 'undefined'): string {
  const parsed = parse(value);
  if (parsed === null) return missing(value, fallback);
  return parsed.toLocaleString(LOCALE, { maximumSignificantDigits: digits });
}

/** Money with two decimals and thousands separators. */
export function formatMoney(value: string | number | null | undefined, fallback = 'undefined'): string {
  const parsed = parse(value);
  if (parsed === null) return missing(value, fallback);
  return parsed.toLocaleString(LOCALE, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** A fraction (0.0123) shown as a percentage with its unit ("1.23 %"). */
export function formatPercent(value: string | number | null | undefined, digits = 2, fallback = 'undefined'): string {
  const parsed = parse(value);
  if (parsed === null) return missing(value, fallback);
  const text = (parsed * 100).toLocaleString(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return `${text} %`;
}

export function formatNum(value: string | number | null | undefined, kind: NumKind, digits?: number): string {
  if (kind === 'money') return formatMoney(value);
  if (kind === 'percent') return formatPercent(value, digits);
  return formatRatio(value, digits);
}
