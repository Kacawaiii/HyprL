import { formatNum, type NumKind } from '../lib/format';

/** A rounded number for reading; the exact value stays in the tooltip. */
export function Num({ value, kind = 'ratio', digits }: {
  value: string | number | null | undefined;
  kind?: NumKind;
  digits?: number;
}) {
  const exact = value === null || value === undefined ? undefined : String(value);
  return <span className="num" title={exact}>{formatNum(value, kind, digits)}</span>;
}
