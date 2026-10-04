import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { Num } from '../components/Num';
import { formatMoney, formatPercent, formatRatio } from '../lib/format';

const LONG = '-0.123456789012345678901234567890';

it('rounds for reading and keeps the exact string in the title', () => {
  const rows: Array<[Parameters<typeof Num>[0], string]> = [
    [{ value: LONG }, '-0.1235'],
    [{ value: '98765.43210987654321', kind: 'money' }, '98,765.43'],
    [{ value: '-1234.567890123456789', kind: 'money' }, '-1,234.57'],
    [{ value: '-0.01234567890123456789', kind: 'percent' }, '-1.23 %'],
    [{ value: '0', kind: 'ratio' }, '0'],
    [{ value: '0', kind: 'money' }, '0.00'],
    [{ value: '0', kind: 'percent' }, '0.00 %'],
    [{ value: '0.000000123456789', kind: 'ratio' }, '0.0000001235'],
  ];
  for (const [props, text] of rows) {
    const { unmount } = render(<Num {...props} />);
    const node = screen.getByText(text);
    expect(node).toHaveAttribute('title', String(props.value));
    unmount();
  }
});

it('shows "undefined" for a missing Sharpe, with no invented title', () => {
  render(<Num value={null} />);
  const node = screen.getByText('undefined');
  expect(node).not.toHaveAttribute('title');
});

it('leaves non-numeric text as served', () => {
  expect(formatRatio('undefined')).toBe('undefined');
  expect(formatMoney('n/a')).toBe('n/a');
  expect(formatPercent('NaN')).toBe('NaN');
});
