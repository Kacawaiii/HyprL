import { render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { LineChart, layoutSeries } from '../components/LineChart';

afterEach(() => vi.restoreAllMocks());

it('keeps every laid-out point inside the box, even for degenerate series', () => {
  const cases: Array<[number[], number | undefined]> = [
    [[100000, 99000, 101500, 98000], 100000],
    [[5], undefined], [[1, 1, 1], 1], [[1, Number.NaN, 3], 2], [[-1e12, 1e12], 0],
  ];
  for (const [values, baseline] of cases) {
    for (const width of [40, 320, 640]) {
      const layout = layoutSeries(values, width, 220, baseline);
      for (const px of layout.xs) {
        expect(px).toBeGreaterThanOrEqual(0);
        expect(px).toBeLessThanOrEqual(width);
      }
      for (const py of [...layout.ys, layout.fillY, ...(layout.baselineY === undefined ? [] : [layout.baselineY])]) {
        expect(Number.isFinite(py)).toBe(true);
        expect(py).toBeGreaterThanOrEqual(0);
        expect(py).toBeLessThanOrEqual(220);
      }
    }
  }
});

it('sizes the canvas to the clipping frame and draws only inside it', () => {
  // The card around the chart has 16px padding: the frame, not the card, is measured.
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function (this: HTMLElement) {
    return this.classList.contains('line-chart') ? 300 : 332;
  });
  const drawn: Array<[string, number, number]> = [];
  const context = {
    setTransform: vi.fn(), clearRect: vi.fn(), setLineDash: vi.fn(), beginPath: vi.fn(), stroke: vi.fn(),
    fill: vi.fn(), closePath: vi.fn(), fillText: vi.fn(), measureText: () => ({ width: 50 }),
    moveTo: (px: number, py: number) => drawn.push(['moveTo', px, py]),
    lineTo: (px: number, py: number) => drawn.push(['lineTo', px, py]),
  };
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D);
  render(<LineChart label="eq" baseline={100000} fill
    points={[100000, 98000, 103000, 99500].map((value, i) => ({ timestamp: `2026-05-0${i + 1}T00:00:00Z`, value }))} />);
  const canvas = screen.getByRole('img', { name: 'eq' }) as HTMLCanvasElement;
  const frame = canvas.parentElement as HTMLElement;
  expect(frame.style.overflow).toBe('hidden');
  expect(canvas.style.position).toBe('absolute');
  expect(canvas.style.width).toBe('300px');
  expect(canvas.width).toBe(300);
  expect(drawn.length).toBeGreaterThan(4);
  for (const [, px, py] of drawn) {
    expect(px).toBeGreaterThanOrEqual(0);
    expect(px).toBeLessThanOrEqual(300);
    expect(py).toBeGreaterThanOrEqual(0);
    expect(py).toBeLessThanOrEqual(220);
  }
});
