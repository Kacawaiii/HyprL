/**
 * A canvas line chart for long, bounded series.
 *
 * Canvas rather than SVG because an equity curve has hundreds of points and
 * each SVG node costs layout; and because the server already downsamples, the
 * browser never holds the full history. `baseline` draws a reference line
 * (initial equity, or zero for a drawdown) so a reader can see the sign
 * without decoding the axis.
 */

import { useEffect, useRef } from 'react';

export interface Layout {
  xs: number[];
  ys: number[];
  left: number;
  right: number;
  top: number;
  bottom: number;
  baselineY?: number;
  fillY: number;
}

/**
 * Pixel layout of a series inside a `width` x `height` box. Pure, so the
 * geometry can be tested: every returned coordinate lies inside the box.
 * Non-finite values are dropped rather than poisoning the domain with NaN.
 */
export function layoutSeries(values: number[], width: number, height: number, baseline?: number): Layout {
  const finite = values.filter(Number.isFinite);
  const candidates = baseline === undefined || !Number.isFinite(baseline) ? finite : [...finite, baseline];
  let low = candidates.length ? Math.min(...candidates) : 0;
  let high = candidates.length ? Math.max(...candidates) : 0;
  if (low === high) { low -= 1; high += 1; }
  const padding = (high - low) * 0.08;
  low -= padding;
  high += padding;
  const left = 8;
  const right = Math.max(left, width - 8);
  const top = 10;
  const bottom = Math.max(top, height - 18);
  const xs = values.map((_, index) =>
    left + (values.length === 1 ? 0 : (index / (values.length - 1)) * (right - left)));
  const y = (value: number) => bottom - ((value - low) / (high - low)) * (bottom - top);
  const ys = values.map((value) => (Number.isFinite(value) ? y(value) : bottom));
  const hasBaseline = baseline !== undefined && Number.isFinite(baseline);
  return { xs, ys, left, right, top, bottom, baselineY: hasBaseline ? y(baseline) : undefined, fillY: hasBaseline ? y(baseline) : y(low) };
}

interface Props {
  points: Array<{ timestamp: string; value: number }>;
  height?: number;
  baseline?: number;
  fill?: boolean;
  label?: string;
}

export function LineChart({ points, height = 220, baseline, fill = false, label }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const frame = frameRef.current;
    if (!canvas || !frame || points.length === 0) return;
    const draw = () => {
      const ratio = window.devicePixelRatio || 1;
      // The frame has no padding of its own, so its clientWidth is exactly the
      // room the card leaves. (Measuring the card itself counted its padding
      // and made the canvas wider than the card's content box.)
      const width = frame.clientWidth || 640;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const context = canvas.getContext('2d');
      if (!context) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const styles = getComputedStyle(canvas);
      const ink = styles.getPropertyValue('--accent').trim() || '#4c8bf5';
      const grid = styles.getPropertyValue('--border').trim() || '#2a2f3a';
      const muted = styles.getPropertyValue('--text-muted').trim() || '#8b93a7';

      const { xs, ys, left, right, baselineY, fillY } = layoutSeries(
        points.map((point) => point.value), width, height, baseline);
      const x = (index: number) => xs[index] ?? left;
      const y = (index: number) => ys[index] ?? 0;

      if (baselineY !== undefined) {
        context.strokeStyle = grid;
        context.setLineDash([4, 4]);
        context.beginPath();
        context.moveTo(left, baselineY ?? 0);
        context.lineTo(right, baselineY ?? 0);
        context.stroke();
        context.setLineDash([]);
      }

      if (fill) {
        context.beginPath();
        context.moveTo(x(0), fillY);
        points.forEach((_, index) => context.lineTo(x(index), y(index)));
        context.lineTo(x(points.length - 1), fillY);
        context.closePath();
        context.fillStyle = `${ink}22`;
        context.fill();
      }

      context.beginPath();
      points.forEach((_, index) => {
        const px = x(index);
        const py = y(index);
        if (index === 0) context.moveTo(px, py);
        else context.lineTo(px, py);
      });
      context.strokeStyle = ink;
      context.lineWidth = 1.5;
      context.stroke();

      const first = points[0];
      const final = points[points.length - 1];
      if (first && final) {
        context.fillStyle = muted;
        context.font = '10px ui-monospace, monospace';
        context.fillText(first.timestamp.slice(0, 10), left, height - 5);
        const lastLabel = final.timestamp.slice(0, 10);
        context.fillText(lastLabel, right - context.measureText(lastLabel).width, height - 5);
      }
    };
    draw();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(draw);
    observer.observe(frame);
    return () => observer.disconnect();
  }, [points, height, baseline, fill]);

  if (points.length === 0) {
    return <div className="state">No points to plot</div>;
  }
  // The canvas is absolutely positioned inside a clipping frame: its pixel
  // size can never widen the card or the grid track around it.
  return (
    <div ref={frameRef} className="line-chart" style={{ position: 'relative', width: '100%', height, overflow: 'hidden' }}>
      <canvas ref={canvasRef} role="img" aria-label={label ?? 'series'}
        style={{ position: 'absolute', left: 0, top: 0, display: 'block', maxWidth: '100%' }} />
    </div>
  );
}
