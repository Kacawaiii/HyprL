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

interface Props {
  points: Array<{ timestamp: string; value: number }>;
  height?: number;
  baseline?: number;
  fill?: boolean;
  label?: string;
}

export function LineChart({ points, height = 220, baseline, fill = false, label }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    if (!canvas || !parent || points.length === 0) return;

    const ratio = window.devicePixelRatio || 1;
    const width = parent.clientWidth || 640;
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

    const values = points.map((point) => point.value);
    const candidates = baseline === undefined ? values : [...values, baseline];
    let low = Math.min(...candidates);
    let high = Math.max(...candidates);
    if (low === high) { low -= 1; high += 1; }
    const padding = (high - low) * 0.08;
    low -= padding;
    high += padding;

    const left = 8;
    const right = width - 8;
    const top = 10;
    const bottom = height - 18;
    const x = (index: number) =>
      left + (points.length === 1 ? 0 : (index / (points.length - 1)) * (right - left));
    const y = (value: number) => bottom - ((value - low) / (high - low)) * (bottom - top);

    if (baseline !== undefined) {
      context.strokeStyle = grid;
      context.setLineDash([4, 4]);
      context.beginPath();
      context.moveTo(left, y(baseline));
      context.lineTo(right, y(baseline));
      context.stroke();
      context.setLineDash([]);
    }

    if (fill) {
      context.beginPath();
      context.moveTo(x(0), y(baseline ?? low));
      points.forEach((point, index) => context.lineTo(x(index), y(point.value)));
      context.lineTo(x(points.length - 1), y(baseline ?? low));
      context.closePath();
      context.fillStyle = `${ink}22`;
      context.fill();
    }

    context.beginPath();
    points.forEach((point, index) => {
      const px = x(index);
      const py = y(point.value);
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
  }, [points, height, baseline, fill]);

  if (points.length === 0) {
    return <div className="state">No points to plot</div>;
  }
  return <canvas ref={canvasRef} role="img" aria-label={label ?? 'series'} />;
}
