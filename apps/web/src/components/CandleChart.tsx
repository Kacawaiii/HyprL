/**
 * Canvas OHLC chart.
 *
 * Canvas rather than SVG on purpose: a few hundred candles is a few hundred
 * DOM nodes in SVG, each with layout and style cost, for pixels that never
 * need to be individually addressable. The server has already reduced the
 * series to at most a screenful of points, so this only has to draw.
 *
 * No indicator is computed here. The chart draws what the API returned.
 */
import { useEffect, useRef } from 'react';
import type { Candle } from '../api/types';

interface Props {
  candles: Candle[];
  height?: number;
}

function readVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name);
  return value.trim() || fallback;
}

export function CandleChart({ candles, height = 320 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || candles.length === 0) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const ratio = window.devicePixelRatio || 1;
    const width = parent.clientWidth;
    canvas.width = Math.max(1, Math.floor(width * ratio));
    canvas.height = Math.max(1, Math.floor(height * ratio));
    const context = canvas.getContext('2d');
    if (!context) return;
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);

    const positive = readVar('--positive', '#3fb950');
    const negative = readVar('--negative', '#f85149');
    const border = readVar('--border', '#232a33');
    const dim = readVar('--text-dim', '#5d6675');

    const highs = candles.map((candle) => Number(candle.high));
    const lows = candles.map((candle) => Number(candle.low));
    const top = Math.max(...highs);
    const bottom = Math.min(...lows);
    const span = top - bottom || 1;
    const padding = { top: 8, right: 56, bottom: 18, left: 8 };
    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;
    const y = (value: number) => padding.top + (1 - (value - bottom) / span) * plotH;

    context.strokeStyle = border;
    context.lineWidth = 1;
    context.font = '10px ui-monospace, monospace';
    context.fillStyle = dim;
    for (let step = 0; step <= 4; step += 1) {
      const value = bottom + (span * step) / 4;
      const line = Math.round(y(value)) + 0.5;
      context.beginPath();
      context.moveTo(padding.left, line);
      context.lineTo(padding.left + plotW, line);
      context.stroke();
      context.fillText(value.toFixed(2), padding.left + plotW + 6, line + 3);
    }

    const slot = plotW / candles.length;
    const bodyW = Math.max(1, Math.min(9, slot * 0.68));
    candles.forEach((candle, index) => {
      const openValue = Number(candle.open);
      const closeValue = Number(candle.close);
      const rising = closeValue >= openValue;
      const colour = rising ? positive : negative;
      const centre = padding.left + slot * (index + 0.5);
      context.strokeStyle = colour;
      context.fillStyle = colour;
      context.beginPath();
      context.moveTo(Math.round(centre) + 0.5, y(Number(candle.high)));
      context.lineTo(Math.round(centre) + 0.5, y(Number(candle.low)));
      context.stroke();
      const yOpen = y(openValue);
      const yClose = y(closeValue);
      const bodyTop = Math.min(yOpen, yClose);
      const bodyHeight = Math.max(1, Math.abs(yClose - yOpen));
      context.fillRect(centre - bodyW / 2, bodyTop, bodyW, bodyHeight);
    });
  }, [candles, height]);

  return (
    <div className="chart-wrap" style={{ height }}>
      <canvas ref={canvasRef} role="img" aria-label="Price history" />
    </div>
  );
}
