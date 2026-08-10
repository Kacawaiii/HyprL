/**
 * The frontend must not re-derive anything the engines decide.
 *
 * A grep is a blunt instrument, but the failure it guards against is exactly
 * the kind that arrives quietly: someone writes `prediction > 0.0025` in a
 * component because it is obvious, and from then on the cockpit and the Python
 * engine can disagree without anyone noticing.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { MAX_CHART_POINTS, MAX_PAGE_SIZE } from '../api/client';

function sourceFiles(directory: string): string[] {
  return readdirSync(directory).flatMap((entry) => {
    const full = join(directory, entry);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return /\.tsx?$/.test(entry) && !full.includes('/test/') ? [full] : [];
  });
}

const files = sourceFiles(join(process.cwd(), 'src'));
const sources = files.map((file) => ({ file, text: readFileSync(file, 'utf8') }));

describe('no trading logic in the browser', () => {
  it('never compares a prediction against a threshold', () => {
    // The naive form is `prediction > 0.0025`, but the one that actually
    // shows up is `Number(row.prediction) > 0.0025`. Match a comparison
    // anywhere on a line that mentions a prediction, and any comparison
    // against the frozen threshold values whatever the left-hand side.
    for (const { file, text } of sources) {
      for (const line of text.split('\n')) {
        if (line.trim().startsWith('//') || line.trim().startsWith('*')) continue;
        // A prediction followed by a comparison against a number. The operator
        // must come AFTER the word, which is what distinguishes a real test
        // from the `=>` of an arrow function returning `row.prediction`.
        expect(line, `${file} :: ${line.trim()}`).not.toMatch(
          /prediction[^\n]*?[<>]=?\s*-?\.?\d/,
        );
        // ... and any comparison against the frozen thresholds, whatever the
        // left-hand side happens to be.
        expect(line, `${file} :: ${line.trim()}`).not.toMatch(/[<>]=?\s*-?0?\.0025\b/);
      }
    }
  });

  it('never computes a direction or a strength', () => {
    for (const { file, text } of sources) {
      expect(text, file).not.toMatch(/=\s*['"]LONG['"]\s*:\s*['"]SHORT['"]/);
      expect(text, file).not.toMatch(/strength\s*=\s*Math\./);
      expect(text, file).not.toMatch(/Math\.min\(1,/);
    }
  });

  it('never sizes a position or computes an exposure', () => {
    for (const { file, text } of sources) {
      expect(text, file).not.toMatch(/target_exposure\s*=\s*[^=]/);
      expect(text, file).not.toMatch(/\*\s*max_long_exposure/);
    }
  });

  it('never recomputes a benchmark metric', () => {
    for (const { file, text } of sources) {
      expect(text, file).not.toMatch(/function\s+rankIc/i);
      expect(text, file).not.toMatch(/spearman/i);
      expect(text, file).not.toMatch(/=\s*.*reduce\(.*rank/i);
    }
  });

  it('contains no order, fee, slippage or P&L concept', () => {
    for (const { file, text } of sources) {
      for (const word of ['placeOrder', 'submitOrder', 'slippage', 'commission', 'computePnl']) {
        expect(text, `${file} :: ${word}`).not.toContain(word);
      }
    }
  });
});

describe('client ceilings mirror the server', () => {
  it('states the same bounds the API enforces', () => {
    expect(MAX_PAGE_SIZE).toBe(1000);
    expect(MAX_CHART_POINTS).toBe(2000);
  });

  it('never requests more than the chart ceiling', () => {
    for (const { file, text } of sources) {
      const matches = text.match(/maxPoints:\s*(\d+)/g) ?? [];
      for (const match of matches) {
        const value = Number(match.split(':')[1]);
        expect(value, file).toBeLessThanOrEqual(MAX_CHART_POINTS);
      }
      const limits = text.match(/limit:\s*(\d+)/g) ?? [];
      for (const match of limits) {
        expect(Number(match.split(':')[1]), file).toBeLessThanOrEqual(MAX_PAGE_SIZE);
      }
    }
  });
});
