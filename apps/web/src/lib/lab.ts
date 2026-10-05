/**
 * Model Lab presentation helpers.
 *
 * Nothing here decides anything: it phrases what the API already established (a frozen criterion's
 * verdict, an edge's sample and method) and composes the exact command an operator runs, because the
 * server refuses browser-originated writes. Optional model outputs stay "not provided".
 */
import type {
  Classification, ExperimentResult, JobStatus, ModelContract, PerformanceBlock,
} from '../api/labTypes';

/** The synthetic generator's own limits (docs/MODEL_LAB_V1.md): it prices exactly these two products. */
export const DATASET_LIMITS = { minBars: 80, maxBars: 600, products: ['BTC-USD', 'ETH-USD'] as const };

export interface DatasetForm {
  products: string[];
  start: string;
  bars: number;
  horizonHours: number;
  seed: number;
}

export const DEFAULT_DATASET_FORM: DatasetForm = {
  products: ['BTC-USD', 'ETH-USD'], start: '2026-06-01T00:00:00+00:00', bars: 120, horizonHours: 4, seed: 7,
};

/** Validation mirrors the server ceilings so an illegal request is never composed; the server stays the authority. */
export function validateDatasetForm(form: DatasetForm): string[] {
  const problems: string[] = [];
  if (form.products.length === 0) problems.push('Select at least one product.');
  if (!Number.isInteger(form.bars) || form.bars < DATASET_LIMITS.minBars || form.bars > DATASET_LIMITS.maxBars) {
    problems.push(`Bars must be a whole number from ${DATASET_LIMITS.minBars} to ${DATASET_LIMITS.maxBars}.`);
  }
  if (!Number.isInteger(form.horizonHours) || form.horizonHours < 1 || form.horizonHours > 24) {
    problems.push('Horizon must be a whole number of hours from 1 to 24.');
  }
  if (Number.isNaN(Date.parse(form.start))) problems.push('Start must be an ISO-8601 instant.');
  if (!Number.isInteger(form.seed) || form.seed < 0) problems.push('Seed must be a non-negative whole number.');
  return problems;
}

/** The body the dataset endpoint accepts: explicitly synthetic, target fixed to forward_return. */
export function datasetRequest(form: DatasetForm): Record<string, unknown> {
  return {
    synthetic: true, products: form.products, start: form.start, bars: form.bars,
    horizon_seconds: form.horizonHours * 3600, seed: form.seed, target: 'forward_return',
  };
}

/** A command for the operator's terminal. The token is read from the environment, never written out. */
export function curlCommand(path: string, body: Record<string, unknown>, host = 'http://127.0.0.1:8787'): string {
  return [
    `curl -s -X POST ${host}${path}`,
    '  -H "Authorization: Bearer $HYPRL_MODEL_LAB_TOKEN"',
    '  -H "Content-Type: application/json"',
    `  -d '${JSON.stringify(body)}'`,
  ].join(' \\\n');
}

export const TERMINAL_STATES = new Set(['COMPLETE', 'FAILED', 'CANCELLED']);

export function isActive(job: Pick<JobStatus, 'state'>): boolean {
  return !TERMINAL_STATES.has(job.state);
}

export function progressPercent(job: Pick<JobStatus, 'progress'>): number {
  return Math.max(0, Math.min(100, Math.round(job.progress * 100)));
}

/** "ZERO" and "TRAIN_MEAN" beaten or not on the frozen criterion, per product. */
export function verdictSentence(result: ExperimentResult['result']): { product: string; met: boolean; text: string }[] {
  const { metric, rule } = result.manifest.decision_criteria;
  return Object.entries(result.criteria_met).map(([product, met]) => ({
    product,
    met,
    text: met
      ? `${product}: the criterion fixed before the run (${metric}, ${rule}) was met on synthetic data.`
      : `${product}: the criterion fixed before the run (${metric}, ${rule}) was not met. This negative result is kept.`,
  }));
}

export interface BaselineRow { name: string; mae: string | null; rmse: string | null; count: number }

/** Test-split rows for the model and each declared baseline, same decisions for every row. */
export function testComparison(result: ExperimentResult['result'], product: string): BaselineRow[] {
  const block = result.metrics[product]?.test;
  if (!block) return [];
  const order = ['model', ...result.manifest.baselines];
  const rows: BaselineRow[] = [];
  for (const name of order) {
    const entry = block[name];
    if (entry) {
      rows.push({
        name: name === 'model' ? result.manifest.parameters.model_id : name,
        mae: entry.mae, rmse: entry.rmse, count: entry.count,
      });
    }
  }
  return rows;
}

/** A model's optional outputs: which it provides and which stay absent. */
export function outputCapabilities(contract: ModelContract): { name: string; provided: boolean; note: string | null }[] {
  const names = ['return', 'target_price', 'class', 'probabilities', 'quantiles', 'scenarios'] as const;
  return names.map((name) => ({ name, provided: contract.outputs[name] !== null, note: contract.outputs[name] }));
}

export function horizonLabel(seconds: number): string {
  if (seconds % 3600 === 0) return `${seconds / 3600} h`;
  if (seconds % 60 === 0) return `${seconds / 60} min`;
  return `${seconds} s`;
}

/** Monitoring floats are JSON numbers; they are displayed, never fed back. */
export function small(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'not available';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  return magnitude < 0.001 || magnitude >= 10000 ? value.toExponential(digits - 1) : value.toPrecision(digits);
}

export const CLASS_EXPLANATION: Record<string, string> = {
  MISSING_DATA: 'Inputs are missing or late. This says nothing about the model.',
  TECHNICAL_DEGRADATION: 'Inference errors, latency or invalid outputs. The infrastructure is at fault, not the market.',
  DRIFT: 'Inputs or predictions moved away from the versioned reference.',
  PERFORMANCE_DROP: 'The measured edge over the baselines fell below its reference.',
};

export function classificationExplanation(item: Classification): string {
  return CLASS_EXPLANATION[item.category] ?? 'Unrecognised category; see the method.';
}

/** The sentence an edge claim must carry: sample and method. Never a verdict of existence. */
export function edgeSentence(block: PerformanceBlock, baseline: string): string {
  const edge = block.edge[baseline];
  if (!edge) return `No paired sample against ${baseline}.`;
  if (edge.mse_reduction === null) return `Edge against ${baseline} is not computable on ${edge.sample} pairs.`;
  const pct = (edge.mse_reduction * 100).toFixed(1);
  return `Squared error ${edge.mse_reduction >= 0 ? 'lower' : 'higher'} by ${Math.abs(Number(pct))} % than ${baseline} on ${edge.sample} paired predictions (${edge.method}). Descriptive only: ${edge.uncertainty === null ? 'no interval or significance test' : 'see uncertainty'}.`;
}

export function formatCount(sample: number): string {
  return sample.toLocaleString('en-US');
}

export function formatEpoch(seconds: number): string {
  return new Date(seconds * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
}
