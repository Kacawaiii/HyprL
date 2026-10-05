import { describe, expect, it } from 'vitest';
import {
  DEFAULT_DATASET_FORM, curlCommand, datasetRequest, edgeSentence, horizonLabel, isActive, outputCapabilities,
  progressPercent, small, testComparison, validateDatasetForm, verdictSentence,
} from '../lib/lab';
import { lab } from './labFixtures';

describe('dataset form', () => {
  it('accepts the default synthetic configuration', () => {
    expect(validateDatasetForm(DEFAULT_DATASET_FORM)).toEqual([]);
  });
  it('mirrors the server ceilings instead of composing an illegal request', () => {
    expect(validateDatasetForm({ ...DEFAULT_DATASET_FORM, bars: 79 })).toHaveLength(1);
    expect(validateDatasetForm({ ...DEFAULT_DATASET_FORM, bars: 601 })).toHaveLength(1);
    expect(validateDatasetForm({ ...DEFAULT_DATASET_FORM, horizonHours: 0 })).toHaveLength(1);
    expect(validateDatasetForm({ ...DEFAULT_DATASET_FORM, products: [] })).toHaveLength(1);
    expect(validateDatasetForm({ ...DEFAULT_DATASET_FORM, start: 'not a date' })).toHaveLength(1);
  });
  it('builds only the explicitly synthetic body the endpoint accepts', () => {
    const body = datasetRequest(DEFAULT_DATASET_FORM);
    expect(body).toEqual({
      synthetic: true, products: ['BTC-USD', 'ETH-USD'], start: '2026-06-01T00:00:00+00:00', bars: 120,
      horizon_seconds: 14400, seed: 7, target: 'forward_return',
    });
  });
});

describe('prepared commands', () => {
  it('reads the token from the environment and never embeds one', () => {
    const command = curlCommand('/api/v1/lab/jobs/abc/cancel', {});
    expect(command).toContain('$HYPRL_MODEL_LAB_TOKEN');
    expect(command).toContain("-d '{}'");
  });
});

describe('experiment verdicts', () => {
  const result = lab.experimentResult.result;
  it('states a failed criterion as retained negative result', () => {
    const verdicts = verdictSentence(result);
    expect(verdicts).toHaveLength(2);
    expect(verdicts.every((verdict) => !verdict.met)).toBe(true);
    expect(verdicts[0]?.text).toContain('negative result is kept');
    expect(verdicts[0]?.text).toContain('test_mae');
  });
  it('compares the model with every declared baseline on the same decisions', () => {
    const rows = testComparison(result, 'BTC-USD');
    expect(rows.map((row) => row.name)).toEqual(['local-momentum-v1', 'ZERO', 'TRAIN_MEAN']);
    expect(new Set(rows.map((row) => row.count)).size).toBe(1);
    expect(testComparison(result, 'SOL-USD')).toEqual([]);
  });
});

describe('model capabilities', () => {
  it('leaves absent optional outputs absent', () => {
    const contract = lab.labModels.models[0].contract;
    const outputs = outputCapabilities(contract);
    expect(outputs.filter((output) => output.provided).map((output) => output.name)).toEqual(['return']);
    expect(outputs.filter((output) => !output.provided)).toHaveLength(5);
  });
  it('labels horizons readably', () => {
    expect(horizonLabel(14400)).toBe('4 h');
    expect(horizonLabel(900)).toBe('15 min');
    expect(horizonLabel(45)).toBe('45 s');
  });
});

describe('jobs', () => {
  it('knows active from terminal states and clamps progress', () => {
    expect(isActive({ state: 'RUNNING' })).toBe(true);
    expect(isActive({ state: 'COMPLETE' })).toBe(false);
    expect(isActive({ state: 'CANCELLED' })).toBe(false);
    expect(progressPercent({ progress: 1.7 })).toBe(100);
    expect(progressPercent({ progress: -1 })).toBe(0);
    expect(progressPercent({ progress: 0.456 })).toBe(46);
  });
});

describe('monitoring phrasing', () => {
  const performance = lab.monitoring.performance;
  it('names the sample, the method and that nothing is concluded', () => {
    const text = edgeSentence(performance, 'ZERO');
    expect(text).toContain(`${performance.edge.ZERO.sample} paired predictions`);
    expect(text).toContain('paired-descriptive-mse-reduction-v1');
    expect(text).toContain('Descriptive only');
  });
  it('does not invent an edge without a pair', () => {
    expect(edgeSentence({ ...performance, edge: {} }, 'ZERO')).toBe('No paired sample against ZERO.');
  });
  it('shows absent numbers as unavailable, never as zero', () => {
    expect(small(null)).toBe('not available');
    expect(small(undefined)).toBe('not available');
    expect(small(Number.NaN)).toBe('not available');
    expect(small(0)).toBe('0');
    expect(small(0.0000123)).toMatch(/e-5$/);
  });
});
