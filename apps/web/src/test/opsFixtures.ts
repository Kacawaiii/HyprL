/** Real-shaped `hyprl-ops-health-v1` payloads. Synthetic values: no path, secret or private count. */
import type { OpsHealth } from '../api/types';

const SHA = 'a'.repeat(40);
const HASH = 'b'.repeat(64);
const versions = {
  git_sha: SHA, implementation_hash: HASH, api: 'trading-lab.app-api.v1',
  specs: { fomc: { revision: 25, hash: 'c'.repeat(64) }, edgar: { revision: 1, hash: 'd'.repeat(64) } },
};

export const opsHealth: OpsHealth = {
  schema: 'hyprl-ops-health-v1', read_only: true, observed_at: '2026-10-05T10:00:00+00:00', status: 'OBSERVED',
  running_versions: versions,
  sources: {
    fomc: {
      state: 'AVAILABLE', horizon: '622', attested_as_of: '2026-10-02T14:48:07+00:00', age_seconds: 7200,
      read_only: true, spec_hash: 'c'.repeat(64), read_state: 'FOMC_RESOLVED',
      freshness_method: 'archive attestation age; no claim of a live feed',
      source_health: { primary_statement: { result_state: null, reason: 'NO_FAILURE', check_at: null } },
    },
    edgar: { state: 'NOT_CONFIGURED' },
  },
  services: {
    app: { state: 'RUNNING', pid: 4242, rss_bytes: 120 * 1024 ** 2, running_versions: versions },
    workers: { state: 'STOPPED', pid: null, running_versions: null },
    edgar: { state: 'STOPPED', pid: null, running_versions: null },
  },
  last_operations: [
    { at: '2026-10-05T08:00:00+00:00', action: 'start', service: 'app', state: 'COMPLETE', code: null },
    { at: '2026-10-05T09:00:00+00:00', action: 'backup', service: 'all', state: 'BLOCKED', code: 'BACKUP_TARGET_EXISTS' },
  ],
  errors: [],
  workers: {
    state: 'OBSERVED', states: { COMPLETE: 3, RUNNING: 1 },
    workers: [{ pid: 777, progress: 0.4, limits: {} }], errors: [],
    budgets: {
      jobs_limit: 1000, jobs_used: 4, jobs_remaining: 996, artifact_bytes_limit: 128 * 1024 ** 2,
      artifact_bytes_used: 2 * 1024 ** 2, queue_limit: 8, queue_used: 1, worker_limit: 1,
    },
  },
  edgar_service: { state: 'NOT_OBSERVED', budgets: null },
  resources: {
    scope: 'API process and configured runtime volume', api_pid: 4242, api_rss_bytes: 90 * 1024 ** 2,
    api_cpu_seconds: 12.5, api_threads: 5, runtime_disk_free_bytes: 20 * 1024 ** 3,
  },
  limitations: ['no live source request', 'archive age is not a live freshness guarantee',
    'missing telemetry remains unknown', 'verify performs no workload or capture'],
};

export const degradedHealth: OpsHealth = {
  ...opsHealth, status: 'DEGRADED',
  errors: ['EDGAR_STATUS_STALE', 'FOMC_SOURCE_UNAVAILABLE'],
  workers: { state: 'NOT_OBSERVED', workers: [], budgets: null },
  edgar_service: {
    state: 'STALE', age_seconds: 90, freshness_limit_seconds: 5, grants_suspended: false,
    storage_incident: false, pending_incidents: 0,
    budgets: {
      authorization_sha256: 'e'.repeat(64), requests_limit: 100, requests_used: 100, requests_remaining: 0,
      not_after: '2026-10-04T00:00:00Z', terminated: true, expired: true,
      scope: 'original durable store; never reuse a grant with another store',
    },
  },
  resources: { ...opsHealth.resources, api_rss_bytes: null, runtime_disk_free_bytes: null },
};
