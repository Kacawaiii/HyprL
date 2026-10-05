/** What the API documentation page lists. Hand-kept and checked against the generated OpenAPI
 *  document (docs/artifacts/b2b_openapi_v1.json) by src/test/api-docs.test.ts, so it cannot drift. */

const REPO = 'https://github.com/Kacawaiii/HyprL/blob/feat/phase5';

export const DOC_LINKS = {
  guide: { label: 'docs/API_B2B_V1.md', href: `${REPO}/docs/API_B2B_V1.md` },
  openapi: { label: 'docs/artifacts/b2b_openapi_v1.json', href: `${REPO}/docs/artifacts/b2b_openapi_v1.json` },
  client: { label: 'examples/b2b_client.py', href: `${REPO}/examples/b2b_client.py` },
} as const;

export const API_BASE = '/api/b2b/v1/projects/{project_id}';

export interface ApiStep {
  step: string;
  method: 'GET' | 'POST';
  path: string;
  permission: string;
  purpose: string;
}

/** The main path, in order: snapshot → dataset → model → prediction → monitoring. */
export const JOURNEY: ApiStep[] = [
  { step: 'Snapshot', method: 'GET', path: '/snapshots', permission: 'read',
    purpose: 'InformationSnapshot at T with per-store horizons, provenance and coverage.' },
  { step: 'Dataset', method: 'POST', path: '/datasets', permission: 'datasets:write',
    purpose: 'Build a versioned dataset (synthetic prices in this release) with its exclusions.' },
  { step: 'Model', method: 'GET', path: '/models', permission: 'read',
    purpose: 'Registered models and the capabilities each really supports.' },
  { step: 'Experiment', method: 'POST', path: '/experiments', permission: 'experiments:write',
    purpose: 'Prepare an experiment with pre-fixed criteria; it runs in an isolated worker.' },
  { step: 'Prediction', method: 'GET', path: '/experiments/{job_id}/predictions', permission: 'read',
    purpose: 'Immutable prediction records with their snapshot and feature evidence.' },
  { step: 'Monitoring', method: 'GET', path: '/observability/monitoring', permission: 'read',
    purpose: 'Missing data, technical degradation, drift and performance, kept apart.' },
];

export const LIMITS = [
  'Local offline v1 on a loopback listener; no TLS, remote hosting or billing.',
  'Every route, including the OpenAPI document, requires a project key with the listed permission.',
  'Experiments are synthetic. No capture, real training, remote model call, broker or order exists.',
  'Results demonstrate infrastructure and establish no market edge.',
];
