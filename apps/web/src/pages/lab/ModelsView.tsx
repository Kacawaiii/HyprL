/** Model registry with declared capabilities. A capability is shown only because the contract declares it. */
import { Badge, EmptyState, Hash } from '../../components/States';
import { apiClient } from '../../api/client';
import type { ModelDescriptor } from '../../api/labTypes';
import { horizonLabel, outputCapabilities } from '../../lib/lab';
import { useCockpit } from '../../state/useCockpit';
import { useLabToken } from '../../state/labToken';
import { useQuery } from '../../state/useQuery';
import { LockedState, QueryBoundary, Synthetic } from './shared';

const CAPABILITIES = ['train', 'predict', 'serialize', 'infer'];

function role(model: ModelDescriptor): { label: string; tone: 'ok' | 'off' | 'warn' } {
  const limits = model.contract.limits;
  if (limits.shadow_only === true) return { label: 'FROZEN REFERENCE (paper)', tone: 'ok' };
  if (model.registration.startsWith('external')) return { label: 'EXTERNAL ADAPTER (local, demonstration)', tone: 'warn' };
  return { label: 'INTERNAL DEMONSTRATION', tone: 'warn' };
}

export function ModelsView() {
  const { token, version } = useLabToken();
  const { selection, update } = useCockpit();
  const models = useQuery(token ? `lab:models:${version}` : null, (signal) => apiClient.getLabModels(token, signal));
  if (!token) return <LockedState what="The model registry" />;
  return (
    <QueryBoundary query={models} label="Loading models">
      {(data) => (
        <div className="stack">
          <p className="lab-note">
            External registration: {data.external_registration}. A page or HTTP request cannot register code; an adapter is installed by the operator
            and then declares its inputs, outputs, horizons, limits and capabilities below.
          </p>
          {data.models.length === 0 && <EmptyState title="No model registered" />}
          <div className="grid grid-2">
            {data.models.map((model) => {
              const contract = model.contract;
              const kind = role(model);
              const selected = selection.model === contract.model_id;
              return (
                <article className="card" key={contract.model_id} aria-label={`Model ${contract.model_id}`}>
                  <div className="row" style={{ flexWrap: 'wrap' }}>
                    <h3 className="card-title" style={{ margin: 0 }}>{contract.model_id}</h3>
                    <Badge tone={kind.tone}>{kind.label}</Badge>
                    {contract.synthetic && <Synthetic />}
                  </div>
                  <p className="lab-note" style={{ margin: '8px 0' }}>{String(contract.limits.method ?? contract.limits.training)}</p>
                  <div className="row" style={{ flexWrap: 'wrap' }} aria-label="Capabilities">
                    {CAPABILITIES.map((name) => (
                      <Badge key={name} tone={contract.capabilities.includes(name) ? 'ok' : 'off'}>
                        {name.toUpperCase()}: {contract.capabilities.includes(name) ? 'YES' : 'NO'}
                      </Badge>
                    ))}
                  </div>
                  <dl style={{ marginTop: 10 }}>
                    <div className="kv"><dt>Horizons</dt><dd>{contract.horizons_seconds.map(horizonLabel).join(', ')}</dd></div>
                    <div className="kv"><dt>Outputs provided</dt><dd>{outputCapabilities(contract).filter((o) => o.provided).map((o) => o.name).join(', ')}</dd></div>
                    <div className="kv"><dt>Outputs not provided</dt><dd>{outputCapabilities(contract).filter((o) => !o.provided).map((o) => o.name).join(', ') || '—'}</dd></div>
                    <div className="kv"><dt>Calibrated probabilities</dt><dd>{contract.limits.calibrated === true ? 'yes' : 'no'}</dd></div>
                    <div className="kv"><dt>Remote calls</dt><dd>{contract.limits.remote_calls === true ? 'yes' : 'none'}</dd></div>
                  </dl>
                  {selection.mode === 'expert' && (
                    <details>
                      <summary>Contract</summary>
                      <dl>
                        <div className="kv"><dt>Version</dt><dd>{contract.version} / {contract.implementation_version}</dd></div>
                        <div className="kv"><dt>Fingerprint</dt><dd><Hash value={model.fingerprint} chars={20} /></dd></div>
                        <div className="kv"><dt>Registration</dt><dd>{model.registration}</dd></div>
                        <div className="kv"><dt>Target</dt><dd>{contract.inputs.target}</dd></div>
                        <div className="kv"><dt>Products</dt><dd>{contract.inputs.products.join(', ')}</dd></div>
                        <div className="kv"><dt>Features</dt><dd>{contract.inputs.features.join(', ')}</dd></div>
                        {Object.entries(contract.limits).map(([key, value]) => (
                          <div className="kv" key={key}><dt>limit: {key}</dt><dd>{String(value)}</dd></div>
                        ))}
                      </dl>
                    </details>
                  )}
                  <div style={{ marginTop: 10 }}>
                    <button className="control" aria-pressed={selected} onClick={() => update({ model: selected ? null : contract.model_id })}>
                      {selected ? 'Selected for the ledger and monitoring' : 'Select this model'}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      )}
    </QueryBoundary>
  );
}
