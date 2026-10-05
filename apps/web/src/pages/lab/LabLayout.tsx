/** The Lab frame: tabs for the chain dataset → experiment → model → ledger → monitoring, plus the hypothesis registry. */
import { NavLink, Outlet } from 'react-router-dom';
import { useCockpit } from '../../state/useCockpit';
import { carrySelection } from '../../lib/cockpit';
import { TokenGate } from './shared';

const TABS = [
  { to: 'datasets', label: 'Datasets' },
  { to: 'experiments', label: 'Experiments' },
  { to: 'models', label: 'Models' },
  { to: 'ledger', label: 'Predictions' },
  { to: 'monitoring', label: 'Monitoring' },
  { to: 'hypotheses', label: 'Hypotheses' },
];

export function LabLayout() {
  const { params } = useCockpit();
  const carry = carrySelection(params);
  return (
    <div className="stack">
      <section className="card" aria-label="Lab navigation">
        <div className="row" style={{ flexWrap: 'wrap', justifyContent: 'space-between' }}>
          <nav className="lab-tabs" aria-label="Lab views">
            {TABS.map((tab) => (
              <NavLink key={tab.to} to={{ pathname: tab.to, search: carry }}
                className={({ isActive }) => `tab${isActive ? ' active' : ''}`}>{tab.label}</NavLink>
            ))}
          </nav>
          <TokenGate />
        </div>
        <p className="lab-note" style={{ marginTop: 8 }}>
          Offline, synthetic and frozen-replay data only. No capture, no real training, no broker.
          The page reads; creating or cancelling a job is a command-line act the page prepares for you.
        </p>
      </section>
      <Outlet />
    </div>
  );
}
