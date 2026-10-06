/** The Lab frame: tabs for the chain dataset → experiment → model → ledger → monitoring, plus the hypothesis registry. */
import { Link, NavLink, Outlet } from 'react-router-dom';
import { useCockpit } from '../../state/useCockpit';
import { carrySelection } from '../../lib/cockpit';
import { JOURNEY } from '../../lib/lab';
import { TokenGate } from './shared';

const TABS = [
  { to: 'datasets', label: 'Datasets' },
  { to: 'experiments', label: 'Experiments' },
  { to: 'models', label: 'Models' },
  { to: 'ledger', label: 'Predictions' },
  { to: 'monitoring', label: 'Monitoring' },
  { to: 'hypotheses', label: 'Hypotheses' },
];

/** The journey in order; Beginner reads what each step does, Expert gets the bare sequence. */
function Journey({ carry, expert }: { carry: string; expert: boolean }) {
  return (
    <ol className="lab-journey" aria-label="Lab journey">
      {JOURNEY.map((step, index) => (
        <li key={step.label}>
          <Link to={{ pathname: step.tab, search: carry }}><strong>{index + 1}. {step.label}</strong></Link>
          {!expert && <span className="lab-note"> {step.detail}</span>}
        </li>
      ))}
    </ol>
  );
}

export function LabLayout() {
  const { params, selection } = useCockpit();
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
        <Journey carry={carry} expert={selection.mode === 'expert'} />
        <p className="lab-note" style={{ marginTop: 8 }}>
          Offline, synthetic and frozen-replay data only. No capture, no real training, no broker. With the operator
          token, this page builds datasets, launches, cancels and monitors synthetic jobs through the local lab listener,
          which runs them in isolated workers; real-data training is WAITING_AUTHORIZATION.
        </p>
      </section>
      <Outlet />
    </div>
  );
}
