import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { AppShell } from './layouts/AppShell';
import { OverviewPage } from './pages/OverviewPage';
import { RadarHomePage } from './pages/RadarHomePage';
import { LoadingState } from './components/States';

// Route-level splitting: the chart and research views should not weigh on a
// first paint of Overview.
const CockpitPage = lazy(() =>
  import('./pages/CockpitPage').then((module) => ({ default: module.CockpitPage })));
const MarketsPage = lazy(() =>
  import('./pages/MarketsPage').then((module) => ({ default: module.MarketsPage })));
const NewsValuePage = lazy(() =>
  import('./pages/NewsValuePage').then((module) => ({ default: module.NewsValuePage })));
const SignalsPage = lazy(() =>
  import('./pages/SignalsPage').then((module) => ({ default: module.SignalsPage })));
const RiskPage = lazy(() =>
  import('./pages/RiskPage').then((module) => ({ default: module.RiskPage })));
const PoliciesPage = lazy(() =>
  import('./pages/PoliciesPage').then((module) => ({ default: module.PoliciesPage })));
const PaperPage = lazy(() =>
  import('./pages/PaperPage').then((module) => ({ default: module.PaperPage })));
const BacktestsPage = lazy(() =>
  import('./pages/BacktestsPage').then((module) => ({ default: module.BacktestsPage })));
const ResearchPage = lazy(() =>
  import('./pages/ResearchPage').then((module) => ({ default: module.ResearchPage })));
const SystemPage = lazy(() =>
  import('./pages/SystemPage').then((module) => ({ default: module.SystemPage })));
const PortfolioPage = lazy(() =>
  import('./pages/PortfolioPage').then((module) => ({ default: module.PortfolioPage })));
const EventsPage = lazy(() =>
  import('./pages/EventsPage').then((module) => ({ default: module.EventsPage })));
const LabLayout = lazy(() =>
  import('./pages/lab/LabLayout').then((module) => ({ default: module.LabLayout })));
const DatasetsView = lazy(() =>
  import('./pages/lab/DatasetsView').then((module) => ({ default: module.DatasetsView })));
const ExperimentsView = lazy(() =>
  import('./pages/lab/ExperimentsView').then((module) => ({ default: module.ExperimentsView })));
const ModelsView = lazy(() =>
  import('./pages/lab/ModelsView').then((module) => ({ default: module.ModelsView })));
const LedgerView = lazy(() =>
  import('./pages/lab/LedgerView').then((module) => ({ default: module.LedgerView })));
const MonitoringView = lazy(() =>
  import('./pages/lab/MonitoringView').then((module) => ({ default: module.MonitoringView })));
const HypothesesView = lazy(() =>
  import('./pages/lab/HypothesesView').then((module) => ({ default: module.HypothesesView })));
const ApiDocsPage = lazy(() =>
  import('./pages/ApiDocsPage').then((module) => ({ default: module.ApiDocsPage })));
const TraderPage = lazy(() =>
  import('./pages/TraderPage').then((module) => ({ default: module.TraderPage })));
const SettingsPage = lazy(() =>
  import('./pages/SettingsPage').then((module) => ({ default: module.SettingsPage })));

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<RadarHomePage />} />
        <Route path="overview" element={<OverviewPage />} />
        <Route path="news-value" element={<Suspense fallback={<LoadingState />}><NewsValuePage /></Suspense>} />
        <Route
          path="cockpit"
          element={<Suspense fallback={<LoadingState />}><CockpitPage /></Suspense>}
        />
        <Route
          path="markets"
          element={<Suspense fallback={<LoadingState />}><MarketsPage /></Suspense>}
        />
        <Route
          path="signals"
          element={<Suspense fallback={<LoadingState />}><SignalsPage /></Suspense>}
        />
        <Route
          path="risk"
          element={<Suspense fallback={<LoadingState />}><RiskPage /></Suspense>}
        />
        <Route path="policies" element={<Suspense fallback={<LoadingState />}><PoliciesPage /></Suspense>} />
        <Route
          path="paper"
          element={<Suspense fallback={<LoadingState />}><PaperPage /></Suspense>}
        />
        <Route
          path="backtests"
          element={<Suspense fallback={<LoadingState />}><BacktestsPage /></Suspense>}
        />
        <Route
          path="research"
          element={<Suspense fallback={<LoadingState />}><ResearchPage /></Suspense>}
        />
        <Route
          path="trader"
          element={<Suspense fallback={<LoadingState />}><TraderPage /></Suspense>}
        />
        <Route
          path="events"
          element={<Suspense fallback={<LoadingState />}><EventsPage /></Suspense>}
        />
        <Route
          path="system"
          element={<Suspense fallback={<LoadingState />}><SystemPage /></Suspense>}
        />
        <Route
          path="portfolio"
          element={<Suspense fallback={<LoadingState />}><PortfolioPage /></Suspense>}
        />
        <Route
          path="api-docs"
          element={<Suspense fallback={<LoadingState />}><ApiDocsPage /></Suspense>}
        />
        <Route
          path="settings"
          element={<Suspense fallback={<LoadingState />}><SettingsPage /></Suspense>}
        />
        <Route path="lab" element={<Suspense fallback={<LoadingState />}><LabLayout /></Suspense>}>
          <Route index element={<Navigate to="datasets" replace />} />
          <Route path="datasets" element={<Suspense fallback={<LoadingState />}><DatasetsView /></Suspense>} />
          <Route path="experiments" element={<Suspense fallback={<LoadingState />}><ExperimentsView /></Suspense>} />
          <Route path="models" element={<Suspense fallback={<LoadingState />}><ModelsView /></Suspense>} />
          <Route path="ledger" element={<Suspense fallback={<LoadingState />}><LedgerView /></Suspense>} />
          <Route path="monitoring" element={<Suspense fallback={<LoadingState />}><MonitoringView /></Suspense>} />
          <Route path="hypotheses" element={<Suspense fallback={<LoadingState />}><HypothesesView /></Suspense>} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
