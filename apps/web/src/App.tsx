import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { AppShell } from './layouts/AppShell';
import { OverviewPage } from './pages/OverviewPage';
import { LoadingState } from './components/States';

// Route-level splitting: the chart and research views should not weigh on a
// first paint of Overview.
const MarketsPage = lazy(() =>
  import('./pages/MarketsPage').then((module) => ({ default: module.MarketsPage })));
const SignalsPage = lazy(() =>
  import('./pages/SignalsPage').then((module) => ({ default: module.SignalsPage })));
const RiskPage = lazy(() =>
  import('./pages/RiskPage').then((module) => ({ default: module.RiskPage })));
const ResearchPage = lazy(() =>
  import('./pages/ResearchPage').then((module) => ({ default: module.ResearchPage })));
const SystemPage = lazy(() =>
  import('./pages/SystemPage').then((module) => ({ default: module.SystemPage })));

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<OverviewPage />} />
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
        <Route
          path="research"
          element={<Suspense fallback={<LoadingState />}><ResearchPage /></Suspense>}
        />
        <Route
          path="system"
          element={<Suspense fallback={<LoadingState />}><SystemPage /></Suspense>}
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
