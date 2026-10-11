import { ApiError } from '../api/client';
import { EmptyState, ErrorState, LoadingState } from './States';

export function AnalysisState({ status, error, retry }: { status: string; error?: Error; retry: () => void }) {
  if (status === 'loading') return <LoadingState label="Chargement du snapshot cockpit" />;
  if (error instanceof ApiError && [404, 503].includes(error.status)) return <EmptyState title="Snapshot phase 2 indisponible"
    detail="L’export en lecture seule doit publier analysis.json. Aucune donnée n’est estimée." />;
  if (error) return <ErrorState error={error} onRetry={retry} />;
  return null;
}
