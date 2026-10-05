"""Bounded read-only research and observability views; HTTP never runs workloads."""
import re

from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.app_api.pagination import decode_cursor, encode_cursor, require_limit
from scripts.trading_lab.platform.contracts import digest, timestamp
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.research.store import ResearchStore, IntegrityError, now
from scripts.trading_lab.research.monitoring import monitor, METHOD, REGIME
from scripts.trading_lab.research.proposals import comparison_diagnostic, comparison_hypothesis, ENGINE


class ResearchApiError(AppApiError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class ResearchViews:
    def __init__(self, root=None):
        self._root = root
        self._comparison = None

    def _store(self):
        if self._root is None:
            raise ResearchApiError('research observability store not configured', 503)
        try:
            return ResearchStore(self._root, read_only=True)
        except FileNotFoundError:
            raise ResearchApiError('research observability store unavailable', 503) from None

    @staticmethod
    def _query(query, allowed):
        if set(query) - set(allowed) or any(len(v) != 1 or not v[0] for v in query.values()):
            raise ResearchApiError('unknown or repeated research query parameter')
        return {k: v[0] for k, v in query.items()}

    def dispatch(self, path, query):
        try:
            return self._dispatch(path, query)
        except ResearchApiError:
            raise
        except IntegrityError:
            raise ResearchApiError('research evidence integrity failure', 409) from None
        except KeyError:
            raise ResearchApiError('research resource not available at this instant', 404) from None
        except (ValueError, TypeError, OverflowError):
            raise ResearchApiError('invalid research selection, reference or page bounds', 400) from None

    def _dispatch(self, path, query):
        if path == '/api/v1/research/comparison':
            self._query(query, ())
            if self._comparison is None:
                plan = comparison_hypothesis()
                self._comparison = {**comparison_diagnostic(), 'hypothesis': plan.to_dict(),
                                    'hypothesis_hash': plan.identity, 'criteria_hash': plan.criteria_hash}
            return self._comparison
        if path == '/api/v1/research/proposals':
            self._query(query, ())
            return {'schema': 'research-proposal-catalogue-v1', 'engine': ENGINE, 'external_model_calls': 0,
                    'execution_enabled': False, 'max_proposals': 2, 'max_trials_per_hypothesis': 8,
                    'models': ['synthetic-ridge-v1', 'local-momentum-v1'], 'synthetic_only': True,
                    'preparation': 'offline research.demo or Python propose(dataset); no HTTP write'}
        if path in {'/api/v1/research/hypotheses', '/api/v1/research/experiments',
                    '/api/v1/observability/predictions', '/api/v1/observability/references', '/api/v1/observability/replays'}:
            values = self._query(query, ('as_of', 'limit', 'cursor', 'product', 'model_id', 'start', 'end'))
            at = timestamp(values.get('as_of') or now())
            size = require_limit(values.get('limit'), default=100, maximum=200)
            kind = {'hypotheses': 'hypothesis', 'experiments': 'trial', 'predictions': 'prediction',
                    'references': 'reference', 'replays': 'replay-summary'}[path.rsplit('/', 1)[-1]]
            if kind != 'prediction' and set(values) & {'product', 'model_id', 'start', 'end'}:
                raise ValueError('filters only apply to predictions')
            selection = {k: values[k] for k in ('product', 'model_id', 'start', 'end') if k in values}
            if selection.get('start'):
                selection['start'] = timestamp(selection['start'])
            if selection.get('end'):
                selection['end'] = timestamp(selection['end'])
            if selection.get('start') and selection.get('end') and selection['start'] >= selection['end']:
                raise ValueError('empty interval')
            binding = {'schema': 'research-page-v1', 'as_of': at, **selection}
            if values.get('cursor') and not values.get('as_of'):
                raise ValueError('continuation requires the original as_of')
            after = int(decode_cursor(values['cursor'], endpoint=kind, product='research', query=binding)) if values.get('cursor') else 0
            store = self._store()
            selected = []
            # Scan chunks so a product filter cannot silently drop later matching rows.
            position = after
            while len(selected) <= size:
                rows = store.records(kind, as_of=at, after=position, limit=500)
                if not rows:
                    break
                for item in rows:
                    p = item['payload']
                    if kind == 'prediction' and ((selection.get('product') and p['product'] != selection['product'])
                            or (selection.get('model_id') and p['model_id'] != selection['model_id'])
                            or (selection.get('start') and p['decision_at'] < selection['start'])
                            or (selection.get('end') and p['decision_at'] >= selection['end']) or p['decision_at'] > at):
                        continue
                    selected.append(item)
                    if len(selected) > size:
                        break
                position = rows[-1]['sequence']
                if len(rows) < 500:
                    break
            page, more = selected[:size], len(selected) > size
            if kind == 'prediction':
                compact = []
                for item in page:
                    view = store.prediction_view(item['identity'], as_of=at)
                    compact.append({**item, 'view': {
                        'prediction_hash': view['prediction_hash'], 'label_state': view['label_state'],
                        'latest_label': view['labels'][-1] if view['labels'] else None,
                        'label_versions': len(view['labels']), 'execution_state': view['execution_state'],
                        'latest_execution': view['executions'][-1] if view['executions'] else None,
                        'latest_decision': view['decisions'][-1] if view['decisions'] else None,
                        'input_quality': view['inputs']['input_quality'] if view['inputs'] else None,
                        'detail_path': '/api/v1/observability/predictions/' + item['identity']}})
                page = compact
            return {'schema': 'research-page-v1', 'as_of': at, 'records': page,
                'page': {'returned': len(page), 'has_more': more,
                         'next_cursor': encode_cursor(endpoint=kind, product='research',
                            last_timestamp=str(page[-1]['sequence']), query=binding) if more else None}}
        if match := re.fullmatch(r'/api/v1/research/hypotheses/([a-f0-9]{64})', path):
            values = self._query(query, ('as_of',))
            at = timestamp(values.get('as_of') or now())
            store = self._store()
            row = store.get(match[1], kind='hypothesis')
            if row['recorded_at'] > at:
                raise KeyError('hypothesis not recorded')
            return {**row, 'criteria_hash': sha256_canonical(row['payload']['decision_criteria']),
                    'trial_history': store.records('trial', parent_id=match[1], as_of=at)}
        if match := re.fullmatch(r'/api/v1/research/experiments/([a-f0-9]{64})', path):
            values = self._query(query, ('as_of',))
            at = timestamp(values.get('as_of') or now())
            store = self._store()
            row = store.get(match[1], kind='experiment')
            if row['recorded_at'] > at:
                raise KeyError('experiment not recorded')
            prepared_hash = row['payload']['artifacts'].get('prepared_hash', row['identity'])
            history = [r for r in store.records('trial', as_of=at) if r['payload']['experiment_hash'] == prepared_hash]
            return {**row, 'criteria_hash': sha256_canonical(row['payload']['decision_criteria']), 'trial_history': history}
        if match := re.fullmatch(r'/api/v1/observability/references/([a-f0-9]{64})', path):
            values = self._query(query, ('as_of',))
            at = timestamp(values.get('as_of') or now())
            row = self._store().get(match[1], kind='reference')
            if row['recorded_at'] > at:
                raise KeyError('reference not recorded')
            return row
        if match := re.fullmatch(r'/api/v1/observability/predictions/([a-f0-9]{64})', path):
            values = self._query(query, ('as_of',))
            return self._store().prediction_view(match[1], as_of=values.get('as_of') or now())
        if path == '/api/v1/observability/monitoring':
            values = self._query(query, ('as_of', 'product', 'model_id', 'start', 'end', 'split', 'artifact_hash', 'reference_hash'))
            if 'as_of' not in values:
                raise ValueError('monitoring requires explicit as_of')
            for name in ('artifact_hash', 'reference_hash'):
                if name in values:
                    digest(values[name])
            return monitor(self._store(), **values)
        if path == '/api/v1/observability/health':
            self._query(query, ())
            store = self._store()
            return {**store.verify(), 'read_only': True, 'method': METHOD, 'regime_definition': REGIME,
                    'network_requests': 0, 'workload_execution': False}
        raise ResearchApiError('no such research endpoint', 404)
