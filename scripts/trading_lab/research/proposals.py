"""Small deterministic proposal catalogue; never executes or contacts providers."""
from scripts.trading_lab.platform.contracts import DatasetManifest
from scripts.trading_lab.research.contracts import Hypothesis
from scripts.trading_lab.sources.canonical import sha256_canonical

ENGINE = 'local-rule-proposals-v1'


def experiment_hypothesis(dataset, prepared, *, max_trials=2):
    manifest = DatasetManifest.from_dict(dataset['manifest'])
    hypothesis = prepared.to_dict()['hypothesis']
    payload = dict(version=ENGINE, statement=hypothesis['statement'], mechanism=hypothesis['mechanism'],
        falsification=hypothesis['falsification'], sources=('synthetic_prices',),
        features={'columns': list(manifest.policies['columns']), 'definition_hash': manifest.features_hash},
        target=manifest.target, horizon_seconds=manifest.horizon_seconds,
        population={'products': list(manifest.products), 'start': manifest.decision_start, 'end': manifest.decision_end,
                    'dataset_hash': manifest.identity, 'rows': manifest.counts['included'], 'studied_window': True},
        baselines=prepared.baselines, splits=prepared.to_dict()['splits'],
        transformations={'fit_on': 'TRAIN_ONLY', 'refit_after_validation': False,
                         'method': prepared.parameters['transformations']},
        costs=prepared.to_dict()['costs'], calendar={'id': manifest.policies['calendar']},
        availability={'rule': manifest.policies['availability'], 'exclusions_hash': sha256_canonical(manifest.to_dict()['exclusions']),
                      'protection_hash': manifest.policies['protection_hash']},
        decision_criteria=prepared.to_dict()['decision_criteria'],
        budgets={'max_trials': max_trials, 'max_rows': 600, 'wall_seconds': prepared.budgets['wall_seconds']},
        scope='EXPLORATORY', synthetic=True)
    if manifest.counts['included'] > payload['budgets']['max_rows']:
        raise ValueError('proposal population exceeds research row budget')
    return Hypothesis(hypothesis_id='synthetic-hypothesis-' + sha256_canonical(payload)[:24], **payload)


def propose(dataset, *, limit=2, max_trials=2):
    if type(limit) is not int or not 1 <= limit <= 2 or type(max_trials) is not int or not 1 <= max_trials <= 8:
        raise ValueError('local proposal limit is 1..2, trial limit is 1..8')
    from scripts.trading_lab.platform.experiments import prepare_experiment
    templates = (
        ('synthetic-ridge-v1', 'Lagged price features reduce synthetic return MAE relative to fixed baselines',
         'Persistent synthetic price dynamics can be represented by a regularized linear model'),
        ('local-momentum-v1', 'Four times the hourly return reduces synthetic four-hour MAE relative to fixed baselines',
         'A recent synthetic price move may continue across the next four hourly bars'))
    proposals = []
    for model, statement, mechanism in templates[:limit]:
        prepared = prepare_experiment(dataset, model_id=model, hypothesis={
            'statement': statement, 'mechanism': mechanism,
            'falsification': 'test MAE fails to beat both ZERO and TRAIN_MEAN', 'scientific_claim': False})
        plan = experiment_hypothesis(dataset, prepared, max_trials=max_trials)
        proposals.append({'hypothesis': plan, 'prepared': prepared})
    return proposals


def comparison_hypothesis():
    from scripts.trading_lab.comparison_protocol import build_v2
    protocol = build_v2()
    return Hypothesis(hypothesis_id='prices-vs-events-protocol-v2', version=ENGINE,
        statement='Attested events improve crypto test MSE beyond prices on the same admissible decisions',
        mechanism='Newly observed FOMC information and revisions can change expected forward returns',
        falsification=protocol['evaluation']['primary']['decision_rule']['NOT_SUPPORTED'],
        sources=('fomc', 'edgar', 'coinbase'), features=protocol['pairing']['features'],
        target='forward_return', horizon_seconds=14400,
        population={'products': protocol['products'], 'studied_windows': 'EXPLORATORY',
                    'new_evaluation': protocol['calendar']['evaluation'], 'admissibility': protocol['admissible_decision']},
        baselines=('PRICES_ONLY', 'ZERO', 'TRAIN_MEAN'), splits=protocol['calendar']['split'],
        transformations={'fit_on': 'TRAIN_ONLY', 'method': protocol['pairing']['model']['fitting'],
                         'embargo': 'only the purge fixed by protocol V2; no additional embargo'},
        costs=protocol['pairing']['frozen_reused'], calendar=protocol['calendar'],
        availability={'warmup': protocol['warmup'], 'pairing': protocol['pairing']['rule'], 'protection': protocol['protection']},
        decision_criteria=protocol['evaluation'], budgets={'max_trials': 1, 'max_rows': 10000, 'wall_seconds': 180,
            'execution_enabled': False, 'protocol_budgets': protocol['requirements']['budgets']},
        scope='PREREGISTERED_PROTOCOL', synthetic=False, protocol_hash=protocol['protocol_hash'])


def pair_decisions(prices_only, prices_events):
    """Check exact populations, labels, splits, costs and clocks; no intersection repair."""
    def indexed(rows):
        result = {}
        for row in rows:
            key = (row['product'], row['decision_at'])
            if key in result:
                raise ValueError('duplicate comparison decision')
            result[key] = row
        return result
    a, b = indexed(prices_only), indexed(prices_events)
    if set(a) != set(b):
        raise ValueError('comparison decisions must be identical; exclude unavailable decisions from both variants upstream')
    for key in a:
        for field in ('label', 'label_available_at', 'split', 'costs_hash'):
            if a[key][field] != b[key][field]:
                raise ValueError('comparison label, availability, split or cost mismatch')
        if not a[key].get('admissible') or not b[key].get('admissible'):
            raise ValueError('comparison includes an inadmissible decision')
    keys = sorted(a)
    return {'method': 'exact-admissible-pairs-v1', 'decisions': len(keys),
            'population_hash': sha256_canonical([[*key, a[key]['label'], a[key]['split'], a[key]['costs_hash']] for key in keys]),
            'scope': 'EXPLORATORY unless the frozen protocol is separately authorized'}


def comparison_diagnostic():
    from scripts.trading_lab.comparison_protocol import build_v2
    protocol = build_v2()
    return {'schema': 'comparison-readiness-v1', 'protocol_hash': protocol['protocol_hash'],
        'state': 'WAITING_DATA', 'execution_enabled': False, 'paired_decisions': 0,
        'reasons': ['NO_ATTESTED_EVENT_PRICE_OVERLAP', 'PROTOCOL_NOT_OPERATOR_ACCEPTED', 'CAPTURE_AND_TRAINING_NOT_AUTHORIZED'],
        'actions': ['operator reviews frozen COMPARISON_PROTOCOL_V2',
            'operator supplies scoped capture and training authorizations for the fixed windows and budgets',
            'obtain causally attested events and matching prices; backfills retain their actual acquisition availability',
            'build identical admissible A/B populations with exclusions in both arms'],
        'studied_windows': 'EXPLORATORY', 'crypto_role': 'PRIMARY_CONFIRMATORY', 'equity_role': 'EXPLORATORY_NO_CLAIM',
        'limitations': ['readiness is the registered campaign status, not a scan of arbitrary stores',
                        'this module cannot capture, train the real variants or unlock holdouts']}
