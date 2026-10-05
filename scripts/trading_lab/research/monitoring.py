"""Causal, cohort-bound descriptive monitoring; no calibrated uncertainty claims."""
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import log, sqrt
from typing import ClassVar, Mapping

from scripts.trading_lab.platform.contracts import Contract, digest, timestamp
from scripts.trading_lab.research.contracts import number
from scripts.trading_lab.sources.canonical import sha256_canonical

METHOD = {'version': 'monitoring-method-v1', 'distribution': 'finite-observed-summary-v1',
    'drift': 'reference-quantile-psi-and-two-sample-ks-statistic-v1', 'bins': 5,
    'pseudocount': 0.000001, 'psi_threshold': 0.2, 'ks_threshold': 0.3, 'minimum_sample': 10,
    'edge': 'paired-descriptive-mse-reduction-v1', 'performance_drop': 'edge decline > 0.05 vs reference',
    'freshness_seconds': 7200, 'latency_ms': 1000, 'error_rate': 0.05,
    'uncertainty': None, 'limitations': ['descriptive monitoring thresholds; no statistical edge verdict',
        'overlapping horizons and serial dependence are not corrected by an interval',
        'reference and current samples must share model, artifact, product and horizon']}
REGIME = {'version': 'price-regimes-v1', 'definition': [
    'HIGH_VOL if atr_pct_14 >= 0.03', 'otherwise UP if return_4 >= 0.01',
    'otherwise DOWN if return_4 <= -0.01', 'otherwise RANGE',
    'UNKNOWN when either feature is unavailable'], 'inputs': ['return_4', 'atr_pct_14'],
    'clock': 'features actually used at decision time; never realized labels'}


@dataclass(frozen=True, kw_only=True)
class MonitoringReference(Contract):
    schema: ClassVar[str] = 'monitoring-reference-v1'
    reference_id: str
    version: str
    binding: Mapping
    as_of: str
    selection: Mapping
    population_hash: str
    predictions: tuple
    features: Mapping
    baseline_edges: Mapping
    method: Mapping
    synthetic: bool

    def validate(self):
        object.__setattr__(self, 'as_of', timestamp(self.as_of))
        digest(self.population_hash)
        for key in ('model_contract_hash', 'artifact_hash'):
            digest(self.binding[key])
        if not self.selection or not self.predictions or self.to_dict()['method'] != METHOD:
            raise ValueError('reference requires the versioned method and observed population')
        for value in self.predictions:
            number(value)
        for values in self.features.values():
            for value in values:
                number(value)


def regime(features):
    values = dict(features or ())
    if values.get('return_4') is None or values.get('atr_pct_14') is None:
        return 'UNKNOWN'
    if number(values['atr_pct_14']) >= .03:
        return 'HIGH_VOL'
    ret = number(values['return_4'])
    return 'UP' if ret >= .01 else 'DOWN' if ret <= -.01 else 'RANGE'


def distribution(values):
    ordered = sorted(number(x) for x in values)
    if not ordered:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'std': None, 'p50': None, 'p95': None}
    mean = sum(ordered) / len(ordered)
    def quantile(p):
        location = (len(ordered) - 1) * p
        low = int(location)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (location - low)
    return {'count': len(ordered), 'min': ordered[0], 'max': ordered[-1], 'mean': mean,
            'std': sqrt(sum((v - mean) ** 2 for v in ordered) / len(ordered)), 'p50': quantile(.5), 'p95': quantile(.95)}


def drift(reference, current):
    ref, cur = sorted(map(number, reference)), sorted(map(number, current))
    if min(len(ref), len(cur)) < METHOD['minimum_sample']:
        return {'state': 'INSUFFICIENT_SAMPLE', 'reference_count': len(ref), 'current_count': len(cur),
                'psi': None, 'ks_statistic': None, 'method': METHOD['drift'], 'minimum_sample': METHOD['minimum_sample']}
    boundaries = sorted({ref[min(len(ref) - 1, len(ref) * i // METHOD['bins'])] for i in range(1, METHOD['bins'])})
    def counts(values):
        bins = [0] * (len(boundaries) + 1)
        for value in values:
            bins[bisect_left(boundaries, value)] += 1
        epsilon = METHOD['pseudocount']
        return [(v + epsilon) / (len(values) + len(bins) * epsilon) for v in bins]
    a, b = counts(ref), counts(cur)
    psi = sum((x - y) * log(x / y) for x, y in zip(a, b))
    i = j = 0
    ks = 0
    for value in sorted(set(ref + cur)):
        while i < len(ref) and ref[i] <= value:
            i += 1
        while j < len(cur) and cur[j] <= value:
            j += 1
        ks = max(ks, abs(i / len(ref) - j / len(cur)))
    return {'state': 'DRIFT' if psi > METHOD['psi_threshold'] or ks > METHOD['ks_threshold'] else 'STABLE',
            'reference_count': len(ref), 'current_count': len(cur), 'psi': psi, 'ks_statistic': ks,
            'boundaries': boundaries, 'method': METHOD['drift'], 'p_value': None, 'uncertainty': None}


def _valid_return(view):
    value = view['prediction']['outputs']['return']
    if value is None:
        return False
    try:
        number(value)
        return True
    except (ValueError, TypeError, OverflowError):
        return False


def _binding(prediction):
    return {k: prediction[k] for k in ('model_id', 'model_contract_hash', 'artifact_hash', 'product', 'horizon_seconds', 'synthetic')}


def cohort(store, *, as_of, product=None, model_id=None, start=None, end=None, split=None, artifact_hash=None):
    at = timestamp(as_of)
    lower = timestamp(start) if start else None
    upper = timestamp(end) if end else None
    if lower and upper and lower >= upper:
        raise ValueError('monitoring interval must be nonempty and half-open')
    selected = []
    after = 0
    while True:
        items = store.records('prediction', as_of=at, after=after, limit=500)
        if not items:
            break
        for item in items:
            p = item['payload']
            if p['decision_at'] > at or (product and p['product'] != product) or (model_id and p['model_id'] != model_id):
                continue
            if (lower and p['decision_at'] < lower) or (upper and p['decision_at'] >= upper) or (artifact_hash and p['artifact_hash'] != artifact_hash):
                continue
            view = store.prediction_view(item['identity'], as_of=at)
            if split and (not view['inputs'] or view['inputs']['split'] != split):
                continue
            if len(selected) >= 10000:
                raise ValueError('monitoring budget exceeded; narrow start/end to at most 10000 predictions')
            selected.append(view)
        after = items[-1]['sequence']
        if len(items) < 500:
            break
    selected.sort(key=lambda v: (v['prediction']['decision_at'], v['prediction_hash']))
    bindings = {sha256_canonical(_binding(v['prediction'])) for v in selected}
    if len(bindings) > 1:
        raise ValueError('select a single model artifact, product and horizon for monitoring')
    return selected


def performance(views):
    pairs = [v for v in views if v['label_state'] == 'AVAILABLE' and _valid_return(v)
             and v['labels'][-1]['target'] == 'forward_return']
    if not pairs:
        return {'sample': 0, 'pending': sum(v['label_state'] == 'PENDING' for v in views),
                'method': 'observed-forward-return-errors-v1', 'model': None, 'baselines': {}, 'edge': {}, 'uncertainty': None}
    y = [number(v['labels'][-1]['value']) for v in pairs]
    pred = [number(v['prediction']['outputs']['return']) for v in pairs]
    def score(values, actual):
        errors = [a - b for a, b in zip(values, actual)]
        return {'sample': len(errors), 'mae': sum(map(abs, errors)) / len(errors),
                'mse': sum(e * e for e in errors) / len(errors),
                'rmse': sqrt(sum(e * e for e in errors) / len(errors))}
    model = score(pred, y)
    baselines, edges = {}, {}
    names = sorted({name for v in pairs for name in (v['inputs']['baselines'] if v['inputs'] else {})})
    for name in names:
        indices = [i for i, v in enumerate(pairs) if v['inputs'] and name in v['inputs']['baselines']]
        actual = [y[i] for i in indices]
        base = score([number(pairs[i]['inputs']['baselines'][name]) for i in indices], actual)
        measured = score([pred[i] for i in indices], actual)
        reduction = 1 - measured['mse'] / base['mse'] if base['mse'] > 0 else None
        baselines[name] = base
        edges[name] = {'sample': len(indices), 'mse_reduction': reduction, 'model_mse_on_same_pairs': measured['mse'],
            'population_hash': sha256_canonical([pairs[i]['prediction_hash'] for i in indices]),
            'method': METHOD['edge'], 'scope': 'EXPLORATORY', 'uncertainty': None,
            'state': 'INSUFFICIENT_SAMPLE' if len(indices) < METHOD['minimum_sample'] else
                     'UNDEFINED_ZERO_BASELINE_MSE' if reduction is None else 'DESCRIPTIVE_ONLY'}
    return {'sample': len(pairs), 'pending': sum(v['label_state'] == 'PENDING' for v in views),
            'method': 'observed-forward-return-errors-v1', 'model': model, 'baselines': baselines,
            'edge': edges, 'uncertainty': None, 'label_policy': 'latest causally available append-only correction'}


def make_reference(store, *, reference_id, as_of, **selection):
    views = cohort(store, as_of=as_of, **selection)
    if not views:
        raise ValueError('no observed predictions for reference')
    features = defaultdict(list)
    for view in views:
        for name, value in (view['inputs']['features'] if view['inputs'] else ()) or ():
            features[name].append(value)
    reference = MonitoringReference(reference_id=reference_id, version='monitoring-reference-v1',
        binding=_binding(views[0]['prediction']), as_of=as_of, selection=selection,
        population_hash=sha256_canonical([v['prediction_hash'] for v in views]),
        predictions=tuple(v['prediction']['outputs']['return'] for v in views if _valid_return(v)),
        features=dict(features), baseline_edges=performance(views)['edge'], method=METHOD,
        synthetic=views[0]['prediction']['synthetic'])
    identity = store.append('reference', reference, object_id=reference_id, recorded_at=as_of)
    return identity


def monitor(store, *, as_of, reference_hash=None, **selection):
    at = timestamp(as_of)
    views = cohort(store, as_of=at, **selection)
    features = defaultdict(list)
    regimes = defaultdict(list)
    periods = defaultdict(list)
    missing_features = missing_quality = quality_gaps = unknown_gaps = 0
    for view in views:
        inputs = view['inputs']
        values = inputs['features'] if inputs else None
        if values is None:
            missing_features += 1
        for name, value in values or ():
            features[name].append(value)
        quality = inputs['input_quality'] if inputs else {}
        missing_quality += quality.get('state') != 'VALID'
        if quality.get('gaps') is None:
            unknown_gaps += 1
        else:
            quality_gaps += int(quality['gaps'])
        regimes[regime(values)].append(view)
        periods[view['prediction']['decision_at'][:7]].append(view)
    predictions = [v['prediction']['outputs']['return'] for v in views if _valid_return(v)]
    current_performance = performance(views)
    reference, drifts = None, {}
    if reference_hash:
        raw = store.get(reference_hash, kind='reference')
        reference = MonitoringReference.from_dict(raw['payload'])
        if raw['recorded_at'] > at or reference.as_of > at:
            raise ValueError('reference not recorded at monitoring time')
        if views and dict(reference.binding) != _binding(views[0]['prediction']):
            raise ValueError('reference cohort mismatch')
        drifts['prediction'] = drift(reference.predictions, predictions)
        for name in sorted(set(reference.features) | set(features)):
            drifts['feature:' + name] = drift(reference.features.get(name, ()), features.get(name, ()))
    inference = []
    model = views[0]['prediction']['model_id'] if views else selection.get('model_id')
    product = views[0]['prediction']['product'] if views else selection.get('product')
    for item in store.records('inference', as_of=at):
        p = item['payload']
        if ((not model or p['model_id'] == model) and (not product or p['product'] == product)
                and (not views or (p['model_contract_hash'], p['artifact_hash'], p['horizon_seconds'], p['synthetic']) ==
                     (views[0]['prediction']['model_contract_hash'], views[0]['prediction']['artifact_hash'],
                      views[0]['prediction']['horizon_seconds'], views[0]['prediction']['synthetic']))
                and (not selection.get('artifact_hash') or p['artifact_hash'] == selection['artifact_hash'])
                and (not selection.get('split') or not views or
                     views[0]['prediction']['decision_at'] <= p['decision_at'] <= views[-1]['prediction']['decision_at'])
                and p['decision_at'] <= at
                and (not selection.get('start') or p['decision_at'] >= timestamp(selection['start']))
                and (not selection.get('end') or p['decision_at'] < timestamp(selection['end']))):
            inference.append(p)
    success = sum(i['status'] == 'SUCCESS' for i in inference)
    latency = distribution([i['latency_ms'] for i in inference if i['latency_ms'] is not None])
    errors = len(inference) - success
    freshness = (datetime.fromisoformat(at) - datetime.fromisoformat(views[-1]['prediction']['decision_at'])).total_seconds() if views else None
    # Expected cadence is explicitly hourly for these adapters; duplicate instants cannot inflate gaps.
    times = sorted({datetime.fromisoformat(v['prediction']['decision_at']) for v in views})
    cadence = 3600 if views and all(v['inputs'] and v['inputs']['provenance'].get('cadence_seconds') == 3600 for v in views) else None
    gaps = sum(max(0, int((b - a).total_seconds() / cadence) - 1) for a, b in zip(times, times[1:])) if cadence else None
    invalid_outputs = sum(v['prediction']['outputs']['return'] is not None and not _valid_return(v) for v in views)
    prediction_errors = sum(bool(v['prediction']['errors']) for v in views)
    classes = []
    if not views or missing_features or missing_quality or quality_gaps or gaps or (freshness is not None and freshness > METHOD['freshness_seconds']):
        classes.append({'category': 'MISSING_DATA', 'missing_feature_rows': missing_features,
            'input_quality_not_valid': missing_quality, 'input_gaps': None if unknown_gaps else quality_gaps, 'known_input_gaps': quality_gaps, 'input_gap_unknown_rows': unknown_gaps, 'decision_gaps': gaps,
            'freshness_seconds': freshness, 'method': 'observed-input-and-hourly-decision-gaps-v1'})
    if invalid_outputs or prediction_errors or (inference and errors / len(inference) > METHOD['error_rate']) or (latency['p95'] is not None and latency['p95'] > METHOD['latency_ms']):
        classes.append({'category': 'TECHNICAL_DEGRADATION', 'attempts': len(inference), 'errors': errors,
                        'p95_ms': latency['p95'], 'invalid_outputs': invalid_outputs, 'prediction_error_rows': prediction_errors, 'method': 'observed-inference-attempts-v1'})
    if any(value['state'] == 'DRIFT' for value in drifts.values()):
        classes.append({'category': 'DRIFT', 'reference_hash': reference_hash, 'method': METHOD['drift'],
                        'sample': len(views), 'signals': [k for k, v in drifts.items() if v['state'] == 'DRIFT']})
    if reference:
        for name, current in current_performance['edge'].items():
            old = reference.baseline_edges.get(name)
            if (old and old['state'] == current['state'] == 'DESCRIPTIVE_ONLY'
                    and old['mse_reduction'] is not None and current['mse_reduction'] is not None
                    and current['mse_reduction'] < old['mse_reduction'] - .05):
                classes.append({'category': 'PERFORMANCE_DROP', 'baseline': name, 'sample': current['sample'],
                    'reference_sample': old['sample'], 'current_reduction': current['mse_reduction'],
                    'reference_reduction': old['mse_reduction'], 'method': METHOD['edge'], 'uncertainty': None})
    return {'schema': 'model-monitoring-v1', 'as_of': at, 'selection': selection, 'sample': len(views),
        'population_hash': sha256_canonical([v['prediction_hash'] for v in views]),
        'binding': _binding(views[0]['prediction']) if views else None,
        'method': METHOD, 'method_hash': sha256_canonical(METHOD), 'regime_definition': REGIME,
        'regime_hash': sha256_canonical(REGIME), 'reference_hash': reference_hash,
        'freshness': {'seconds_since_last_decision': freshness, 'threshold_seconds': METHOD['freshness_seconds'],
                      'decision_gaps': gaps, 'cadence_seconds': cadence, 'input_gaps': None if unknown_gaps else quality_gaps, 'known_input_gaps': quality_gaps, 'input_gap_unknown_rows': unknown_gaps},
        'output_quality': {'invalid_returns': invalid_outputs, 'prediction_error_rows': prediction_errors},
        'input_quality': {'sample': len(views), 'missing_features': missing_features, 'not_valid': missing_quality},
        'inference': {'attempts': len(inference), 'errors': errors,
            'availability': success / len(inference) if inference else None, 'latency_ms': latency,
            'state': 'OBSERVED' if inference else 'NOT_OBSERVED'},
        'distributions': {'prediction': distribution(predictions), 'features': {k: distribution(v) for k, v in features.items()}},
        'drift': drifts, 'performance': current_performance,
        'performance_by_product': {product: current_performance} if product else {},
        'performance_by_period': {p: performance(v) for p, v in periods.items()},
        'performance_by_regime': {p: performance(v) for p, v in regimes.items()},
        'classification': classes, 'scope': 'EXPLORATORY', 'uncertainty': None,
        'limitations': ['pending labels excluded; labels never used to define regimes',
                        'no live freshness claim from an archived replay',
                        'unobserved inference latency and availability remain null',
                        'baseline advantage is descriptive and does not establish or refute an edge']}
