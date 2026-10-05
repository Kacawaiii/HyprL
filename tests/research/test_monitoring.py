from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from scripts.trading_lab.research.demo import synthetic_scenarios
from scripts.trading_lab.research.monitoring import (monitor, make_reference, drift, regime, distribution, METHOD, MonitoringReference)
from tests.research.conftest import issue, label, AT, RECORDED


def test_injected_synthetic_scenarios_distinguish_all_four_categories(store):
    result = synthetic_scenarios(store, at=RECORDED)
    assert result['synthetic'] and result['sample'] == 23
    assert result['classifications'] == ['DRIFT', 'MISSING_DATA', 'PERFORMANCE_DROP', 'TECHNICAL_DEGRADATION']
    view = monitor(store, as_of=RECORDED, product='SYNTHETIC-DEMO', model_id='injected-synthetic-monitor-v1',
        split='current', reference_hash=result['reference_hash'])
    assert view['freshness']['decision_gaps'] == 1
    assert view['inference']['availability'] < 1 and view['inference']['latency_ms']['p95'] == 1500
    assert view['performance']['sample'] == 23
    assert view['performance']['edge']['ZERO']['method'] == METHOD['edge']
    assert view['performance']['edge']['ZERO']['sample'] == 23
    assert view['performance']['edge']['ZERO']['uncertainty'] is None
    assert view['performance_by_regime']['HIGH_VOL']['sample'] == 23
    assert view['scope'] == 'EXPLORATORY'


def test_pending_labels_never_enter_performance_or_regime(store):
    p, _ = issue(store)
    store.append_label(label(p))
    before = monitor(store, as_of=AT, product=p.product)
    assert before['performance']['sample'] == 0 and before['performance']['pending'] == 1
    assert before['performance_by_regime']['RANGE']['sample'] == 0
    assert before['inference']['availability'] is None and before['inference']['latency_ms']['count'] == 0
    after = monitor(store, as_of=RECORDED, product=p.product)
    assert after['performance']['sample'] == 1 and after['performance']['edge']['ZERO']['state'] == 'INSUFFICIENT_SAMPLE'
    assert after['performance_by_product'][p.product]['sample'] == 1


def test_reference_version_cohort_and_as_of_binding(store):
    p, _ = issue(store, split='validation')
    ref = make_reference(store, reference_id='synthetic-ref-v1', as_of=AT, product=p.product, split='validation')
    record = MonitoringReference.from_dict(store.get(ref)['payload'])
    assert record.identity == ref and record.method['version'] == 'monitoring-method-v1'
    with pytest.raises(ValueError):
        replace(record, method={**METHOD, 'psi_threshold': 10})
    with pytest.raises(ValueError):
        monitor(store, as_of='2026-05-01T00:00:00Z', product=p.product, reference_hash=ref)
    issue(store, identifier='other-model', model='synthetic-other', recorded=AT)
    with pytest.raises(ValueError, match='single'):
        monitor(store, as_of=AT, product=p.product)
    with pytest.raises(ValueError, match='mismatch'):
        monitor(store, as_of=AT, product=p.product, model_id='synthetic-other', reference_hash=ref)


def test_constant_distributions_ks_psi_and_sample_limits():
    stable = drift([1] * 24, [1] * 24)
    assert stable['state'] == 'STABLE' and stable['psi'] == 0 and stable['ks_statistic'] == 0
    shifted = drift([1] * 24, [2] * 24)
    assert shifted['state'] == 'DRIFT' and shifted['ks_statistic'] == 1 and shifted['psi'] > 0
    assert shifted['p_value'] is None
    assert drift([1], [2] * 24)['state'] == 'INSUFFICIENT_SAMPLE'
    assert distribution([])['mean'] is None
    assert distribution([1, 2, 3])['mean'] == 2
    with pytest.raises(ValueError):
        distribution(['NaN'])


@pytest.mark.parametrize('features,expected', [([], 'UNKNOWN'), ([['return_4', '.02'], ['atr_pct_14', '.01']], 'UP'),
    ([['return_4', '-.02'], ['atr_pct_14', '.01']], 'DOWN'),
    ([['return_4', '0'], ['atr_pct_14', '.04']], 'HIGH_VOL')])
def test_regimes_use_explicit_decision_features(features, expected):
    assert regime(features) == expected


def test_invalid_raw_output_is_kept_and_classified_without_poisoning_metrics(store):
    p, e = issue(store, value='NaN')
    store.append_label(label(p))
    result = monitor(store, as_of=RECORDED, product=p.product)
    assert store.get(p.identity)['payload']['outputs']['return'] == 'NaN'
    assert result['performance']['sample'] == 0
    assert result['distributions']['prediction']['count'] == 0
    assert result['output_quality']['invalid_returns'] == 1
    assert any(c['category'] == 'TECHNICAL_DEGRADATION' for c in result['classification'])
