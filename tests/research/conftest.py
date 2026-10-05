from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import localcontext

import pytest

from scripts.trading_lab.platform.contracts import PredictionRecord, LabelRecord, OUTPUTS
from scripts.trading_lab.platform.datasets import synthetic_dataset
from scripts.trading_lab.research.contracts import PredictionEvidence
from scripts.trading_lab.research.store import ResearchStore
from scripts.trading_lab.sources.canonical import sha256_canonical

AT = '2026-06-01T00:00:00+00:00'
RECORDED = '2026-06-04T00:00:00+00:00'
H = 'a' * 64


@pytest.fixture
def store(tmp_path):
    return ResearchStore(tmp_path / 'research')


@pytest.fixture(scope='session')
def dataset():
    with localcontext() as ctx:
        ctx.prec = 34
        return synthetic_dataset(products=['BTC-USD'], bars=120)


def issue(store, *, identifier='synthetic-p', decision=AT, recorded=AT, value='.01', features=None,
          model='synthetic-test', product='BTC-USD', split='test', quality='VALID', artifact=H):
    features = features or [['return_4', '.001'], ['atr_pct_14', '.01']]
    snapshot = {'schema': 'synthetic-test-input-v1', 'synthetic': True, 'as_of': decision,
                'features': features, 'event_ids': []}
    p = PredictionRecord(prediction_id=identifier, model_id=model, model_contract_hash=H,
        artifact_hash=artifact, product=product, decision_at=decision, horizon_seconds=14400,
        snapshot_hash=sha256_canonical(snapshot), features_hash=sha256_canonical(features), event_ids=(),
        outputs={k: value if k == 'return' else None for k in OUTPUTS}, synthetic=True)
    e = PredictionEvidence(prediction_id=identifier, prediction_hash=p.identity, recorded_at=recorded,
        features=features, snapshot=snapshot, baselines={'ZERO': '0'}, input_quality={'state': quality, 'gaps': 0},
        split=split, provenance={'synthetic': True, 'method': 'synthetic-test-v1', 'cadence_seconds': 3600})
    store.issue(p, e)
    return p, e


def label(p, *, realized=None, available=None, recorded=None, value='.02', version='1'):
    due = (datetime.fromisoformat(p.decision_at) + timedelta(seconds=p.horizon_seconds)).isoformat()
    return LabelRecord(label_id='synthetic-label-' + p.prediction_id, prediction_id=p.prediction_id,
        prediction_hash=p.identity, product=p.product, horizon_seconds=p.horizon_seconds,
        realized_at=realized or due, available_at=available or due, recorded_at=recorded or available or due,
        target='forward_return', value=value, version=version,
        provenance={'synthetic': True, 'method': 'synthetic-after-horizon-label-v1'})
