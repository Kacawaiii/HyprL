"""Synthetic admission examples; no live providers or private source fixtures."""
from copy import deepcopy

import pytest

from scripts.trading_lab.agent_v2.inputs import validate_snapshot as frozen_snapshot
from scripts.trading_lab.agent_v2_quality.contracts import Blocked, instant
from scripts.trading_lab.agent_v2_quality.inputs import validate_snapshot
from scripts.trading_lab.agent_v2_quality.registry import SIC_SECTORS, admissible_registry, select_scope, sic_sector
from scripts.trading_lab.agent_v2_quality.synthetic import bundle


@pytest.mark.parametrize(('sic', 'sector'), [
    ('1311', 'energy'), ('2911', 'energy'), ('3571', 'technology'),
    ('7372', 'technology'), ('6021', 'financials'), ('2834', 'health'),
    ('3841', 'health'), ('5047', 'health'), ('5122', 'health'),
    ('6798', 'real_estate'), ('6512', 'real_estate'), ('4911', 'utilities'),
    ('3711', 'industrials'), ('5411', 'consumer'), ('0100', 'consumer'),
    (None, None), ('', None), ('0000', None), ('9999', None),
    (7372, None), (True, None), ('7372.0', None), (' 7372', None),
])
def test_fixed_sic_risk_sectors(sic, sector):
    assert sic_sector(sic) == sector


def test_sector_ranges_are_ordered_and_disjoint():
    assert all(low <= high for low, high, _ in SIC_SECTORS)
    assert all(left[1] < right[0] for left, right in zip(SIC_SECTORS, SIC_SECTORS[1:]))


def test_selection_is_order_independent_top_60_then_hunter_and_total_200():
    ranked = [{'symbol': f'M{i:03}', 'score': 1} for i in reversed(range(61))]
    hunter = [f'H{i:03}' for i in reversed(range(150))] + ['M000']
    selected, review = select_scope(ranked, hunter)
    assert selected == [f'M{i:03}' for i in range(60)] + [f'H{i:03}' for i in range(139)]
    assert review['excluded_cap'] == 11
    assert review['union'] == 210
    assert select_scope(ranked[::-1], hunter[::-1]) == (selected, review)


def test_unknown_dates_and_sic_are_excluded_with_all_reasons_and_no_backfill():
    ranked = [{'symbol': 'DATED', 'score': 3}, {'symbol': 'UNKNOWN', 'score': 2}]
    meta = {s: {'cik': '0000000001', 'market_cap': 1e9} for s in ['DATED', 'UNKNOWN', 'NOSIC']}
    evidence = {'date': '2026-10-29', 'source_url': 'https://example.invalid/earnings', 'source_hash': 'a' * 64}
    sec = {'DATED': {'cik': '0000000001', 'sic': '7372'}}
    rows, review = admissible_registry(ranked, ['NOSIC', 'UNKNOWN'], meta,
                                      {'DATED': [evidence], 'NOSIC': [evidence]}, sec)
    assert [r['symbol'] for r in rows] == ['DATED']
    assert rows[0]['cluster'] == 'technology'
    assert rows[0]['momentum_member'] is True
    assert review['excluded_by_primary_reason'] == {'EARNINGS_DATE_UNVERIFIED': 1, 'SIC_MISSING': 1}
    assert review['excluded_by_any_reason'] == {'EARNINGS_DATE_UNVERIFIED': 1, 'SIC_MISSING': 2}
    assert review['selected_equities'] == 3


def test_wrong_submissions_identity_never_assigns_a_sector():
    rows, review = admissible_registry([], ['ACME'], {'ACME': {'cik': '0000000001', 'market_cap': 1e9}},
        {'ACME': [{'date': '2026-10-29', 'source_url': 'https://example.invalid/', 'source_hash': 'b' * 64}]},
        {'ACME': {'cik': '0000000002', 'sic': '7372'}})
    assert rows == []
    assert review['excluded_by_primary_reason'] == {'SIC_IDENTITY_MISMATCH': 1}


def test_regime_etf_can_have_no_corporate_cap_but_equities_still_require_it():
    at = instant('2026-10-23T12:45:00Z')
    snapshot = bundle(at)['snapshot']
    spy = next(r for r in snapshot['instruments'] if r['symbol'] == 'SPY')
    spy['market_cap'] = None
    validate_snapshot(snapshot, at, 'SYNTHETIC')
    bad = deepcopy(snapshot)
    next(r for r in bad['instruments'] if r['symbol'] == 'ACME')['market_cap'] = None
    with pytest.raises(Blocked, match='MARKET_CAP_REQUIRED'):
        validate_snapshot(bad, at, 'SYNTHETIC')
    legacy = deepcopy(snapshot)
    legacy['schema'] = 'agent-v2-snapshot-v1'
    with pytest.raises(Blocked, match='INVALID_SCHEMA'):
        frozen_snapshot(legacy, at, 'SYNTHETIC')
    with pytest.raises(Blocked, match='INVALID_SCHEMA'):
        validate_snapshot(legacy, at, 'SYNTHETIC')


def test_revision_two_journal_is_rejected_without_changing_its_bytes(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts.trading_lab.agent_v2_quality import ledger
    from scripts.trading_lab.agent_v2_quality.spec import ARTIFACTS
    old_spec = json.loads((ARTIFACTS / 'trader_shadow_v2_spec_v2.json').read_text())
    old_config = json.loads((ARTIFACTS / 'trader_shadow_v2_preregistration_v2.json').read_text())
    old_hash = old_config.pop('canonical_sha256')
    root = tmp_path / 'revision-two'
    with monkeypatch.context() as patch:
        patch.setattr(ledger, 'SPEC_HASH', old_spec['canonical_sha256'])
        patch.setattr(ledger, 'PREREG_HASH', old_hash)
        patch.setattr(ledger, 'configuration', lambda: old_config)
        journal = ledger.Journal.initialize(root, 'SYNTHETIC', 'd' * 40,
                                            instant('2026-10-23T12:00:00Z'))
        journal.close()
    before = hashlib.sha256((root / 'shadow.sqlite').read_bytes()).hexdigest()
    with pytest.raises(Blocked, match='FROZEN_CONFIG_MISMATCH'):
        ledger.Journal(root, read_only=True)
    assert hashlib.sha256((root / 'shadow.sqlite').read_bytes()).hexdigest() == before


def test_metadata_exclusions_never_promote_the_61st_momentum_name():
    ranked = [{'symbol': f'M{i:03}', 'score': 100 - i} for i in range(61)]
    rows, review = admissible_registry(ranked, [], {}, {}, {})
    assert rows == []
    assert review['selected_equities'] == 60
    assert 'M060' not in {r['symbol'] for r in review['excluded']}
