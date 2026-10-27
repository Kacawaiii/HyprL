"""Forward archive gaps retain usable evidence without hiding an unusable decision window."""
from datetime import timedelta

import pytest

from scripts.trading_lab.replay_bench.grant import ReplayError, iso
from scripts.trading_lab.replay_seal_forward import inputs
from tests.replay_seal_forward.test_inputs import fixture, T


def windows(*intervals):
    return [({'start': iso(a), 'end': iso(b)}, {'next_page_token': None}) for a, b in intervals]


def test_prebuild_lookback_gap_keeps_both_forward_contexts(tmp_path):
    archive, _ = fixture(tmp_path)
    archive.news_windows = windows((T - timedelta(hours=72) + timedelta(minutes=20, seconds=11), T))
    bundles = inputs.build(archive, T, 'a' * 64, live=True)
    for bundle in bundles.values():
        assert bundle['manifest']['data_quality']['excluded_day'] is False
        assert bundle['manifest']['news_archive_coverage']['state'] == 'PARTIAL'
        assert bundle['manifest']['data_quality']['counts_by_reason']['NEWS_ARCHIVE_COVERAGE_MISSING'] == 1
    assert bundles['ab']['context']['universe']
    assert [c['symbol'] for c in bundles['cd']['context']['candidates']] == ['SMLL']
    # The historical replay's strict exclusion policy is unchanged.
    assert inputs.build(archive, T, 'a' * 64)['ab']['manifest']['data_quality']['excluded_day']


@pytest.mark.parametrize('minutes,excluded', [(5, False), (30, False), (31, True), (72 * 60, True)])
def test_recent_gap_boundary_and_missing_decision_window(tmp_path, minutes, excluded):
    archive, _ = fixture(tmp_path)
    end = T - timedelta(minutes=minutes)
    archive.news_windows = windows((T - timedelta(hours=72), end)) if end > T - timedelta(hours=72) else []
    bundles = inputs.build(archive, T, 'a' * 64, live=True)
    assert all(b['manifest']['data_quality']['excluded_day'] is excluded for b in bundles.values())
    assert bool(bundles['ab']['context']['universe']) is not excluded


def test_short_internal_gap_excludes_only_news_in_that_interval(tmp_path):
    archive, _ = fixture(tmp_path)
    left, right = T - timedelta(hours=2), T - timedelta(hours=1, minutes=50)
    archive.news_windows = windows((T - timedelta(hours=72), left), (right, T))
    archive.news[2] = dict(archive.news[1], id=2, headline='uncovered item',
                           created_at=iso(left), updated_at=iso(left))
    bundles = inputs.build(archive, T, 'a' * 64, live=True)
    assert bundles['ab']['context']['universe']
    assert [h['news_id'] for h in bundles['cd']['context']['candidates'][0]['headlines']] == [1]
    assert any(r['scope'] == 'news' and r['identity'] == '2' and r['reason'] == 'NEWS_ARCHIVE_COVERAGE_MISSING'
               for r in bundles['ab']['manifest']['data_quality']['records'])


def test_only_complete_pagination_chains_can_fill_a_gap(tmp_path):
    archive, _ = fixture(tmp_path)
    start = T - timedelta(hours=72)
    middle = T - timedelta(minutes=10)
    archive.news_windows = windows((start, middle), (middle, T))
    assert archive.news_gaps(start, T) == []
    with pytest.raises(ReplayError, match='NEWS_ARCHIVE_COVERAGE_MISSING'):
        archive.require_news(start, T)  # The historical single-query contract stays strict.
    archive.news_windows[-1][1]['next_page_token'] = 'missing-page'
    assert archive.news_gaps(start, T) == [(middle, T)]


def test_even_a_short_gap_blocks_when_it_covers_the_entire_decision_window(tmp_path):
    at = T.replace(hour=20, minute=10)
    archive, _ = fixture(tmp_path, at)
    archive.news_windows = windows((at - timedelta(hours=72), at - timedelta(minutes=10)))
    bundles = inputs.build(archive, at, 'a' * 64, live=True)
    assert all(b['manifest']['data_quality']['excluded_day'] for b in bundles.values())
