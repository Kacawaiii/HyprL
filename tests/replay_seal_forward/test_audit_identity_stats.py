import json

import pytest

from scripts.trading_lab.replay_bench import marketdata as md, snapshot as sn
from scripts.trading_lab.replay_seal_forward import anon, experiment as ex, run, spec, stats


def test_anonymized_control_masks_symbols_names_dates_benchmarks_and_prompt():
    assets = ['AAPL', 'META', 'NVDA', 'COST', 'CAT', 'LINK-USD']
    title = ('AAPL Tim Cook iPhone, META Mark Zuckerberg Llama, NVDA Jensen Huang Blackwell; '
             'COST CAT LINK and cost cat link CONCAT COSTLY AAPL_extra on 2026-10-16 and October 16, 2026')
    price = {'state': 'AVAILABLE', 'recent_closes': [100, 101], 'returns': {'1': .01},
             'last_price_at': '2026-10-15T20:00:00Z'}
    context = {'schema': 'replay-context-v1', 'provenance': spec.PROVENANCE, 'session': '2026-10-16',
               'decision_time': '2026-10-16T12:00:00Z', 'price_cutoff': '2026-10-16T12:00:00Z',
               'universe': assets, 'prices': {s: dict(price, asset=s) for s in assets + list(sn.BENCHMARKS)},
               'headlines': {'AAPL': {'state': 'AVAILABLE', 'items': [dict(news_id=998877, title=title,
                   symbols=['AAPL', 'META', 'NVDA'], source='identifying_source', url='https://example.invalid/AAPL',
                   published_at='2026-10-15T18:00:00Z')]}}, 'exclusions': [], 'limitations': ['replay']}
    result, mapping = anon.anonymize({'context': context}, 17)
    assert anon.anonymize({'context': context}, 17) == (result, mapping)
    item = next(iter(result['headlines'].values()))['items'][0]
    assert all(s.startswith('STOCK_') for s in item['symbols'])
    for name in ('Tim Cook', 'iPhone', 'Mark Zuckerberg', 'Llama', 'Jensen Huang', 'Blackwell'):
        assert name not in item['title']
    assert 'cost cat link CONCAT COSTLY AAPL_extra' in item['title']
    assert item['news_id'] == 1 and item['source'] == 'newswire' and item['url'] == 'https://news.invalid/1'
    assert result['session'] == 'DAY+0' and item['published_at'] == 'DAY-1 18:00:00 UTC'
    assert '2026' not in json.dumps(result)
    for s in sn.BENCHMARKS:
        assert result['prices'][s]['recent_closes'] != price['recent_closes']
        assert result['prices'][s]['returns'] == price['returns']
        assert result['prices'][s]['recent_closes'][1]/result['prices'][s]['recent_closes'][0] == pytest.approx(1.01)
    assert 'replay' not in run.prompt('analyst', result).lower()


def rows(probability, n=20):
    days = md.sessions('2026-07-01', '2026-08-31')[:n]
    return [dict(panel=stats.PRIMARY['panel'], session=s.session_date, asset='SYNTH', horizon='1d',
                 p=probability, outperform=True) for s in days]


def test_primary_bootstrap_per_model_and_paired_difference_and_no_winner_without_CI():
    result = stats.primary_endpoint(rows(.9), rows(.5), .5)
    assert result['brier_skill'] == pytest.approx({'opus': .96, 'gpt': 0})
    assert result['ci95']['difference'][0] > 0 and result['verdict'] == 'OPUS_BETTER'
    assert result['ci95']['opus'][0] == pytest.approx(.96)
    assert stats.primary_endpoint(rows(.9), rows(.5), .5) == result
    equal = stats.primary_endpoint(rows(.5), rows(.5), .5)
    assert equal['ci95']['difference'] == [0, 0] and equal['verdict'] == 'NO_WINNER'
    short = stats.primary_endpoint(rows(.9, 9), rows(.5, 9), .5)
    assert short['ci95']['difference'] is None and short['verdict'] == 'NO_WINNER'
    assert 'No winner' in stats.wording(short)
    assert stats.primary_endpoint(rows(.9), rows(.5), None)['state'] == 'BASE_RATE_UNAVAILABLE'


def test_primary_per_model_keeps_unpaired_rows_and_refuses_different_targets():
    result = stats.primary_endpoint(rows(.9), rows(.5, 10), .5)
    assert result['per_model_sessions'] == {'opus': 20, 'gpt': 10}
    assert result['unpaired_dropped'] == {'opus': 10, 'gpt': 0}
    other = rows(.5)
    other[0]['outperform'] = False
    with pytest.raises(Exception, match='PRIMARY_TARGET_MISMATCH'):
        stats.primary_endpoint(rows(.9), other, .5)


def test_replay_call_order_uses_day_ordinal_for_all_groups_and_no_test_anonymization():
    days = ex.planned('dev', 'ab')
    assert ex.call_order(days[0]) == ('opus', 'gpt')
    assert ex.call_order(days[1]) == ('gpt', 'opus')
    assert ex.call_order(days[4]) == ('opus', 'gpt')
    assert spec.spec()['statistics']['primary_endpoint'] == stats.PRIMARY
    assert spec.spec()['evidence']['replay'].startswith('EXPLORATORY')
    with pytest.raises(Exception, match='PERIOD_OR_GROUP_NOT_ALLOWED'):
        ex.planned('test', 'anon')
