"""Synthetic sector/competitor news associations preserve observation time."""
from datetime import timedelta

from scripts.radar.analysis import cluster, link_anomalies
from tests.radar.test_radar import AT, story


def test_telecom_and_tower_anomalies_link_to_one_event_without_causal_claim():
    event = cluster([story(headline='Telecom stocks sell off as SpaceX expands Starlink',
                           symbols=['T', 'VZ', 'TMUS'])], at=AT)[0]
    rows = [{'symbol': symbol, 'move_atr': move, 'volume_vs_20d': 3,
             'closed_at': AT.isoformat(), 'received_at': AT.isoformat(), 'partial_day': False}
            for symbol, move in [('T', -4.8), ('VZ', -4.5), ('TMUS', -5.3), ('CCI', 6.2), ('AMT', 4.5), ('SBAC', 2.9), ('MRNA', 2.1)]]
    groups = link_anomalies(rows, [event], AT)
    telecom = next(g for g in groups if g['theme'] == 'telecom_satellites')
    assert {r['symbol'] for r in telecom['anomalies']} == {'T', 'VZ', 'TMUS', 'CCI', 'AMT', 'SBAC'}
    assert all(r['news_status'] == 'MATCHED' and r['news_links'][0]['event_id'] == event['id'] for r in telecom['anomalies'])
    assert next(r for g in groups for r in g['anomalies'] if r['symbol'] == 'MRNA')['news_status'] == 'NO_NEWS_FOUND'
    exposures = {r['symbol']: r for r in event['transmission_hypotheses']}
    assert exposures['T']['direction'] == 'loser'
    assert 'concurrence' in exposures['T']['mechanism']
    assert exposures['CCI']['direction'] == 'uncertain'  # rising towers do not prove a beneficiary


def test_news_after_cutoff_or_outside_anomaly_window_is_never_used():
    old = cluster([story(headline='Verizon earnings fall', symbols=['VZ'], published=AT-timedelta(days=5))])[0]
    future = cluster([story(slug='future', headline='Telecom stocks sell off', received=AT+timedelta(hours=1))])[0]
    rows = [{'symbol': 'VZ', 'move_atr': -4, 'closed_at': AT.isoformat(), 'received_at': AT.isoformat()}]
    groups = link_anomalies(rows, [old, future], AT)
    assert groups[0]['anomalies'][0]['news_status'] == 'NO_NEWS_FOUND'


def test_independent_and_primary_support_change_importance_without_hype_weight():
    single = cluster([story(publisher='wire')])[0]
    primary = cluster([story(publisher='issuer', primary=True)])[0]
    corroborated = cluster([story(publisher='wire'), story(slug='two', publisher='editor')])[0]
    syndicated = cluster([story(headline='Reuters: Micron raises memory guidance after earnings', publisher='yahoo'),
                          story(slug='two', headline='Reuters: Micron raises memory guidance after earnings', publisher='benzinga')])[0]
    assert primary['importance'] > corroborated['importance'] > single['importance']
    assert syndicated['importance_components']['source_support'] == 0
    retail = cluster([story(publisher='wire'), story(slug='retail', publisher='channel', retail=True)])[0]
    assert retail['importance'] == single['importance']


def test_tower_beneficiary_is_conditional_when_reporting_explicitly_names_it():
    row = story(headline='SpaceX could benefit cell-tower owners', symbols=['CCI', 'AMT', 'SBAC'])
    row['summary'] = 'Expansion could benefit American Tower, Crown Castle and SBA Communications.'
    event = cluster([row])[0]
    towers = [r for r in event['transmission_hypotheses'] if r['symbol'] in ('CCI', 'AMT', 'SBAC') and r['role'] == 'supplier']
    assert towers and all(r['direction'] == 'beneficiary' and r['mechanism'].startswith('Si ') for r in towers)


def test_unrelated_provider_basket_and_same_broad_sector_do_not_explain_an_anomaly():
    event = cluster([story(headline='Microsoft cloud subscription revenue rises', symbols=['MSFT', 'MRNA', 'DE'])])[0]
    rows = [{'symbol': 'MRNA', 'move_atr': 3, 'closed_at': AT.isoformat()}]
    assert link_anomalies(rows, [event], AT)[0]['anomalies'][0]['news_status'] == 'NO_NEWS_FOUND'


def test_business_map_keeps_carrier_suppliers_and_competitors_distinct():
    event = cluster([story(headline='AT&T raises wireless network spending')])[0]
    exposures = {(r['symbol'], r['role']) for r in event['transmission_hypotheses']}
    assert ('VZ', 'competitor') in exposures
    assert ('CCI', 'supplier') in exposures
    assert ('CCI', 'competitor') not in exposures


def test_explicit_headline_decline_wins_over_an_unrelated_summary_rally():
    row = story(headline='AT&T, Verizon and T-Mobile tumble on satellite competition')
    row['summary'] = 'Wireless carriers trade lower. Airline fares and Treasury yields are rising.'
    event = cluster([row])[0]
    carriers = [r for r in event['transmission_hypotheses'] if r['symbol'] in ('T', 'VZ', 'TMUS') and r['role'] == 'direct']
    assert len(carriers) == 3 and all(r['direction'] == 'loser' for r in carriers)
