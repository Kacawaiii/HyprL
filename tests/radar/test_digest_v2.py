"""Synthetic quantities at the digest boundary; no provider requests."""
import json

import pytest

from scripts.radar.analysis import cluster
from scripts.radar.core import RadarError
from scripts.radar.digest import summarize, validate
from tests.radar.test_radar import scenario, state, story


def test_captured_numbers_round_in_french_and_conditional_text_is_allowed():
    event = cluster([story(headline='Micron revenue rises 7.26% to $8.12 billion')])[0]
    output = {'events': [scenario(event)]}
    output['events'][0]['summary'] = 'Micron: revenus en hausse de 7,3%, à 8,12 milliards de dollars.'
    output['events'][0]['impact'] = 'Si les volumes doublent, les fournisseurs pourraient bénéficier de la demande.'
    assert validate(output, [event]) == output


@pytest.mark.parametrize('claim', ['Micron: hausse de 8,1%.', 'Micron: revenus de 7,26 milliards de dollars.'])
def test_number_cannot_borrow_another_unit(claim):
    event = cluster([story(headline='Micron revenue rises 7.26% to $8.12 billion')])[0]
    output = {'events': [scenario(event)]}
    output['events'][0]['summary'] = claim
    with pytest.raises(RadarError, match='LLM_NUMBER_UNSUPPORTED'):
        validate(output, [event])


def test_guard_retry_gets_error_and_number_ledger_within_budget(state):
    store, grant, _, _ = state
    event = cluster([story(headline='Micron revenue rises 7.26%')])[0]
    prompts = []
    def runner(prompt, root):
        prompts.append(prompt)
        output = {'events': [scenario(event)]}
        output['events'][0]['summary'] = 'Micron: hausse de 99%.' if len(prompts) == 1 else 'Micron: hausse de 7,3%.'
        return output
    evidence = summarize(store, grant, [event], {}, runner=runner)
    assert evidence['calls'] == 2 and store.counts()['llm'] == 2
    assert 'LLM_NUMBER_UNSUPPORTED' in prompts[1]
    assert 'allowed_numbers' in json.loads(prompts[0].split('\n', 1)[1])


def test_observed_return_cannot_be_used_for_a_different_asset_or_horizon():
    event = cluster([story()])[0]
    panel = {'SPY': {'symbol': 'SPY', 'last': 778.6, 'returns_pct': {'1d': 0.194, '5d': 1.16}}}
    output = {'events': [scenario(event)]}
    output['events'][0]['priced_in'] = 'SPY: +0,19% sur 1d; causalité non établie.'
    assert validate(output, [event], panel) == output
    for claim in ['QQQ: +0,19%; causalité non établie.', 'SPY: -0,19%; causalité non établie.',
                  'SPY: +0,19% sur 5d; causalité non établie.']:
        output['events'][0]['priced_in'] = claim
        with pytest.raises(RadarError, match='LLM_NUMBER_UNSUPPORTED'):
            validate(output, [event], panel)


def test_retry_never_exceeds_remaining_daily_budget(state):
    store, grant, _, _ = state
    store.reserve(grant, 'llm', 'sonnet', 2)
    event = cluster([story()])[0]
    calls = []
    def runner(*args):
        calls.append(1)
        output = {'events': [scenario(event)]}
        output['events'][0]['summary'] = 'Micron +99%.'
        return output
    with pytest.raises(RadarError, match='DAILY_BUDGET'):
        summarize(store, grant, [event], {}, runner=runner)
    assert len(calls) == 1 and store.counts()['llm'] == 2


def test_spelled_financial_numbers_and_wrong_issuer_are_not_a_guard_bypass():
    event = cluster([story(headline='Micron rises 7.26%; Nvidia falls 8.12%')])[0]
    output = {'events': [scenario(event)]}
    for claim in ['Micron gagne trois pour cent.', 'Nvidia gagne 7,3%.']:
        output['events'][0]['summary'] = claim
        with pytest.raises(RadarError, match='LLM_NUMBER_UNSUPPORTED'):
            validate(output, [event])


def test_long_decimals_are_checked_and_signed_declines_can_use_french_words():
    event = cluster([story(headline='Micron falls 7.26%')])[0]
    output = {'events': [scenario(event)]}
    output['events'][0]['summary'] = 'Micron: baisse de 7,3%.'
    assert validate(output, [event]) == output
    output['events'][0]['summary'] = 'Micron: baisse de 7,2619%.'
    with pytest.raises(RadarError, match='LLM_NUMBER_UNSUPPORTED'):
        validate(output, [event])


def test_one_clause_cannot_borrow_another_clauses_asset_or_horizon():
    event = cluster([story()])[0]
    panel = {'SPY': {'symbol': 'SPY', 'last': 778.6, 'returns_pct': {'1d': 0.194, '5d': 1.16}}}
    output = {'events': [scenario(event)]}
    for claim in ['SPY: +0,19% sur 5d; SPY: +1,16% sur 1d.', 'QQQ: +0,19%; SPY: +1,16%.']:
        output['events'][0]['priced_in'] = claim
        with pytest.raises(RadarError, match='LLM_NUMBER_UNSUPPORTED'):
            validate(output, [event], panel)
