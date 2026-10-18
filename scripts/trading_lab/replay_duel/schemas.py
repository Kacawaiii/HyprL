"""Identical strict contracts for both subscriptions; forecast and trade are separate."""
from copy import deepcopy

from jsonschema import Draft202012Validator, FormatChecker
from scripts.trading_lab.trader_agent.schemas import ANALYST, for_openai_strict, obj, array, STRING
from scripts.trading_lab.replay_bench.run import C_VIEW
from scripts.trading_lab.replay_bench.grant import ReplayError, instant

P = {'type': 'number', 'minimum': 0, 'maximum': 1}
ANALYST_SCHEMA = for_openai_strict(deepcopy(ANALYST))
ANALYST_SCHEMA['properties']['views']['items']['properties']['p_outperform'] = P
REVIEWER_SCHEMA = obj({'verdicts': array(obj({
    'asset': STRING, 'horizon': {'enum': ['1d', '5d']},
    'verdict': {'enum': ['KEEP', 'DOWNGRADE', 'REJECT']},
    'p_outperform': P, 'note': STRING,
}), 100)})
NEWS_VIEW = for_openai_strict(deepcopy(C_VIEW))
NEWS_VIEW['properties']['p_outperform'] = P
FORECAST = obj({'asset': STRING, 'horizon': {'enum': ['2d', '10d']}, 'p_outperform': P})
NEWS_SCHEMA = obj({'views': array(NEWS_VIEW, 40), 'forecasts': array(FORECAST, 80)})
SCHEMAS = {'analyst': ANALYST_SCHEMA, 'reviewer': REVIEWER_SCHEMA, 'news': NEWS_SCHEMA}


def validate(schema, output):
    if not Draft202012Validator(schema, format_checker=FormatChecker()).is_valid(output):
        raise ReplayError('MODEL_SCHEMA_INVALID')


def coverage(keys, wanted):
    if len(keys) != len(set(keys)) or set(keys) != set(wanted):
        raise ReplayError('FORECAST_COVERAGE_INVALID')


def direction(view):
    p = view['p_outperform']
    if view['view'] == 'UP' and p <= .5 or view['view'] == 'DOWN' and p >= .5:
        raise ReplayError('DIRECTION_INVALID')


def analyst(output, context):
    validate(ANALYST_SCHEMA, output)
    coverage([(v['asset'], v['horizon']) for v in output['views']],
             [(a, h) for a in context['universe'] for h in ('1d', '5d')])
    known = {(instant(n['published_at']), n['url']) for g in context['headlines'].values() for n in g['items']}
    for v in output['views']:
        direction(v)
        if v['view'] != 'ABSTAIN' and not v['catalysts']:
            raise ReplayError('UNSUPPORTED_VIEW')
        if any((instant(n['published_at']), n['url']) not in known for n in v['catalysts']):
            raise ReplayError('CITATION_NOT_IN_SNAPSHOT')


def reviewer(output, original):
    validate(REVIEWER_SCHEMA, output)
    before = {(v['asset'], v['horizon']): v for v in original['views']}
    coverage([(v['asset'], v['horizon']) for v in output['verdicts']], before)
    for v in output['verdicts']:
        p = before[(v['asset'], v['horizon'])]['p_outperform']
        if v['verdict'] == 'KEEP' and v['p_outperform'] != p:
            raise ReplayError('REVIEW_CHANGED_KEEP')
        if v['verdict'] == 'DOWNGRADE' and not min(p, .5) <= v['p_outperform'] <= max(p, .5):
            raise ReplayError('REVIEW_RAISED_PROBABILITY')
        # A rejected trade retains a real forecast, constrained to a conservative revision.
        if v['verdict'] == 'REJECT' and not min(p, .5) <= v['p_outperform'] <= max(p, .5):
            raise ReplayError('REVIEW_RAISED_PROBABILITY')


def news(output, context):
    validate(NEWS_SCHEMA, output)
    candidates = {c['symbol']: {h['news_id'] for h in c['headlines']} for c in context['candidates']}
    coverage([v['symbol'] for v in output['views']], candidates)
    coverage([(v['asset'], v['horizon']) for v in output['forecasts']],
             [(a, h) for a in candidates for h in ('2d', '10d')])
    for v in output['views']:
        direction(v)
        if not set(v['news_ids']) <= candidates[v['symbol']]:
            raise ReplayError('CITATION_NOT_IN_SNAPSHOT')
        if v['view'] != 'ABSTAIN' and not v['news_ids']:
            raise ReplayError('UNSUPPORTED_VIEW')
