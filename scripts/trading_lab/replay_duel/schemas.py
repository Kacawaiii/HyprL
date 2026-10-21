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


VIEWS_FILLED = 'abstain_views_filled'


def abstain_view(symbol):
    """A candidate without a model view: no opinion, no citation, no trade."""
    # holding_sessions 2 is unused for ABSTAIN and already inside the mandatory 2d forecast label plan.
    return {'symbol': symbol, 'view': 'ABSTAIN', 'p_outperform': .5, 'holding_sessions': 2, 'event_code': 'none',
            'news_ids': [], 'reason': 'NO_VIEW_RETURNED', 'counter_thesis': '', 'falsifier': ''}


def news(output, context):
    """Validate a news answer, then normalize it in place and return it.

    Operator decision, uniform for both models in replay and forward and taken before any outcome
    was scored: a candidate symbol without a view is an ABSTAIN view (p_outperform 0.5, no news_ids,
    no trade). Duplicate views and views for unknown symbols stay invalid; forecasts must still cover
    every candidate x {2d, 10d}. The model-facing schema and prompts are unchanged. Validation runs
    entirely before the fill, and the normalized output records how many views were filled.
    """
    validate(NEWS_SCHEMA, output)
    candidates = {c['symbol']: {h['news_id'] for h in c['headlines']} for c in context['candidates']}
    symbols = [v['symbol'] for v in output['views']]
    if len(symbols) != len(set(symbols)) or not set(symbols) <= set(candidates):
        raise ReplayError('FORECAST_COVERAGE_INVALID')
    coverage([(v['asset'], v['horizon']) for v in output['forecasts']],
             [(a, h) for a in candidates for h in ('2d', '10d')])
    for v in output['views']:
        direction(v)
        if not set(v['news_ids']) <= candidates[v['symbol']]:
            raise ReplayError('CITATION_NOT_IN_SNAPSHOT')
        if v['view'] != 'ABSTAIN' and not v['news_ids']:
            raise ReplayError('UNSUPPORTED_VIEW')
    missing = [s for s in candidates if s not in set(symbols)]
    output['views'].extend(abstain_view(s) for s in missing)
    output[VIEWS_FILLED] = len(missing)
    return output
