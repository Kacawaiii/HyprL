"""Shared duel contracts, plus the coded clock used only by the anonymous control."""
from copy import deepcopy

from scripts.trading_lab.replay_duel.schemas import *  # noqa: F403
from scripts.trading_lab.replay_duel import schemas as shared

ANON_ANALYST_SCHEMA = deepcopy(ANALYST_SCHEMA)
ANON_ANALYST_SCHEMA['properties']['views']['items']['properties']['catalysts']['items']['properties']['published_at'] = {
    'type': 'string', 'pattern': r'^DAY[+-]\d+ (?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d UTC$'}


def analyst_schema(context):
    return ANON_ANALYST_SCHEMA if context['decision_time'].startswith('DAY') else ANALYST_SCHEMA


def analyst(output, context):
    if analyst_schema(context) is ANALYST_SCHEMA:
        return shared.analyst(output, context)
    validate(ANON_ANALYST_SCHEMA, output)
    coverage([(v['asset'], v['horizon']) for v in output['views']],
             [(a, h) for a in context['universe'] for h in ('1d', '5d')])
    # Coded clocks are opaque citation identities. Parsing them as ISO leaked a ValueError.
    known = {(n['published_at'], n['url']) for g in context['headlines'].values() for n in g['items']}
    for view in output['views']:
        direction(view)
        if view['view'] != 'ABSTAIN' and not view['catalysts']:
            raise ReplayError('UNSUPPORTED_VIEW')
        if any((n['published_at'], n['url']) not in known for n in view['catalysts']):
            raise ReplayError('CITATION_NOT_IN_SNAPSHOT')
