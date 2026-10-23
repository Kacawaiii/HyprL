"""Evidence and judgments contain no prices, quantities or invented instruments."""
from copy import deepcopy

from scripts.trading_lab.agent_v2.contracts import SNAPSHOT as FROZEN_SNAPSHOT
from scripts.trading_lab.agent_v2.contracts import (  # shared strict JSON primitives
    Blocked, HASH, SYMBOL, TEXT, array, canonical, digest, enum, instant, iso,
    obj, strict_json, utc, validate,
)

SNAPSHOT = deepcopy(FROZEN_SNAPSHOT)
SNAPSHOT['properties']['schema'] = enum('agent-v2-quality-snapshot-v3')
SNAPSHOT['properties']['instruments']['items']['properties']['market_cap'] = {
    'anyOf': [{'type': 'number', 'exclusiveMinimum': 0}, {'type': 'null'}]}

FACT = obj({'symbol': SYMBOL, 'source_id': HASH,
            'kind': enum('guidance_raised', 'guidance_cut', 'one_off', 'dilution',
                         'accounting', 'legal', 'other'),
            'quote': {'type': 'string', 'minLength': 1, 'maxLength': 1500},
            'interpretation': TEXT})
EXTRACTION = obj({'facts': array(FACT, 200)})
JUDGMENT = obj({'reviews': array(obj({
    'plan_id': HASH, 'verdict': enum('KEEP', 'SIZE_DOWN', 'VETO'),
    'reason': TEXT, 'falsifier': TEXT, 'source_ids': array(HASH, 20),
}), 40)})
ARMS = ('baseline', 'extraction', 'veto', 'sizing', 'combined')
RED_FLAGS = ('guidance_cut', 'dilution', 'accounting', 'legal')


def check_facts(value, sources):
    validate(value, EXTRACTION)
    index = {s['id']: s for s in sources}
    seen = set()
    for fact in value['facts']:
        source = index.get(fact['source_id'])
        key = digest(fact)
        if (source is None or fact['symbol'] != source['symbol']
                or fact['quote'] not in source['text'] or key in seen):
            raise Blocked('FACT_CITATION_INVALID')
        seen.add(key)
    return value


def check_judgment(value, plans, sources):
    validate(value, JUDGMENT)
    reviews = {r['plan_id']: r for r in value['reviews']}
    if len(reviews) != len(value['reviews']) or set(reviews) != {p['plan_id'] for p in plans}:
        raise Blocked('JUDGMENT_PLAN_MISMATCH')
    for plan in plans:
        review = reviews[plan['plan_id']]
        allowed = {s['id'] for s in sources if s['symbol'] == plan['symbol']}
        if (not review['source_ids'] or len(set(review['source_ids'])) != len(review['source_ids'])
                or not set(review['source_ids']) <= allowed):
            raise Blocked('JUDGMENT_CITATION_INVALID')
    return reviews


def factors(plan, facts, analyst, contradictor, disabled=()):
    a, b = analyst['verdict'], contradictor['verdict']
    extraction = 0 if any(f['symbol'] == plan['symbol'] and f['kind'] in RED_FLAGS
                          for f in facts['facts']) else 1
    veto = 0 if 'VETO' in (a, b) else 1
    sizing = .5 if 'SIZE_DOWN' in (a, b) else 1
    values = {'baseline': 1, 'extraction': extraction, 'veto': veto, 'sizing': sizing}
    values['combined'] = (1 if 'extraction' in disabled else extraction) * (
        1 if 'veto' in disabled else veto) * (1 if 'sizing' in disabled else sizing)
    return values
