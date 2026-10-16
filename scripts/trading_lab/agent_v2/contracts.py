"""Strict JSON and small contracts shared by the isolated shadow runtime."""
from datetime import datetime, timezone
import hashlib
import json
import math

from jsonschema import Draft202012Validator


class Blocked(ValueError):
    pass


def utc():
    return datetime.now(timezone.utc)


def instant(value):
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            raise ValueError()
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        raise Blocked('INVALID_TIME') from None


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise Blocked('DUPLICATE_JSON_KEY')
            out[key] = value
        return out
    def constant(_):
        raise Blocked('NONFINITE_JSON')
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError) as error:
        if isinstance(error, Blocked):
            raise
        raise Blocked('INVALID_JSON') from None


def validate(value, schema):
    if not Draft202012Validator(schema).is_valid(value):
        raise Blocked('INVALID_SCHEMA')
    # jsonschema treats NaN as a number; reject it even for programmatic inputs.
    canonical(value)
    return value


def obj(properties, required=None):
    return {'type': 'object', 'properties': properties, 'required': list(properties) if required is None else required,
            'additionalProperties': False}


def array(item, maximum=1000):
    return {'type': 'array', 'items': item, 'maxItems': maximum}


TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 20000}
TIME = {'type': 'string', 'minLength': 20, 'maxLength': 35}
NUMBER = {'type': 'number', 'exclusiveMinimum': 0}
BOOL = {'type': 'boolean'}
SYMBOL = {'type': 'string', 'pattern': '^[A-Z][A-Z0-9.\\-]{0,9}$'}
HASH = {'type': 'string', 'pattern': '^[a-f0-9]{64}$'}


def enum(*values):
    return {'enum': list(values)}


SOURCE = obj({'id': TEXT, 'url': TEXT, 'published_at': TIME, 'received_at': TIME, 'text': TEXT,
              'symbol': SYMBOL, 'kind': enum('8-K', 'earnings', 'other'), 'sha256': HASH})
BAR = obj({'t': TIME, 'available_at': TIME, 'c': NUMBER, 'h': NUMBER, 'l': NUMBER, 'v': NUMBER})
INSTRUMENT = obj({'symbol': SYMBOL, 'asset': enum('equity', 'etf'), 'cluster': TEXT,
                  'market_cap': NUMBER, 'dollar_volume': NUMBER, 'primary_domains': array(TEXT, 10),
                  'bars': array(BAR, 400), 'momentum_member': BOOL})
QUOTE = obj({'symbol': SYMBOL, 'at': TIME, 'received_at': TIME, 'bid': NUMBER, 'ask': NUMBER,
             'bid_size': NUMBER, 'ask_size': NUMBER, 'source': enum('alpaca_iex', 'synthetic')})
CALENDAR_EVENT = obj({'kind': enum('CPI', 'FOMC', 'NFP', 'EARNINGS'), 'at': TIME,
                      'symbol': {'type': ['string', 'null']}})
SNAPSHOT = obj({'schema': enum('agent-v2-snapshot-v1'), 'mode': enum('SYNTHETIC', 'SHADOW_FORWARD'),
                'at': TIME, 'metadata_at': TIME, 'calendar_at': TIME,
                'instruments': array(INSTRUMENT, 600), 'sources': array(SOURCE, 100),
                'events': array(CALENDAR_EVENT, 100), 'quotes': array(QUOTE, 600),
                'hunter': obj({'asof': TEXT, 'generated': TIME, 'candidates': array({'type': 'object'}, 100)})})
QUOTES = obj({'schema': enum('agent-v2-quotes-v1'), 'mode': enum('SYNTHETIC', 'SHADOW_FORWARD'),
              'at': TIME, 'quotes': array(QUOTE, 600)})
NEWS_EVENT = obj({'source_id': TEXT, 'symbol': SYMBOL, 'kind': enum('8-K', 'earnings', 'other'),
                  'sentiment': enum('negative', 'neutral', 'positive'),
                  'code': enum('guidance_cut', 'earnings_miss', 'financing_distress', 'fraud_regulatory',
                               'beat_and_raise', 'other', 'no_event'),
                  'evidence': {'type': 'string', 'minLength': 1, 'maxLength': 1000}})
NEWS = obj({'events': array(NEWS_EVENT, 100)})
VETO_CODES = ('DATA_STALE', 'SOURCE_INVALID', 'EVENT_RISK', 'CAP', 'CLUSTER', 'CONFLICT', 'NO_EVIDENCE')
SUPERVISOR = obj({'reviews': array(obj({'plan_id': HASH, 'verdict': enum('ALLOW', 'VETO'),
                                      'reason': enum('NONE', *VETO_CODES)}), 100)})


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
