"""A per-event quantity ledger, with units and explicit rounding tolerance."""
from decimal import Decimal, InvalidOperation
import re

from .core import RadarError
from .entities import fold, NAMES, ALIASES
from .registry import CRYPTO_NAMES
from .relationships import EXTERNAL_NAMES

NUMBER = re.compile(r'(?<![\w])(?P<prefix>[$€])?\s*(?P<number>[+−-]?\d+(?:[.,\u202f\u00a0 ]\d{3})*(?:[.,]\d+)?)(?!\d|[.,]\d)(?P<tail>\s*(?:trillions?|billions?|millions?|milliards?|mille|thousand|[kmb](?!\w))?\s*(?:de\s+)?(?:%|percent|pour cent|ATR|bps|points? de base|×|x(?!\w)|dollars?|euros?|USD|EUR)?)', re.I)
SCALES = {'trillion': 12, 'trillions': 12, 'billion': 9, 'billions': 9, 'milliard': 9, 'milliards': 9,
          'million': 6, 'millions': 6, 'm': 6, 'b': 9, 'k': 3, 'mille': 3, 'thousand': 3}


def named_subject(text, position, symbols):
    matches = []
    normalized = fold(text)
    for symbol in symbols:
        names = [symbol, NAMES.get(symbol, ''), *ALIASES.get(symbol, []), *EXTERNAL_NAMES.get(symbol, []),
                 CRYPTO_NAMES.get(symbol.split('/')[0], '')]
        for name in filter(None, names):
            for match in re.finditer(r'(?<!\w)' + re.escape(fold(name)) + r'(?!\w)', normalized):
                if match.end() <= position:
                    matches.append((position - match.end(), symbol))
    return min(matches)[1] if matches else None


def movement(text, position):
    context = fold(text[max(0, position - 65):position])
    down = list(re.finditer(r'\b(fall\w*|drop\w*|plunge\w*|sink\w*|sank|down|loss\w*|baisse|recul\w*|chute\w*)\b', context))
    up = list(re.finditer(r'\b(rise\w*|rose|gain\w*|gagne\w*|up|hausse|surge\w*)\b', context))
    if down and (not up or down[-1].start() > up[-1].start()):
        return 'down'
    if up:
        return 'up'
    return None


def quantities(text, *, french=False):
    for match in NUMBER.finditer(text):
        # Numeric identifiers such as Q3, 1d or the issuer 3M are not quantities.
        if not match['tail'].strip() and match.end('number') < len(text) and text[match.end('number')].isalnum():
            continue
        if match.group().strip() in ('1m', '1d', '5d', '3M'):
            continue
        raw = match['number'].replace('−', '-').replace('\u202f', '').replace('\u00a0', '').replace(' ', '')
        # Captures use English grouping; French output accepts decimal commas.
        if not french and re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', raw):
            raw = raw.replace(',', '')
        elif ',' in raw and '.' in raw:
            raw = raw.replace('.', '').replace(',', '.')
        else:
            raw = raw.replace(',', '.')
        try:
            value = Decimal(raw)
        except InvalidOperation:
            raise RadarError('LLM_NUMBER_FORMAT') from None
        tail = fold(match['tail']).strip()
        scale = next((SCALES[w] for w in tail.split() if w in SCALES), 0)
        unit = ('pct' if re.search(r'%|percent|pour cent', tail) else
                'atr' if 'atr' in tail else 'bps' if re.search(r'bps|point.*base', tail) else
                'ratio' if re.search(r'×|\bx\b', tail) else
                'usd' if match['prefix'] == '$' or re.search(r'dollar|usd', tail) else
                'eur' if match['prefix'] == '€' or re.search(r'euro|eur', tail) else 'number')
        yield {'value': str(value * (10 ** scale)), 'unit': unit,
               'quantum': str(Decimal(10) ** (value.as_tuple().exponent + scale)),
               'literal': match.group().strip(), 'start': match.start(), 'end': match.end()}


def ledger(event, regime=None):
    result = []
    for story in event['stories']:
        for text in (story['headline'], story['summary']):
            for number in quantities(text):
                symbol = named_subject(text, number['start'], event['entities']['symbols'] + event['entities'].get('businesses', []))
                result.append({**number, 'subject': symbol or 'event', 'source': story['id'],
                               'movement': movement(text, number['start']) if number['unit'] in ('pct', 'atr') else None,
                               'basis': 'captured_headline_or_summary'})
    for row in event.get('priced_in', {}).get('measurements', []):
        for key, unit in [('move_atr', 'atr'), ('return_pct', 'pct')]:
            if row.get(key) is not None:
                result.append({'value': str(row[key]), 'unit': unit, 'subject': row['symbol'], 'basis': key})
    for label, row in (regime or {}).items():
        if row.get('last') is not None:
            result.append({'value': str(row['last']), 'unit': 'number', 'subject': row['symbol'], 'label': label, 'basis': 'regime_price'})
        for horizon, value in row.get('returns_pct', {}).items():
            if value is not None:
                result.append({'value': str(value), 'unit': 'pct', 'subject': row['symbol'], 'label': label, 'horizon': horizon, 'basis': 'regime_return'})
    return result


def check_numbers(text, facts, *, subject=None):
    normalized = fold(text)
    # Quantities must use digits; grammatical articles remain unrestricted.
    if re.search(r'\b(?:zero|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|cent)\b', normalized):
        raise RadarError('LLM_NUMBER_UNSUPPORTED')
    horizons = {'1d': ['1d', 'un jour', 'une seance'], '5d': ['5d', 'cinq jours', 'cinq seances'],
                '1m': ['1m', 'un mois']}
    requested_horizons = {h for h, names in horizons.items() if any(re.search(r'(?<!\w)' + re.escape(n) + r'(?!\w)', normalized) for n in names)}
    for number in quantities(text, french=True):
        value, tolerance = Decimal(number['value']), Decimal(number['quantum']) / 2
        claim_subject = subject or named_subject(text, number['start'], {f['subject'] for f in facts if f['subject'] != 'event'})
        claim_movement = movement(text, number['start']) if number['unit'] in ('pct', 'atr') else None
        def supported(fact):
            if fact['unit'] != number['unit']:
                return False
            if fact.get('horizon') and requested_horizons and fact['horizon'] not in requested_horizons:
                return False
            if fact['subject'] != 'event':
                if claim_subject and claim_subject != fact['subject']:
                    return False
                names = [fact['subject'], fact.get('label', ''), NAMES.get(fact['subject'], '')]
                if claim_subject != fact['subject'] and not any(name and fold(name) in normalized for name in names):
                    return False
            captured = Decimal(fact['value'])
            if claim_movement and fact.get('movement') and claim_movement != fact['movement']:
                return False
            if fact.get('movement') == 'down':
                captured = -abs(captured)
            compared = -abs(value) if claim_movement == 'down' else value
            # Rounding must never change the sign of an observed move.
            return (compared == 0 or captured == 0 or (compared > 0) == (captured > 0)) and abs(compared - captured) <= tolerance
        if not any(supported(fact) for fact in facts):
            error = RadarError('LLM_NUMBER_UNSUPPORTED')
            error.detail = {'quantity': number['literal'], 'unit': number['unit'], 'subject': claim_subject,
                            'rule': 'same captured subject/unit/horizon, sign and half-last-digit rounding'}
            raise error
