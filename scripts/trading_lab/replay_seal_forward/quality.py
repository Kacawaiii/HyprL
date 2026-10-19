"""Outcome-independent data exclusions. Integrity and programming errors never enter this list."""
import math

from scripts.trading_lab.replay_bench.grant import ReplayError, instant

DATA_ERRORS = frozenset({
    'SEALED_SOURCE_DATA_MISSING', 'VENDOR_REQUEST_INDEX_MISSING', 'LIVE_VENDOR_REQUEST_INDEX_MISSING', 'LIVE_ARCHIVE_RECEIVED_AFTER_T',
    'ACTIVE_AND_INACTIVE_ASSETS_REQUIRED', 'NEWS_ARCHIVE_COVERAGE_MISSING',
    'CANDIDATE_HISTORY_COVERAGE_MISSING', 'FIXED_UNIVERSE_PRE_JULY_HISTORY_MISSING',
    'FIXED_UNIVERSE_HISTORY_INCOMPLETE', 'CANDIDATE_ASSET_METADATA_MISSING',
    'INVALID_ASSET_METADATA', 'CONTRADICTORY_ASSET_METADATA', 'INVALID_BAR',
    'CONTRADICTORY_BARS', 'DUPLICATE_PRICE_SESSION', 'PRICE_OFF_CALENDAR', 'CRYPTO_BAR_NOT_UTC_DAY',
    'INVALID_PRICE', 'MISSING_PRICES', 'DUPLICATE_PRICE', 'PRICE_GAP_OR_STALE',
    'INCOMPLETE_LABEL_COVERAGE', 'INCOMPLETE_MARK_PATH', 'MISSING_CRYPTO_MARK',
    'MISSING_SYNCHRONIZED_SPY_MARK', 'OFFLINE_CACHE_MISS',
})


def require_data_error(error):
    if error.code not in DATA_ERRORS:
        raise error
    return error.code


def valid_number(value, *, positive=True):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and (value > 0 if positive else value >= 0))


def validate_bar(row):
    try:
        instant(row['t'])
        if (not all(valid_number(row[k]) for k in ('o', 'h', 'l', 'c')) or
                not valid_number(row['v'], positive=False) or
                row['l'] > min(row['o'], row['c']) or row['h'] < max(row['o'], row['c']) or row['l'] > row['h']):
            raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError, ReplayError):
        raise ReplayError('INVALID_BAR') from None


def news_reason(row):
    if any(not row.get(k) for k in ('created_at', 'updated_at')):
        return 'NEWS_CLOCK_MISSING'
    try:
        created, updated = instant(row['created_at']), instant(row['updated_at'])
    except (TypeError, ValueError, AttributeError, ReplayError):
        return 'NEWS_CLOCK_INVALID'
    if updated < created:
        return 'NEWS_CLOCK_CONTRADICTORY'
    if (not isinstance(row.get('id'), int) or isinstance(row.get('id'), bool) or
            any(not isinstance(row.get(k), str) or not row[k] for k in ('headline', 'source')) or
            not isinstance(row.get('symbols'), list) or
            any(not isinstance(s, str) for s in row['symbols'])):
        return 'NEWS_FIELDS_INVALID'
    return None


def asset_reason(row):
    if row is None:
        return 'CANDIDATE_ASSET_METADATA_MISSING'
    if any(not isinstance(row.get(k), str) or not row[k] for k in ('symbol', 'class', 'name', 'exchange', 'status')):
        return 'INVALID_ASSET_METADATA'
    if any(k in row and not isinstance(row[k], bool) for k in ('tradable', 'shortable', 'easy_to_borrow')):
        return 'INVALID_ASSET_METADATA'
    return None


class DataTransport:
    """Distinguish a sealed cache's absent request from an altered bound object."""
    def __init__(self, transport):
        self.transport = transport

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def get(self, origin, path, params=None):
        try:
            return self.transport.get(origin, path, params)
        except ReplayError as error:
            if error.code == 'PREDICTION_CACHE_NOT_SEALED':
                import hashlib
                from urllib.parse import urlencode
                url = origin + path + ('?' + urlencode(sorted(params.items())) if params else '')
                key = hashlib.sha256(url.encode()).hexdigest()
                allowed = getattr(self.transport, 'allowed_cache', None)
                if allowed is not None and key not in allowed:
                    raise ReplayError('SEALED_SOURCE_DATA_MISSING') from None
            raise
