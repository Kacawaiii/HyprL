"""Offline, deterministic scope and SEC SIC sectors for the shadow registry.

SIC ranges use SEC's 1987 SIC divisions/industry groups. This fixed risk-sector
aggregation is a governance mapping, not GICS or a source claim about an issuer.
Source: https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list
"""
from collections import Counter
import math
import re

from .contracts import Blocked

# Ordered, disjoint inclusive ranges; unlisted and missing codes fail closed.
SIC_SECTORS = (
    (100, 999, 'consumer'),       # agriculture, forestry, fishing
    (1000, 1299, 'industrials'),  # metals and coal mining
    (1300, 1399, 'energy'),       # oil/gas extraction
    (1400, 1799, 'industrials'),  # nonmetal mining and construction
    (2000, 2799, 'consumer'),     # food, textiles, wood, paper, printing
    (2800, 2829, 'industrials'),  # chemicals/plastics
    (2830, 2839, 'health'),       # pharmaceuticals
    (2840, 2899, 'industrials'),  # other chemicals
    (2900, 2999, 'energy'),       # petroleum refining
    (3000, 3569, 'industrials'),  # materials and machinery
    (3570, 3579, 'technology'),   # computing equipment
    (3580, 3659, 'industrials'),
    (3660, 3699, 'technology'),   # communications/electronics
    (3700, 3809, 'industrials'),  # transport equipment
    (3810, 3839, 'technology'),   # instruments
    (3840, 3859, 'health'),       # medical/ophthalmic devices
    (3860, 3999, 'industrials'),
    (4000, 4799, 'industrials'),  # transport
    (4800, 4899, 'technology'),   # communications
    (4900, 4999, 'utilities'),
    (5000, 5039, 'industrials'),  # durable wholesale
    (5040, 5046, 'technology'),
    (5047, 5047, 'health'),       # medical wholesale
    (5048, 5119, 'industrials'),
    (5120, 5129, 'health'),       # drug wholesale
    (5130, 5169, 'consumer'),
    (5170, 5179, 'energy'),       # petroleum wholesale
    (5180, 5999, 'consumer'),     # other wholesale and retail
    (6000, 6499, 'financials'),
    (6500, 6599, 'real_estate'),
    (6700, 6797, 'financials'),
    (6798, 6798, 'real_estate'),  # REITs
    (6799, 6999, 'financials'),
    (7000, 7299, 'consumer'),
    (7300, 7369, 'industrials'),
    (7370, 7379, 'technology'),   # software/data processing
    (7380, 7999, 'consumer'),
    (8000, 8099, 'health'),
    (8100, 8999, 'industrials'),
)


def sic_sector(sic):
    # The submissions wire uses a string; do not coerce floats/bools/unknowns.
    if not isinstance(sic, str) or not re.fullmatch(r'[0-9]{3,4}', sic):
        return None
    number = int(sic)
    return next((sector for low, high, sector in SIC_SECTORS if low <= number <= high), None)


def select_scope(ranked, hunter_symbols):
    """Top 60 by score DESC, symbol ASC, then hunter symbol ASC; SPY uses one slot.

    Select before metadata exclusions, with no backfill. The 199-equity cap
    preserves momentum priority and a total registry limit of 200 including SPY.
    """
    if (len({r['symbol'] for r in ranked}) != len(ranked)
            or any(type(r['score']) not in (int, float) or not math.isfinite(r['score']) for r in ranked)):
        raise Blocked('MOMENTUM_RANKING_INVALID')
    top = [r['symbol'] for r in sorted(ranked, key=lambda r: (-r['score'], r['symbol']))[:60]]
    ordered = list(dict.fromkeys(top + sorted(set(hunter_symbols))))
    if 'SPY' in ordered:
        raise Blocked('EQUITY_SCOPE_CONTAINS_REGIME_ETF')
    return ordered[:199], {'ranked': len(ranked), 'momentum_top': top,
                           'hunter_count': len(set(hunter_symbols)), 'union': len(ordered),
                           'excluded_cap': len(ordered[199:])}


def admissible_registry(ranked, hunter_symbols, metadata, earnings, submissions):
    """Return admitted equities and an exhaustive reason ledger, without I/O.

    Earnings evidence must be a source date plus URL and response-body digest.
    SIC identity must match the exact zero-padded submissions CIK. Exclusions
    have all applicable reasons; primary counts are disjoint in the order below.
    """
    scope, selection = select_scope(ranked, hunter_symbols)
    top = set(selection['momentum_top'])
    rows, excluded = [], []
    for symbol in scope:
        meta = metadata.get(symbol, {})
        dates = earnings.get(symbol, [])
        sec = submissions.get(symbol, {})
        reasons = []
        if not dates:
            reasons.append('EARNINGS_DATE_UNVERIFIED')
        elif any(not re.fullmatch(r'\d{4}-\d{2}-\d{2}', e.get('date', ''))
                 or not e.get('source_url', '').startswith('https://')
                 or not re.fullmatch('[a-f0-9]{64}', e.get('source_hash', '')) for e in dates):
            raise Blocked('EARNINGS_EVIDENCE_INVALID')
        cik = meta.get('cik')
        if not isinstance(cik, str) or not re.fullmatch(r'\d{10}', cik):
            reasons.append('CIK_UNVERIFIED')
        if not sec.get('sic'):
            reasons.append('SIC_MISSING')
        elif sec.get('cik') != cik:
            reasons.append('SIC_IDENTITY_MISMATCH')
        elif not sic_sector(sec['sic']):
            reasons.append('SIC_UNMAPPED')
        cap = meta.get('market_cap')
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
            reasons.append('MARKET_CAP_UNVERIFIED')
        if reasons:
            excluded.append({'symbol': symbol, 'reasons': reasons})
            continue
        rows.append({**meta, 'symbol': symbol, 'asset': 'equity', 'cluster': sic_sector(sec['sic']),
                     'momentum_member': symbol in top,
                     'hunter_candidate': symbol in set(hunter_symbols), 'sic': sec['sic']})
    return rows, {**selection, 'selected_equities': len(scope), 'admitted_equities': len(rows),
                  'excluded': excluded,
                  'excluded_by_primary_reason': dict(Counter(r['reasons'][0] for r in excluded)),
                  'excluded_by_any_reason': dict(Counter(reason for r in excluded for reason in r['reasons']))}
