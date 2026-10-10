"""Generate an UNSIGNED draft for operator review. Never grants itself access."""
from datetime import timedelta
from urllib.parse import urlsplit
import argparse
import json

from .core import iso, now, write_private
from .registry import FEEDS


def draft(at):
    scope = {}
    for _, url, _, _ in FEEDS:
        parsed = urlsplit(url)
        origin = 'https://' + parsed.netloc
        rule = scope.setdefault(origin, {'methods': ['GET'], 'paths': []})
        rule['paths'].append(parsed.path)
    scope.update({
        'https://data.alpaca.markets': {'methods': ['GET'], 'paths': [
            '/v1beta1/news', '/v1beta1/screener/stocks/most-actives',
            '/v1beta1/screener/stocks/movers', '/v1beta1/screener/crypto/movers',
            '/v2/stocks/bars', '/v1beta3/crypto/us/bars']},
        'https://api.gdeltproject.org': {'methods': ['GET'], 'paths': ['/api/v2/doc/doc']},
        'https://www.youtube.com': {'methods': ['GET'], 'paths': ['/feeds/videos.xml'], 'path_prefixes': ['/@']},
        'https://query1.finance.yahoo.com': {'methods': ['GET'], 'path_prefixes': ['/v8/finance/chart/']},
    })
    return {'authorization': 'news-radar-v1', 'operator_signed': False,
            'granted_at': iso(at), 'not_after': iso(at + timedelta(days=7)),
            'scope': scope,
            'budgets': {'alpaca': 400, 'gdelt': 60, 'rss': 300, 'youtube': 400, 'market': 24, 'llm': 2},
            'spacing_seconds': {'gdelt': 7, 'rss': 900, 'youtube': 900},
            'llm': {'cli': 'claude', 'model': 'sonnet', 'tools': [], 'max_calls_per_day': 2},
            'purpose': 'Independent headline/summary radar and observational paper follow-up only; no broker or trader context changes.',
            'review_required': 'Operator must review scope, expiry and budgets, and personally mark this file signed. SEC/Fed are disabled. Yahoo charts supply the exact regime instruments absent from Alpaca.'}


def main():
    parser = argparse.ArgumentParser(description='UNSIGNED operator draft; network is never contacted')
    parser.add_argument('output')
    args = parser.parse_args()
    write_private(args.output, json.dumps(draft(now()), ensure_ascii=False, indent=2) + '\n')
    print('UNSIGNED_DRAFT_WRITTEN; not an authorization')


if __name__ == '__main__':
    main()
