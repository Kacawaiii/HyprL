"""Offline cockpit decision chains, overlays and descriptive news comparisons.

Only explicit journal results are scored. An intent is not a fill, an open P&L is
not a realized result, and an asset label is not evidence a radar event caused it.
"""
from collections import defaultdict
from urllib.parse import urlsplit

from .cockpit_export import assert_clean, clean_url, digest, norm_symbol, number, text, utc_now

SCHEMA = 'cockpit-analysis-v1'
PRIMARY_HOSTS = ('sec.gov', 'federalreserve.gov', 'bls.gov', 'fcc.gov')
MIN_DESCRIPTIVE_N = 30  # Display caution only; never a significance/validation threshold.


def source(value):
    if isinstance(value, str):
        value = {'url': value}
    url = clean_url(value.get('url') or value.get('source_url'))
    host = (urlsplit(url).hostname or '') if url else ''
    official_host = any(host == h or host.endswith('.' + h) for h in PRIMARY_HOSTS)
    # A company primary source needs explicit provenance; a publisher's name alone is not proof.
    primary = bool(url and (value.get('primary') is True or official_host and value.get('primary') is not False))
    return {'url': url, 'publisher': text(value.get('publisher') or value.get('source'), 80),
            'published_at': text(value.get('published_at'), 40),
            'verification': 'OFFICIEL' if primary else 'FIL' if url else 'NON VERIFIE'}


def sources(values):
    if isinstance(values, str):
        values = [values]
    return [source(v) for v in (values or [])[:5] if isinstance(v, (str, dict))]


def chain(*, fact=None, citations=None, expectations=None, expectation_at=None, priced_in=None,
          scenario=None, entry=None, invalidation=None, result=None):
    refs = sources(citations)
    return {'fact': text(fact, 400), 'sources': refs,
            'verification': refs[0]['verification'] if refs else 'NON VERIFIE',
            'expectations': text(expectations, 400), 'expectation_at': text(expectation_at, 40),
            'priced_in': text(priced_in, 400), 'scenario': text(scenario, 400),
            'entry_condition': text(entry, 400), 'invalidation': text(invalidation, 400),
            'result': result}


def radar_chain(event):
    scenario = event.get('scenario') or {}
    return chain(fact=event.get('headline'), citations=event.get('stories'),
                 expectations=event.get('expectations'), expectation_at=event.get('expectations_at'),
                 priced_in=scenario.get('priced_in'), scenario=scenario.get('summary') or scenario.get('impact'),
                 entry=scenario.get('entry_condition'), invalidation=scenario.get('invalidation'))


def trader_chain(view, *, entry=None, result=None):
    raw = view.get('raw_view') or {}
    citations = raw.get('catalysts') or []
    # Catalyst fact is a short model citation, never an article body.
    fact = '; '.join(c.get('fact', '') for c in citations if isinstance(c.get('fact'), str)) or None
    return chain(fact=fact, citations=citations, expectations=raw.get('expectations'),
                 expectation_at=raw.get('expectations_at'), priced_in=raw.get('priced_in_assessment'),
                 scenario=raw.get('confidence_reason'), entry=raw.get('entry_condition') or entry,
                 invalidation=raw.get('falsifier'), result=result)


def label_result(label):
    return None if not label else {'pnl_after_costs': None, 'r_after_costs': None,
        'net_return': label['net_unit_pnl'], 'costs': label['cost_roundtrip'],
        'at': label['available_at'], 'basis': 'modelled_roundtrip_cost'}


def journal_result(row, intent=None):
    value = row.get('result') or (row.get('journal') or {}).get('result') or {}
    if not isinstance(value, dict):
        return None
    net = number(value.get('pnl_after_costs'))
    costs = number(value.get('costs'))
    if net is None and costs is not None and number(value.get('gross_pnl')) is not None:
        net = number(number(value['gross_pnl']) - costs)
    # Explicit after-cost numbers may be served without an itemized cost breakdown.
    r = number(value.get('r_after_costs'))
    risk = number((intent or {}).get('risk_usd'))
    if r is None and net is not None and risk is not None and risk > 0:
        r = number(net / risk)
    if net is None and r is None:
        return None
    return {'pnl_after_costs': net, 'r_after_costs': r, 'net_return': None,
            'costs': costs, 'at': text(row.get('at'), 40), 'basis': 'journal_explicit_after_costs'}


def journal_chain(row, *, result=None):
    j = row.get('journal') or {}
    return chain(fact=j.get('event') or row.get('fact') or j.get('reason') or row.get('reason'),
                 citations=j.get('sources'), expectations=j.get('expectations'),
                 expectation_at=j.get('expectations_at'), priced_in=j.get('priced_in'),
                 scenario=j.get('scenario'), entry=j.get('plan'), invalidation=j.get('invalidation'), result=result)


def book_trades(journal):
    trades, open_by_symbol = [], defaultdict(list)
    for row in sorted(journal, key=lambda r: r.get('at') or ''):
        symbol = norm_symbol(row.get('symbol'))
        if row.get('action') == 'buy_intent' and symbol:
            j = row.get('journal') or {}
            tag = text(j.get('tag') or row.get('tag') or j.get('engine') or 'unclassified', 60)
            group = 'news' if tag.lower() in ('news', 'news_driven', 'actualité', 'actualite') else (
                'technical' if tag.lower() in ('technical', 'technique', 'momo_v0', 'momentum', 'sleeve') else 'unclassified')
            trade = {'id': digest({'at': row.get('at'), 'symbol': symbol, 'index': len(trades)}),
                     'symbol': text(row.get('symbol'), 24), 'at': text(row.get('at'), 40), 'tag': tag,
                     'group': group, 'state': 'INTENT', 'closed_at': None, 'risk_usd': number(row.get('risk_usd')),
                     'entry': number(row.get('limit')), 'stop': number(row.get('stop')), 'target': number(row.get('target')),
                     'decision': journal_chain(row)}
            trades.append(trade)
            open_by_symbol[symbol].append((trade, row))
        elif row.get('action') in ('closed', 'close', 'post_mortem') and symbol:
            candidates = open_by_symbol[symbol]
            # No guessing which lot a close belongs to when several intentions overlap.
            if len(candidates) == 1:
                trade, intent = candidates.pop()
                trade['state'] = 'CLOSED'
                trade['closed_at'] = text(row.get('at'), 40)
                trade['decision']['result'] = journal_result(row, intent)
    return trades


def news_stats(trades):
    out = []
    for group in ('news', 'technical', 'unclassified'):
        rows = [t for t in trades if t['group'] == group]
        pnl = [t['decision']['result']['pnl_after_costs'] for t in rows
               if t['state'] == 'CLOSED' and t['decision']['result'] and t['decision']['result']['pnl_after_costs'] is not None]
        rs = [t['decision']['result']['r_after_costs'] for t in rows
              if t['state'] == 'CLOSED' and t['decision']['result'] and t['decision']['result']['r_after_costs'] is not None]
        out.append({'group': group, 'count': len(rows), 'closed': sum(t['state'] == 'CLOSED' for t in rows),
                    'scored': len(pnl), 'r_samples': len(rs),
                    'hit_rate': number(sum(p > 0 for p in pnl) / len(pnl)) if pnl else None,
                    'average_r': number(sum(rs) / len(rs)) if rs else None,
                    'pnl_after_costs': number(sum(pnl), 2) if pnl else None,
                    'state': 'TOO_EARLY' if len(pnl) < MIN_DESCRIPTIVE_N else 'DESCRIPTIVE_ONLY'})
    return out


def build_analysis(radar, trader=None, paper=None, journal=None, *, health=None, units=None, generated_at=None):
    trader = trader or {}
    at = generated_at or utc_now()
    prices, overlays = defaultdict(list), []
    for c in trader.get('contexts', []):
        for asset, p in c.get('prices', {}).items():
            closes = p.get('recent_closes') or []
            price = number(closes[-1]) if closes and p.get('state') == 'AVAILABLE' else None
            if price is not None:
                prices[asset].append({'at': p.get('last_price_at'), 'price': price, 'basis': 'reference_close'})
    for asset, p in (radar.get('regime') or {}).items():
        if number(p.get('last')) is not None and p.get('at'):
            prices[p.get('symbol') or asset].append({'at': p['at'], 'price': number(p['last']), 'basis': 'radar_observation'})
    for label in trader.get('labels', []):
        for key in ('entry', 'exit'):
            if number(label.get(key + '_price')) is not None and label.get(key + '_at'):
                prices[label['asset']].append({'at': label[key + '_at'], 'price': number(label[key + '_price']), 'basis': 'label_price'})
        overlays.append({'id': digest(label), 'kind': 'outcome', 'asset': label['asset'], 'at': label['realized_at'],
                         'horizon': label['horizon'], 'analyst': label['model_id'].split(':')[-1],
                         'label': 'Résultat IA après coûts', 'net_return': label['net_unit_pnl'], 'decision': None})
    labels = {(l['run_id'], norm_symbol(l['asset']), l['model_id'].split(':')[-1], l['horizon']): l
              for l in trader.get('labels', [])}
    for run in trader.get('runs', [])[-60:]:
        for v in run['views']:
            label = labels.get((run['run_id'], norm_symbol(v.get('asset')), v.get('analyst'), v.get('horizon')))
            overlays.append({'id': digest([run['run_id'], v.get('asset'), v.get('analyst'), v.get('horizon')]),
                             'kind': 'decision', 'asset': v.get('asset'), 'at': run['at'], 'horizon': v.get('horizon'),
                             'analyst': v.get('analyst'), 'label': v.get('view'), 'verdict': v.get('verdict'),
                             'decision': trader_chain(v, entry=(run.get('entries') or {}).get(v.get('horizon')), result=label_result(label))})
    for e in radar.get('events', []):
        for asset in list(dict.fromkeys(h.get('symbol') for h in e.get('transmission_hypotheses', []) if h.get('symbol')))[:8]:
            overlays.append({'id': digest([e['id'], asset]), 'kind': 'event', 'asset': asset,
                             'at': e.get('first_received_at'), 'horizon': None, 'label': text(e.get('headline'), 300),
                             'decision': radar_chain(e)})
    trades = book_trades(journal or [])
    for t in trades:
        for kind, time in [('entry', t['at']), ('exit', t['closed_at'])]:
            if time:
                overlays.append({'id': t['id'] + kind, 'kind': 'book_' + kind, 'asset': t['symbol'], 'at': time,
                                 'horizon': None, 'label': 'Claude book ' + t['tag'], 'price': t['entry'] if kind == 'entry' else None,
                                 'price_basis': 'intent_limit' if kind == 'entry' else None, 'stop': t['stop'] if kind == 'entry' else None,
                                 'decision': t['decision']})
    for a in (paper or {}).get('accounts', []):
        if a['account'] == 'claude_book':
            for p in a['positions']:
                if p.get('last') is not None:
                    prices[p['symbol']].append({'at': a.get('observed_at') or (paper or {}).get('generated_at'),
                                               'price': p['last'], 'basis': 'cached_book_mark'})
    snapshot = {'schema': SCHEMA, 'generated_at': at, 'prices': dict(prices), 'overlays': overlays[-12000:],
                'book_trades': trades[-300:], 'news': {'groups': news_stats(trades), 'minimum_n': MIN_DESCRIPTIVE_N,
                  'by_tag': [{**news_stats([{**t, 'group': 'news'} for t in trades if t['tag'] == tag])[0], 'group': tag}
                             for tag in sorted({t['tag'] for t in trades})],
                  'tag_counts': {tag: sum(t['tag'] == tag for t in trades) for tag in sorted({t['tag'] for t in trades})},
                  'ai_scores': trader.get('scorecard', {}).get('scores', {}) if trader.get('scorecard') else {},
                  'context_comparison': 'UNAVAILABLE',
                  'note': 'Aucune variante appariée avec/sans actualité identifiée. Les baselines ne sont pas une ablation news.'},
                'system': {'radar_at': radar.get('cutoff'), 'sources': [
                  {'source': text(s.get('source'), 100), 'status': text(s.get('status'), 40),
                   'reason': text(s.get('reason'), 160), 'checked_at': text(s.get('checked_at'), 40),
                   'items': number(s.get('items'), 0)} for s in radar.get('sources', [])],
                  'budget_date': radar.get('date'), 'budgets_used': radar.get('budgets_used') or {},
                  'trader': health or {}, 'units': units or []},
                'limitations': ['Prix observés seulement : aucune histoire continue inventée entre deux observations.',
                                 'Un événement est placé à sa réception, pas à la date où sa conséquence devient connue.',
                                 'Une intention Claude book ne prouve pas une exécution; son prix limite est étiqueté.',
                                 'Comparaisons descriptives, sans attribution causale ni avantage démontré.']}
    assert_clean(snapshot)
    return snapshot
