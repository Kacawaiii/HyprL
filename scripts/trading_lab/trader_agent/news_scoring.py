"""Prospective paired-session comparisons, apart from historical base scores."""
from collections import defaultdict

from .config import CRYPTO_PRODUCTS
from .news_spec import PREREG_HASH, preregistration, spec


def comparison(predictions, labels, summaries, metric):
    from . import retirement_spec
    rule = spec()['promotion']
    preregistration()
    complete = {s['run_id']: s['base_run_id'] for s in summaries
                if s.get('schema') == 'trader-news-run-v1' and s.get('status') in {'COMPLETE', 'DEGRADED'}}
    base_ids = set(complete.values())
    samples, issued = defaultdict(list), defaultdict(int)
    for row in predictions:
        p = row['payload']
        signal = p['signal']
        arm = 'news' if signal.get('arm') == 'news' else 'base'
        from .slots import PREREG_HASH as pm_hash
        retired = p['artifact_hash'] == retirement_spec.PREREG_HASH
        expected = retirement_spec.PREREG_HASH if retired else pm_hash if signal.get('slot') == 'pm' else PREREG_HASH
        if (arm == 'news' and (signal['run_id'] not in complete or p['artifact_hash'] != expected)
                or arm == 'base' and signal['run_id'] not in base_ids):
            continue
        definition, view = signal['label_definition'], signal['view']
        variant = definition.get('variant', '')
        population = ('weekend_crypto' if 'weekend_crypto' in variant else
                      'after_hours' if 'after_hours' in variant else 'catchup' if 'catchup' in variant else 'weekday')
        if signal.get('slot') == 'pm':
            population = 'pm_' + population
        if retired:
            population = retirement_spec.POPULATION + ':' + population
        target = 'raw' if p['product'] in CRYPTO_PRODUCTS else 'SPY_relative'
        key = (population, 'crypto' if target == 'raw' else 'equity_etf', definition['horizon'], target, view['analyst'])
        issued[(key, arm)] += 1
        label = labels.get(row['identity'])
        if label:
            value = label['value']
            r = value['raw_return' if target == 'raw' else 'spy_relative_return']
            samples[(key, arm)].append({'p': view['p_outperform'], 'view': view['view'], 'y': int(r > 0),
                'return': r, 'pnl': value['net_unit_pnl'], 'session': signal['session'], 'climatology': .5})
    keys = {key for key, _ in issued}
    if not keys:
        keys = {(population, asset_class, horizon, target, role)
                for population, asset_class, target in [('weekday', 'equity_etf', 'SPY_relative'),
                                                      ('weekend_crypto', 'crypto', 'raw')]
                for horizon in ('1d', '5d') for role in ('analyst_claude', 'analyst_gpt', 'reviewer_claude', 'reviewer_gpt', 'consensus')}
    comparisons = []
    for key in sorted(keys):
        entries = {}
        for arm in ('base', 'news'):
            result = metric(samples[(key, arm)], issued[(key, arm)])
            entries[arm] = {'n': result['non_abstained'], **{k: result[k] for k in
                ('issued', 'realized', 'pending', 'hit_rate', 'brier', 'abstention_rate')}}
        base, news = entries['base'], entries['news']
        enough = min(base['n'], news['n']) >= rule['minimum_directional_per_arm']
        state = ('TOO_EARLY' if not enough else 'DESCRIPTIVE_ONLY_NO_SPY_TARGET' if key[3] != 'SPY_relative' else
                 'ELIGIBLE_FOR_OPERATOR_REVIEW' if news['hit_rate'] > base['hit_rate'] and news['brier'] < base['brier'] else
                 'NOT_ELIGIBLE')
        comparisons.append(dict(zip(('population', 'asset_class', 'horizon', 'target', 'role'), key)) |
                           {'state': 'RETIRED' if key[4] in retirement_spec.RETIRED else state, **entries})
    return {'schema': 'trader-news-ab-scorecard-v1', 'preregistration_hash': PREREG_HASH,
            'mode': 'SHADOW_ONLY', 'promotion_rule': rule, 'comparisons': comparisons,
            'paired_base_runs': len(base_ids), 'complete_news_runs': len(complete),
            'role_lifecycle': retirement_spec.lifecycle()}
