"""Forward decisions and separate quote-time execution, all mutations journaled."""
from datetime import timedelta

from .contracts import Blocked, SUPERVISOR, digest, instant, iso, utc, validate
from .core import candidates, features, plan_id, ranking
from .inputs import classify, fresh, holding_days, quote_map, session, validate_quotes, validate_snapshot
from .models import SyntheticClaude
from .risk import equity, exposure, governor, size
from .spec import spec


class Service:
    def __init__(self, ledger, runner, *, clock=utc):
        self.ledger, self.runner, self.clock = ledger, runner, clock
        self.settings = spec()
        self.mode = ledger.binding['mode']
        if self.mode == 'SHADOW_FORWARD' and isinstance(runner, SyntheticClaude):
            raise Blocked('SYNTHETIC_RUNNER_FORBIDDEN')
        if self.mode == 'SHADOW_FORWARD' and not ledger.records('activation'):
            raise Blocked('RUNTIME_NOT_ACTIVATED')

    def blocked(self, action, error):
        if str(error) not in ('MARKET_CLOSED', 'OUTSIDE_DECISION_SLOT'):
            self.ledger.append([('blocked', {'action': action, 'reason': str(error)})], self.clock())

    def decide(self, snapshot):
        with self.ledger.owned():
            try:
                return self._decide(snapshot)
            except Blocked as error:
                self.blocked('decide', error)
                raise

    def _decide(self, snapshot):
        started = self.clock()
        day, opened, closed = session(started)
        # Two predeclared slots; do not backfill missed decisions.
        slots = [(opened + timedelta(minutes=10), 'open'), (closed - timedelta(minutes=20), 'close')]
        found = [(time, slot) for time, slot in slots if 0 <= (started - time).total_seconds() <= 180]
        if not found:
            raise Blocked('OUTSIDE_DECISION_SLOT')
        _, slot = found[0]
        cycle = day + ':' + slot
        if any(r['body']['cycle'] == cycle for r in self.ledger.records('cycle_start')):
            raise Blocked('SLOT_ALREADY_ATTEMPTED')
        instruments = validate_snapshot(snapshot, started, self.mode)
        self.ledger.append([('cycle_start', {'cycle': cycle, 'snapshot_hash': digest(snapshot)})], started)
        news_output = self.runner.call('news', {'at': iso(started), 'sources': snapshot['sources']})
        now = self.clock()
        validate_snapshot(snapshot, now, self.mode)
        news = classify(news_output, snapshot, now)
        state = self.ledger.state()
        trail_events = []
        for symbol, pos in state['positions'].items():
            instrument = instruments.get(symbol)
            if instrument is None:
                raise Blocked('HELD_INSTRUMENT_MISSING')
            if instant(instrument['bars'][-1]['available_at']) < instant(pos['filled_at']):
                continue
            close = instrument['bars'][-1]['c']
            best = max(pos.get('best_close', pos['entry']), close)
            stop = pos['stop']
            if pos['playbook'] == 'hunter_runner' and best >= pos['entry'] * 1.5:
                stop = max(stop, best * .75)
            elif pos['playbook'] in ('equity_momentum', 'etf_trend', 'primary_news'):
                feature = features(instrument['bars'], 100 if pos['playbook'] == 'etf_trend' else 20)
                if feature:
                    stop = max(stop, close - 3 * feature['atr'])
            update = {'symbol': symbol, 'stop': stop, 'best_close': best}
            pos.update(update)
            trail_events.append(('trail', update))
        quotes = quote_map(snapshot, now, self.mode, max_age=300)
        if any(s not in quotes for s in state['positions']):
            raise Blocked('HELD_QUOTE_MISSING')
        raw, ranks = candidates(snapshot, instruments, news, now, self.settings)
        events = trail_events + [('snapshot', {'cycle': cycle, 'hash': digest(snapshot), 'snapshot': snapshot}),
                  ('extraction', {'cycle': cycle, 'events': news})]
        plans = []
        for candidate in raw:
            plan, reason = size(candidate, state, quotes, snapshot, news, self.settings, now)
            if reason:
                events.append(('veto', {'cycle': cycle, 'symbol': candidate['symbol'], 'layer': candidate['layer'],
                                        'by': 'risk', 'reason': reason}))
                continue
            plan.update({'cycle': cycle, 'planned_at': iso(now), 'snapshot_hash': digest(snapshot),
                         'standing_exit': {'side': 'SELL', 'max_qty': plan['qty'], 'stop': plan['stop'],
                                           'holding_days': plan['holding_days'],
                                           'triggers': ['STOP', 'TIME_STOP', 'NEGATIVE_PRIMARY_NEWS', 'MOMENTUM_EXIT']}})
            plan['plan_id'] = plan_id(plan)
            plans.append(plan)
            # Reserve all hard risks before reviewing any plan in this batch.
            state['pending'][plan['plan_id']] = plan
            events.append(('plan', plan))
        self.ledger.append(events, now)
        if plans:
            context = {'at': iso(now), 'plans': plans, 'rules': self.settings,
                       'equity': equity(state, quotes), 'exposure': exposure(state, quotes, self.settings),
                       'calendar': snapshot['events'], 'calendar_at': snapshot['calendar_at'],
                       'metadata_at': snapshot['metadata_at'], 'instruments': [
                           {k: r[k] for k in ('symbol', 'asset', 'cluster', 'dollar_volume', 'market_cap')}
                           for r in snapshot['instruments']],
                       'quotes': snapshot['quotes'], 'news': news}
            try:
                result = self.runner.call('supervisor', context)
                validate(result, SUPERVISOR)
                ids = [r['plan_id'] for r in result['reviews']]
                if len(ids) != len(set(ids)) or set(ids) != {p['plan_id'] for p in plans}:
                    raise Blocked('SUPERVISOR_PLAN_MISMATCH')
                for review in result['reviews']:
                    if (review['verdict'] == 'ALLOW') != (review['reason'] == 'NONE'):
                        raise Blocked('SUPERVISOR_REASON_INVALID')
            except Blocked as error:
                self.ledger.append([('veto', {'cycle': cycle, 'plan_id': p['plan_id'], 'symbol': p['symbol'],
                                            'layer': p['layer'], 'by': 'supervisor', 'reason': str(error)})
                                    for p in plans], self.clock())
                raise
            approved_at = self.clock()
            if (approved_at - started).total_seconds() > 300:
                raise Blocked('DECISION_EXPIRED')
            reviews = {r['plan_id']: r for r in result['reviews']}
            result_events = []
            for plan in plans:
                review = reviews[plan['plan_id']]
                result_events.append(('supervisor', review))
                if review['verdict'] == 'VETO':
                    result_events.append(('veto', {'cycle': cycle, 'plan_id': plan['plan_id'], 'symbol': plan['symbol'],
                                                   'layer': plan['layer'], 'by': 'supervisor', 'reason': review['reason']}))
                else:
                    result_events.append(('approved', {**plan, 'approved_at': iso(approved_at),
                                                       'expires_at': iso(approved_at + timedelta(minutes=5))}))
            self.ledger.append(result_events, approved_at)
        self.ledger.append([('cycle_complete', {'cycle': cycle, 'plans': len(plans)})], self.clock())
        return {'cycle': cycle, 'plans': len(plans)}

    def watch(self, quote_snapshot):
        with self.ledger.owned():
            try:
                return self._watch(quote_snapshot)
            except Blocked as error:
                self.blocked('watch', error)
                raise

    def _watch(self, quote_snapshot):
        at = self.clock()
        _, opened, closed = session(at)
        if not opened <= at <= closed:
            raise Blocked('MARKET_CLOSED')
        quotes = validate_quotes(quote_snapshot, at, self.mode)
        state = self.ledger.state()
        if any(s not in quotes for s in state['positions']):
            raise Blocked('HELD_QUOTE_MISSING')
        snapshots = {r['body']['hash']: r['body']['snapshot'] for r in self.ledger.records('snapshot')}
        extractions = {r['body']['cycle']: r['body']['events'] for r in self.ledger.records('extraction')}
        latest = list(snapshots.values())[-1] if snapshots else None
        latest_news = list(extractions.values())[-1] if extractions else []
        ranks = ranking({r['symbol']: r for r in latest['instruments']}) if latest else {}
        events = []
        # Protective exits are deterministic and cannot be delayed by model availability.
        for symbol, pos in list(state['positions'].items()):
            q = quotes[symbol]
            if instant(q['at']) < instant(pos['filled_at']):
                raise Blocked('QUOTE_BEFORE_POSITION')
            reason = None
            days = holding_days(pos['filled_at'], at)
            negative = any(e['symbol'] == symbol and e['primary'] and e['sentiment'] == 'negative'
                           and fresh(e['published_at'], at, 3 * 86400) for e in latest_news)
            if q['bid'] <= pos['stop']:
                reason = 'STOP'
            elif days >= pos['holding_days']:
                reason = 'TIME_STOP'
            elif negative:
                reason = 'NEGATIVE_PRIMARY_NEWS'
            elif latest and pos['playbook'] == 'equity_momentum' and at.weekday() == 2:
                rank = ranks.get(symbol)
                if not rank or not rank['qualified'] or rank['rank'] > rank['keep_cut']:
                    reason = 'MOMENTUM_EXIT'
            if reason:
                price = q['bid'] * (1 - self.settings['slippage_per_side'])
                fee = price * pos['qty'] * self.settings['fee_per_side']
                pnl = (price - pos['entry']) * pos['qty'] - pos['entry_fee'] - fee
                body = {'symbol': symbol, 'plan_id': pos['plan_id'], 'qty': pos['qty'], 'exit': price,
                        'exit_fee': fee, 'closed_at': iso(at), 'quote_at': q['at'], 'quote_hash': digest(q),
                        'reason': reason, 'pnl': pnl, 'r': pnl / pos['initial_risk'],
                        'supervisor_plan_id': pos['plan_id'], 'authorization': 'REVIEWED_STANDING_EXIT'}
                events.append(('exit_fill', body))
                state['cash'] += price * pos['qty'] - fee
                del state['positions'][symbol]
        for identity, plan in list(state['pending'].items()):
            if at >= instant(plan['expires_at']):
                events.append(('expired', {'plan_id': identity, 'reason': 'EXPIRED'}))
                del state['pending'][identity]
                continue
            quote = quotes.get(plan['symbol'])
            if quote is None or instant(quote['at']) < instant(plan['approved_at']):
                continue
            snapshot = snapshots[plan['snapshot_hash']]
            news = extractions[plan['cycle']]
            del state['pending'][identity]
            sized, reason = size(plan, state, quotes, snapshot, news, self.settings, at)
            if reason:
                events.append(('fill_veto', {'plan_id': identity, 'reason': reason}))
                continue
            qty = min(plan['qty'], sized['qty'])
            price, fee = sized['entry'], sized['entry'] * qty * self.settings['fee_per_side']
            risk = sized['initial_risk'] / sized['qty'] * qty
            body = {**plan, 'qty': qty, 'entry': price, 'entry_fee': fee, 'initial_risk': risk,
                    'filled_at': iso(at), 'quote_at': quote['at'], 'quote_hash': digest(quote),
                    'best_close': price}
            events.append(('entry_fill', body))
            state['cash'] -= price * qty + fee
            state['positions'][plan['symbol']] = body
        nav = equity(state, quotes)
        heat, clusters, gross = exposure(state, quotes, self.settings)
        factor, governor_code = governor(state, nav, at)
        events.append(('mark', {'equity': nav, 'cash': state['cash'], 'heat': heat / nav if nav > 0 else None,
                                'clusters': clusters, 'gross': gross, 'governor': governor_code,
                                'quote_snapshot_hash': digest(quote_snapshot)}))
        self.ledger.append(events, at)
        return {'equity': nav, 'events': len(events), 'positions': len(state['positions'])}
