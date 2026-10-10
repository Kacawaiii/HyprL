"""Bounded French digests; private credentials and at-most-once delivery."""
from datetime import timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener
import fcntl
import json
import os
import re

from .core import NoRedirect, RadarError, digest, instant, iso
from .digest import safe


def compact(value, maximum):
    value = safe(value)
    if len(value.encode('utf-16-le')) // 2 <= maximum:
        return value
    value = value.encode('utf-16-le')[:(maximum - 1) * 2].decode('utf-16-le', errors='ignore')
    return value + '…'


def event_lines(event, index=None):
    scenario = event.get('scenario') or {}
    title = scenario.get('summary') or event['headline']
    mechanisms = scenario.get('exposures') or event.get('transmission_hypotheses') or []
    mechanism = mechanisms[0]['mechanism'] if mechanisms else 'Métadonnées seules : effet économique non évalué.'
    measured = event.get('priced_in', {}).get('measurements', [])
    priced = '; '.join(f"{r['symbol']} {r['move_atr']:+.2f} ATR" for r in measured[:2]) if measured else 'inconnu : données événementielles insuffisantes'
    return [f"{str(index) + '. ' if index else ''}{compact(title, 115)} ({event['importance']}/100)",
            'Mécanisme: ' + compact(mechanism, 155),
            'Déjà intégré: ' + compact(priced, 100) + '; causalité non établie.']


def short_digest(report, full_path):
    rows = []
    for label in ('SPY', 'QQQ', '10y_yield', 'dollar_index', 'BTC', 'VIX'):
        row = report.get('regime', {}).get(label, {})
        change = row.get('returns_pct', {}).get('1d')
        rows.append(label + ': ' + (f'{change:+.2f}% (1j)' if change is not None else 'inconnu') +
                    (' périmé' if row.get('status') == 'STALE' else ''))
    slot = {'morning': 'matin', 'evening': 'soir', 'manual': 'validation'}.get(report['slot'], 'validation')
    state = {'READY': 'prêt', 'PARTIAL': 'partiel', 'BLOCKED': 'bloqué', 'SYNTHETIC': 'synthétique'}.get(report['status'], 'inconnu')
    lines = [f"Radar {slot} — {report['date']} — {state}", 'Régime: ' + '; '.join(rows)]
    events = sorted(report['events'], key=lambda e: (-e['importance'], -e['evidence']['score'], e['id']))[:5]
    for index, event in enumerate(events, 1):
        lines.extend(event_lines(event, index))
    if not events:
        lines.append('Aucun événement disponible.')
    anomalies = []
    for row in report.get('anomalies', [])[:3]:
        links = row.get('news_links', [])
        explanation = ('lien temporel avec ' + compact(links[0]['headline'], 90)) if links else 'aucune nouvelle trouvée'
        anomalies.append(f"{row['symbol']} {row['move_atr']:+.2f} ATR : {explanation}")
    lines.append('Anomalies: ' + ('; '.join(anomalies) if anomalies else 'aucune détectée dans les données disponibles') + '; causalité non établie.')
    lines.append('Consensus inconnu; volumes IEX partiels; mécanismes conditionnels.')
    # A private host path is deliberately not exposed through a public web server.
    footer = 'Rapport complet: file://' + str(full_path)
    message = '\n'.join(lines + [footer])
    if len(message.encode('utf-16-le')) // 2 > 3500:
        raise RadarError('TELEGRAM_DIGEST_TOO_LONG')
    return message


def post_message(url, payload):
    try:
        request = Request(url, data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        with build_opener(NoRedirect()).open(request, timeout=20) as response:
            raw = response.read(100_001)
            if len(raw) > 100_000:
                raise RadarError('TELEGRAM_RESPONSE_TOO_LARGE')
            return response.status, raw
    except HTTPError as error:
        # Never expose the token-bearing request URL or Telegram response text.
        return error.code, b''
    except Exception:
        raise RadarError('TELEGRAM_TRANSPORT_UNCONFIRMED') from None


class TelegramDelivery:
    def __init__(self, store, grant, *, credentials=None, post=post_message):
        self.store, self.grant, self.post = store, grant, post
        rule = grant.payload['scope']['https://api.telegram.org']
        self.credentials = credentials or rule.get('credentials')

    def secrets(self):
        try:
            values = {}
            for line in Path(self.credentials).expanduser().read_text().splitlines():
                if '=' in line and not line.lstrip().startswith('#'):
                    key, value = line.removeprefix('export ').split('=', 1)
                    values[key.strip()] = value.strip().strip('\"\'')
            token, recipient = values['TELEGRAM_BOT_TOKEN'], values['ADMIN_TELEGRAM_ID']
            if not re.fullmatch(r'\d+:[A-Za-z0-9_-]+', token) or not re.fullmatch(r'[1-9]\d*', recipient):
                raise ValueError()
            return token, recipient
        except (OSError, ValueError, KeyError, TypeError):
            raise RadarError('TELEGRAM_CREDENTIALS_MISSING_OR_INVALID') from None

    def send(self, key, message, *, reserve_digests=False, digest_slot=None):
        self.grant.check(self.store.clock())
        self.grant.permit_delivery()
        if not isinstance(message, str) or not message.strip() or len(message.encode('utf-16-le')) // 2 > 3500:
            raise RadarError('TELEGRAM_DIGEST_TOO_LONG')
        key = digest(key)
        with (self.store.root / 'telegram.lock').open('a') as lock:
            os.chmod(lock.name, 0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RadarError('TELEGRAM_DELIVERY_BUSY') from None
            previous = self.store.latest('delivery', key)
            if previous:
                return {'status': 'DEDUPLICATED', 'previous_status': previous['status']}
            token, recipient = self.secrets()
            maximum = self.grant.payload['budgets']['telegram']
            if reserve_digests:
                # Reserve capacity for both scheduled digests within the same
                # twelve-attempt ceiling; do not let a burst starve the evening.
                today = self.store.clock().date().isoformat()
                delivered_slots = {r.get('digest_slot') for r in self.store.rows('delivery')
                                   if r.get('day') == today and r.get('digest_slot')}
                maximum -= len({'morning', 'evening'} - delivered_slots)
            self.store.reserve(self.grant, 'telegram', 'operator', maximum)
            attempt = {'status': 'UNCONFIRMED', 'at': iso(self.store.clock()), 'day': self.store.clock().date().isoformat(),
                       'message_hash': digest(message), 'characters_utf16': len(message.encode('utf-16-le')) // 2}
            if digest_slot:
                attempt['digest_slot'] = digest_slot
            # Write before dispatch: a crash or lost response cannot trigger a
            # blind retry that sends a second operator notification.
            self.store.append('delivery', key, attempt)
            try:
                status, raw = self.post('https://api.telegram.org/bot' + token + '/sendMessage',
                                        {'chat_id': recipient, 'text': message, 'disable_web_page_preview': True})
                response = json.loads(raw)
                if status == 200 and response.get('ok') is True and type(response['result']['message_id']) is int:
                    attempt = {**attempt, 'status': 'SENT', 'message_id': response['result']['message_id']}
                else:
                    attempt = {**attempt, 'status': 'FAILED', 'reason': 'TELEGRAM_REJECTED'}
            except Exception:
                attempt = {**attempt, 'status': 'UNCONFIRMED', 'reason': 'TELEGRAM_TRANSPORT_OR_RESPONSE_UNCONFIRMED'}
            self.store.append('delivery', key, attempt)
            return attempt

    def digest(self, report, full_path):
        return self.send('digest:' + report['date'] + ':' + report['slot'], short_digest(report, full_path), digest_slot=report['slot'])

    def alerts(self, events, full_path):
        results = []
        at = self.store.clock()
        for event in sorted(events, key=lambda e: (-e['importance'], e['id'])):
            if event['importance'] < 80:
                continue
            times = [s.get('published_at') or s.get('filing', {}).get('updated_at') for s in event['stories']]
            if not any(t and at - timedelta(days=1) <= instant(t) <= at for t in times):
                continue  # initial archive/backlog is not an immediate new alert
            message = '\n'.join(['ALERTE Radar — événement important', *event_lines(event),
                                 'Source: ' + event['stories'][0]['url'], 'Rapport complet: file://' + str(full_path)])
            try:
                results.append(self.send('alert:' + event['id'], message, reserve_digests=True))
            except RadarError as error:
                results.append({'status': 'BLOCKED', 'reason': str(error)})
                break
        return results
