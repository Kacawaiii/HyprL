"""Expiry-only operator records. Identities and source documents remain unchanged.

The records bind byte SHA256s, not canonical JSON hashes. This module has no
network dependencies and is also usable by standalone acquisition tools.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

TRADER_RECORD = 'trader-v1-extension-2026-10-26.json'
TRADER_RECORDS = (TRADER_RECORD, 'trader-v1-extension-2-2026-10-31.json')
SEC_RECORD = 'sec-user-agent-extension-2026-10-26.json'
SEC_SCOPE = 'https://www.sec.gov'


class ExtensionError(ValueError):
    pass


def instant(value):
    at = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if at.tzinfo is None:
        raise ExtensionError('INVALID_EXTENSION_TIME')
    return at.astimezone(timezone.utc)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ExtensionError('DUPLICATE_EXTENSION_KEY')
            result[key] = value
        return result

    def constant(_):
        raise ExtensionError('NONFINITE_EXTENSION_JSON')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def expiry_receipt(path, payload, at, *, slot=None, scope=None, prospective=False):
    """Return separate extension evidence only when the original grant expired.

    Before expiry the old window is authoritative, even if no extension exists.
    Every extended use rechecks all referenced bytes; no cached approval survives
    deletion, replacement or a changed scope. Prospective approval caps require
    this same evidence before original expiry; the decision must already exist
    at the actual approval-check time. Errors never include private data.
    """
    try:
        if at < instant(payload['granted_at']):
            raise ExtensionError('AUTHORIZATION_NOT_STARTED')
        if at < instant(payload['not_after']) and not prospective:
            return {}
        if path is None or (slot is None) == (scope is None):
            raise ExtensionError('EXTENSION_CONTEXT_REQUIRED')
        path = Path(path).expanduser().resolve(strict=True)
        trader = slot is not None
        records = [path.parent / name for name in (TRADER_RECORDS if trader else (SEC_RECORD,))
                   if (path.parent / name).exists()]
        matches = []
        for record in records:
            receipt = _record_receipt(record, path, payload, at, slot=slot, scope=scope)
            if receipt is not None:
                matches.append(receipt)
        if len(matches) != 1:
            raise ExtensionError('EXTENSION_REFERENCE_MISSING_OR_AMBIGUOUS')
        return matches[0]
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        raise ExtensionError('AUTHORIZATION_EXTENSION_UNAVAILABLE_OR_INVALID') from None


def _record_receipt(record, path, payload, at, *, slot, scope):
    """Validate a whole named record; independent grants never imply renewal."""
    trader = slot is not None
    try:
        raw = record.read_bytes()
        value = strict_json(raw)
        keys = {'schema', 'decided_at', 'decided_by', 'decision', 'extends'}
        if trader:
            keys.add('slots')
        if (not isinstance(value, dict) or set(value) != keys
                or value['schema'] != ('trader-authorization-extension-v1' if trader else 'authorization-extension-v1')
                or any(not isinstance(value[k], str) or not value[k].strip() for k in ('decided_by', 'decision'))
                or instant(value['decided_at']) > at
                or not isinstance(value['extends'], list) or not value['extends']):
            raise ExtensionError('INVALID_EXTENSION_SCHEMA')
        if trader and (value['slots'] != ['am'] or slot not in value['slots']):
            raise ExtensionError('EXTENSION_SLOT_NOT_AUTHORIZED')
        if not trader and scope != SEC_SCOPE:
            raise ExtensionError('EXTENSION_SCOPE_NOT_AUTHORIZED')
        seen, match = set(), None
        for entry in value['extends']:
            entry_keys = {'file', 'sha256', 'not_after_from', 'not_after_to'}
            if not trader:
                entry_keys.add('scope_keys')
            if (not isinstance(entry, dict) or set(entry) != entry_keys
                    or not isinstance(entry['file'], str) or not Path(entry['file']).is_absolute()
                    or not isinstance(entry['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', entry['sha256'])
                    or not trader and entry['scope_keys'] != [SEC_SCOPE]):
                raise ExtensionError('INVALID_EXTENSION_ENTRY')
            target = Path(entry['file']).resolve(strict=True)
            if target in seen or target.samefile(record):
                raise ExtensionError('EXTENSION_SELF_REFERENCE_OR_DUPLICATE')
            seen.add(target)
            original = target.read_bytes()
            source = strict_json(original)
            if (sha256(original).hexdigest() != entry['sha256']
                    or source['not_after'] != entry['not_after_from']
                    or instant(entry['not_after_to']) <= instant(entry['not_after_from'])
                    or instant(value['decided_at']) >= instant(entry['not_after_to'])
                    or not trader and SEC_SCOPE not in source['scope']):
                raise ExtensionError('EXTENSION_BINDING_MISMATCH')
            if target == path:
                if source != payload or not at < instant(entry['not_after_to']):
                    raise ExtensionError('EXTENSION_BINDING_OR_WINDOW_MISMATCH')
                match = {'authorization_file_sha256': entry['sha256'],
                         'authorization_extension_sha256': sha256(raw).hexdigest(),
                         'authorization_not_after': entry['not_after_to']}
        return match
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        raise ExtensionError('AUTHORIZATION_EXTENSION_UNAVAILABLE_OR_INVALID') from None


def sec_user_agent(path, at):
    """Reuse the SEC contact grant, never its endpoint permissions or budgets."""
    try:
        payload = strict_json(Path(path).read_bytes())
        if (payload['authorization'] != 'news-radar-sec-fed-telegram-v1'
                or payload['extends'] != 'news-radar-v1' or payload['operator_signed'] is not True
                or instant(payload['not_after']) <= instant(payload['granted_at'])):
            raise ExtensionError('INVALID_SEC_AUTHORIZATION')
        metadata = expiry_receipt(path, payload, at, scope=SEC_SCOPE)
        ua = payload['scope'][SEC_SCOPE]['user_agent']
        if not isinstance(ua, str) or not ua.strip() or any(ord(c) < 32 for c in ua):
            raise ExtensionError('INVALID_SEC_USER_AGENT')
        identity = sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        return ua, {'authorization_hash': identity, **metadata}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        raise ExtensionError('SEC_USER_AGENT_AUTHORIZATION_UNAVAILABLE_OR_EXPIRED') from None
