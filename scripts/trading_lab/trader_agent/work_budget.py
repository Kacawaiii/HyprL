"""Prospective, operator-approved work envelopes; existing grants remain frozen."""
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical
from .config import TraderError, instant, strict_json

APPROVAL_NAME = 'trader-agent-work-budget-amendment-v2.json'
SPEC_HASH = 'b823b3a56d1c2e2d03dc8358364a6ee6b61d6c681d8f4a99ec5973776964ed15'
REQUIRES_APPROVAL_FROM = '2026-10-27T00:00:00Z'
ROOT = Path(__file__).resolve().parents[3] / 'docs/artifacts'
STOP_ERRORS = frozenset({'WORK_BUDGET_EXHAUSTED', 'WORK_BUDGET_OVERRUN',
    'WORK_BUDGET_APPROVAL_REQUIRED', 'WORK_BUDGET_APPROVAL_INVALID', 'WORK_BUDGET_PROMPT_TOO_LARGE',
    'MODEL_USAGE_INVALID',
    'MODEL_PERMISSION_PROTOCOL_INVALID', 'WORK_BUDGET_SPEC_HASH_MISMATCH', 'WORK_BUDGET_UNACCOUNTED'})


def spec():
    value = strict_json((ROOT / 'trader_work_budget_spec_v2.json').read_text())
    if value.pop('canonical_sha256') != SPEC_HASH or sha256_canonical(value) != SPEC_HASH:
        raise TraderError('WORK_BUDGET_SPEC_HASH_MISMATCH')
    return value


def policy(grant, at):
    approval = getattr(grant, 'work_budget_approval', None)
    if approval is None:
        if at >= instant(REQUIRES_APPROVAL_FROM):
            raise TraderError('WORK_BUDGET_APPROVAL_REQUIRED')
        return None  # frozen historical replay, before the prospective cutover
    value = spec()
    if not isinstance(approval, dict) or any(not isinstance(approval.get(key), str)
            for key in ('approved_at', 'effective_from', 'not_after')):
        raise TraderError('WORK_BUDGET_APPROVAL_INVALID')
    try:
        parent_expiry = grant.payload['not_after']
        if instant(approval['not_after']) > instant(parent_expiry):
            parent_expiry = grant.extension_metadata(at, slot='am', prospective=True)['authorization_not_after']
        valid = (approval['schema'] == 'trader-agent-work-budget-amendment-v2'
            and approval['status'] == 'APPROVED'
            and approval['spec_hash'] == SPEC_HASH
            and approval['parent_preregistration_hash'] == value['base_preregistration_hash']
            and approval['base_grant_hash'] == (grant.base_identity or grant.identity)
            and instant(approval['approved_at']) <= instant(approval['effective_from']) <= at
            < instant(approval['not_after']) <= instant(parent_expiry))
    except (KeyError, ValueError, TypeError):
        valid = False
    if not valid:
        raise TraderError('WORK_BUDGET_APPROVAL_INVALID')
    return value


def charge(ledger, kind, at):
    """Called again inside the dispatch transaction, including on direct reservations."""
    probe = kind.startswith('rehearsal_')
    role = kind.removeprefix('rehearsal_')
    if role not in {'analyst_claude', 'reviewer'}:
        return None
    # A reservation alone cannot dispatch a model. Historical/synthetic providers
    # remain usable at arbitrary synthetic dates; ModelRunner.once requires the
    # prospective approval before launching any real CLI.
    if ledger.grant.work_budget_approval is None:
        return None
    value = policy(ledger.grant, at)
    if value is None:
        return None
    budget = value['admission']
    return {'arm': 'monitor' if probe else ledger.arm, 'role': role,
            'tokens': budget['probe_per_call' if probe else 'per_call'],
            'arm_cap': budget['probe_per_slot' if probe else 'per_arm'],
            'role_cap': budget['probe_per_call'] * 2 if probe else budget['per_call'] * 2,
            'slot_cap': budget['per_slot'], 'spec_hash': SPEC_HASH,
            'approval_hash': sha256_canonical(ledger.grant.work_budget_approval)}


def usage(event):
    """The final per-model totals include helpers. Never add aggregate usage a second time."""
    models = event.get('modelUsage')
    if not isinstance(models, dict) or not models:
        raise TraderError('MODEL_USAGE_INVALID')
    new, cached = 0, 0
    for model in models.values():
        if not isinstance(model, dict):
            raise TraderError('MODEL_USAGE_INVALID')
        amounts = []
        for key in ('inputTokens', 'cacheCreationInputTokens', 'outputTokens', 'cacheReadInputTokens'):
            number = model.get(key)
            if type(number) is not int or number < 0:
                raise TraderError('MODEL_USAGE_INVALID')
            amounts.append(number)
        new += sum(amounts[:3])
        cached += amounts[3]
    return new, cached
