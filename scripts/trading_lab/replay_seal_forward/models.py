"""Serial no-tool dispatch, 150-call nightly caps per model, durable no-refund intents.

Opus rides the subscription by default (150/night, and every Opus call of the UTC day stays inside
the 150/day replay grant, keeping one call for the same-day forward session). An optional API-key
path for Opus (grant replay-bench-api-v1) exists but is off unless explicitly enabled; a
"Credit balance is too low" answer is a durable hard stop in either mode.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.trader_agent.config import TraderError, strict_json
from scripts.trading_lab.trader_agent.runners import gpt_events
from scripts.trading_lab.replay_bench.grant import Budget, ReplayError, instant, iso, now
from scripts.trading_lab.replay_bench.models import argv as claude_argv, night_remaining, QUOTA
from scripts.trading_lab.replay_bench.storage import atomic_json, append_event
from . import schemas, spec


NIGHT_CAP = 150
OPUS_DAY_KEYS = ('model:opus', 'forward:model:opus', 'probe:opus')
CREDIT_STOP = 'api-credit-stop.json'


def forward_session(at):
    """True when this UTC date carries a forward decision (its 09:05 New York falls on it)."""
    from scripts.trading_lab.replay_bench import marketdata as md
    day = iso(at)[:10]
    return day >= spec.spec()['forward']['first_session'] and bool(md.sessions(day, day))


class DuelBudget(Budget):
    """Share the legacy budget file, but each subscription has its own ceiling and stop."""
    def reserve_model(self, family, limit, at, night_limit=NIGHT_CAP):
        if family not in ('opus', 'gpt'):
            raise ReplayError('MODEL_NOT_ALLOWED')
        if (self.path.parent / CREDIT_STOP).exists() and family == 'opus':
            raise ReplayError('API_CREDIT_EXHAUSTED')
        with self._locked() as data:
            day = data.setdefault(iso(at)[:10], {})
            if day.get('stopped:' + family) or family == 'opus' and day.get('stopped'):
                raise ReplayError('MODEL_USAGE_LIMIT')
            key = 'model:' + family
            if day.get(key, 0) >= min(limit, night_limit, NIGHT_CAP):
                raise ReplayError('BUDGET_EXHAUSTED')
            if family == 'gpt' and day.get(key, 0) + day.get('probe:gpt', 0) >= min(limit, night_limit, NIGHT_CAP):
                raise ReplayError('BUDGET_EXHAUSTED')
            if family == 'opus':
                # Inside the 150/day grant across night, forward and probe; keep the forward call.
                reserve = 1 if forward_session(at) and not day.get('forward:model:opus') else 0
                if sum(day.get(k, 0) for k in OPUS_DAY_KEYS) + reserve >= limit:
                    raise ReplayError('BUDGET_EXHAUSTED')
            day[key] = day.get(key, 0) + 1

    def credit_stop(self, at):
        atomic_json(self.path.parent / CREDIT_STOP, {'state': 'API_CREDIT_EXHAUSTED', 'at': iso(at),
                                                     'note': 'hard stop; the operator removes this file'})
        self.stop_model('opus', at)

    def reserve_probe(self, family, limit, at, *, authorized_retry=False):
        if family not in ('opus', 'gpt'):
            raise ReplayError('MODEL_NOT_ALLOWED')
        if family == 'opus' and (self.path.parent / CREDIT_STOP).exists():
            raise ReplayError('API_CREDIT_EXHAUSTED')
        with self._locked() as data:
            day = data.setdefault(iso(at)[:10], {})
            if day.get('stopped:' + family) or family == 'opus' and day.get('stopped'):
                raise ReplayError('MODEL_USAGE_LIMIT')
            used = day.get('probe:' + family, 0)
            if used >= (2 if authorized_retry else 1):
                raise ReplayError('PROBE4_ALREADY_ASKED')
            if family == 'opus' and sum(day.get(k, 0) for k in OPUS_DAY_KEYS) >= limit:
                raise ReplayError('BUDGET_EXHAUSTED')
            if family == 'gpt' and day.get('model:gpt', 0) + day.get('probe:gpt', 0) >= limit:
                raise ReplayError('BUDGET_EXHAUSTED')
            day['probe:' + family] = used + 1

    def stop_model(self, family, at):
        with self._locked() as data:
            day = data.setdefault(iso(at)[:10], {})
            day['stopped:' + family] = 'MODEL_USAGE_LIMIT'
            # The older Opus dispatcher must also see the subscription stop.
            if family == 'opus':
                day['stopped'] = 'MODEL_USAGE_LIMIT'

    def model_stopped(self, family, at):
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        day = data.get(iso(at)[:10], {})
        return day.get('stopped:' + family) or (day.get('stopped') if family == 'opus' else None)


def argv(family, schema, schema_file):
    identities = spec.spec()['models']['identities']
    if family == 'opus':
        args = claude_argv(identities[family], schema)
        # Match the neutral system instruction sent to Codex via its developer override.
        args[args.index('--system-prompt') + 1] = SYSTEM
        return args
    if family != 'gpt':
        raise ReplayError('MODEL_NOT_ALLOWED')
    args = ['codex', '--no-daemon', 'exec', '--strict-config', '--ignore-user-config', '--ignore-rules', '--ephemeral',
            '--sandbox', 'read-only', '--skip-git-repo-check', '--model', identities[family],
            '-c', 'web_search="disabled"', '-c', 'project_doc_max_bytes=0',
            '-c', 'suppress_unstable_features_warning=true',
            '-c', 'developer_instructions=' + json.dumps(SYSTEM),
            '-c', 'model_reasoning_effort="medium"',
            '-c', 'model_provider="openai"', '-c', 'forced_login_method="chatgpt"']
    for feature in ('shell_tool', 'unified_exec', 'code_mode', 'code_mode_only', 'code_mode_host', 'multi_agent', 'multi_agent_v2', 'apps', 'plugins',
                    'hooks', 'browser_use', 'computer_use', 'image_generation', 'skill_search',
                    'view_image', 'sleep_tool', 'tool_suggest', 'memories', 'goals',
                    'unbounded_connection_retries', 'remote_plugin'):
        args += ['--disable', feature]
    return args + ['--enable', 'skip_host_skill_discovery', '--output-schema', str(schema_file), '--json', '-']


SYSTEM = 'Answer only the supplied research question using the supplied data. No tools.'


def require_systemd(runner):
    """Production model dispatch must inherit an agent user-unit invocation."""
    if runner is subprocess.run and not os.environ.get('INVOCATION_ID'):
        raise ReplayError('MODEL_SYSTEMD_REQUIRED')


def no_tools(event):
    """Fail closed on every non-message Codex item, even on error or cached response."""
    if event.get('type', '').startswith('item.'):
        if event.get('item', {}).get('type') not in ('agent_message', 'reasoning'):
            raise ReplayError('TAINTED_RUN')
    for key in ('type', 'name', 'tool_name'):
        name = str(event.get(key, '')).lower()
        if any(x in name for x in ('tool_use', 'tool_call', 'web_search', 'web_fetch', 'command_execution',
                                   'file_change', 'function_call', 'exec_command', 'apply_patch')):
            raise ReplayError('TAINTED_RUN')
    if event.get('permission_denials'):
        raise ReplayError('TAINTED_RUN')
    for value in event.values():
        if isinstance(value, dict):
            no_tools(value)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, dict):
                    no_tools(child)


def decode(raw, err, returncode, family, schema):
    try:
        events = gpt_events(raw) if family == 'gpt' else [strict_json(raw)]
    except (TraderError, ValueError, TypeError):
        events = []
    errors = err + ''.join(json.dumps(e) for e in events if e.get('type') in ('error', 'turn.failed') or
                          e.get('is_error') or e.get('item', {}).get('type') == 'error')
    if not events:
        errors += raw
    credit = family == 'opus' and ('credit balance is too low' in errors.lower() or 'credit balance is too low' in raw.lower())
    quota = any(x in errors.lower() for x in (*QUOTA, '429', 'quota exceeded', 'codex limit'))
    for event in events:
        try:
            no_tools(event)
        except ReplayError as error:
            # Reject tainted output AND retain a quota/credit stop. A tainted
            # envelope cannot erase the provider's stop for a later dispatch.
            raise ReplayError(error.code, {'usage_limit': quota, 'credit_stop': credit}) from None
    if credit:
        raise ReplayError('API_CREDIT_EXHAUSTED')
    if quota:
        raise ReplayError('MODEL_USAGE_LIMIT')
    if returncode or not events:
        raise ReplayError('MODEL_FAILED' if returncode else 'MODEL_JSON_INVALID')
    identity = spec.spec()['models']['identities'][family]
    if family == 'opus':
        event = events[0]
        if event.get('is_error') or event.get('subtype', 'success') != 'success':
            raise ReplayError('MODEL_FAILED')
        reported = sorted(event.get('modelUsage', {}))
        if reported != [identity]:
            raise ReplayError('MODEL_IDENTITY_CHANGED')
        output = event.get('structured_output')
    else:
        if any(e.get('type') in ('error', 'turn.failed') for e in events):
            raise ReplayError('MODEL_FAILED')
        if sum(e.get('type') == 'thread.started' for e in events) != 1 or sum(e.get('type') == 'turn.completed' for e in events) != 1:
            raise ReplayError('MODEL_ENVELOPE_INVALID')
        # This CLI does not normally emit server model identity. Never invent an attestation.
        reported = sorted({e['model'] for e in events if isinstance(e.get('model'), str)})
        if reported and reported != [identity]:
            raise ReplayError('MODEL_IDENTITY_CHANGED')
        messages = [e['item']['text'] for e in events if e.get('type') == 'item.completed'
                    and e.get('item', {}).get('type') == 'agent_message']
        try:
            output = strict_json(messages[-1])
        except (IndexError, TypeError, ValueError):
            raise ReplayError('MODEL_JSON_INVALID') from None
    schemas.validate(schema, output)
    return output, reported


class ApiBilling:
    """Optional Opus API-key billing (grant replay-bench-api-v1). Off unless REPLAY_OPUS_BILLING=api.

    The key is read from the grant's key file at dispatch only (never logged). Spend is summed from
    total_cost_usd in a durable ledger before every call: $60 for the replay, $10 per month forward.
    """
    def __init__(self, grant_file, ledger_file, *, clock=now):
        self.grant_file = Path(grant_file)
        self.grant = json.loads(Path(grant_file).read_text())
        if self.grant.get('id') != 'replay-bench-api-v1':
            raise ReplayError('API_GRANT_INVALID')
        if sha256_canonical(self.grant) != spec.spec()['data_collection']['api_grant_sha256']:
            raise ReplayError('API_GRANT_HASH_MISMATCH')
        self.ledger, self.clock = Path(ledger_file), clock
        if not self.clock() < instant(self.grant['expires'] + 'T23:59:59Z'):
            raise ReplayError('API_GRANT_EXPIRED')

    def key(self):
        file = Path(self.grant['claude_api']['key_file'])
        if not file.is_file() or file.stat().st_mode & 0o077:
            raise ReplayError('API_KEY_FILE_MISSING')
        values = dict(l.split('=', 1) for l in file.read_text().splitlines() if '=' in l and not l.startswith('#'))
        if not values.get('ANTHROPIC_API_KEY', '').strip():
            raise ReplayError('API_KEY_FILE_MISSING')
        return values['ANTHROPIC_API_KEY'].strip().strip('"')

    def spent(self, scope, month=None):
        rows = json.loads(self.ledger.read_text()) if self.ledger.is_file() else []
        return sum(r.get('cost_usd', 0) for r in rows if r['scope'] == scope and (month is None or r['at'][:7] == month))

    def check(self, scope):
        if scope not in ('replay', 'forward'):
            raise ReplayError('API_SCOPE_INVALID')
        if json.loads(self.grant_file.read_text()) != self.grant:
            raise ReplayError('API_GRANT_CHANGED')
        if self.clock() >= instant(self.grant['expires'] + 'T23:59:59Z'):
            raise ReplayError('API_GRANT_EXPIRED')
        rows = json.loads(self.ledger.read_text()) if self.ledger.is_file() else []
        if any(r.get('state') in ('INTENT', 'UNKNOWN') for r in rows):
            raise ReplayError('API_SPEND_UNKNOWN')
        cap = self.grant['claude_api']['replay_cap_usd' if scope == 'replay' else 'forward_duel_cap_usd_per_month']
        remaining = cap - self.spent(scope, None if scope == 'replay' else iso(self.clock())[:7])
        if remaining <= 0:
            raise ReplayError('API_DOLLAR_CAP_REACHED')
        return remaining

    def begin(self, scope, identity):
        remaining = self.check(scope)
        rows = json.loads(self.ledger.read_text()) if self.ledger.is_file() else []
        rows.append({'scope': scope, 'identity': identity, 'state': 'INTENT', 'at': iso(self.clock())})
        atomic_json(self.ledger, rows)
        return remaining

    def record(self, scope, identity, raw, *, credit_stop=False):
        try:
            cost = float(json.loads(raw)['total_cost_usd'])
            if not math.isfinite(cost) or cost < 0:
                raise ValueError
        except (ValueError, TypeError, KeyError, AttributeError):
            cost = 0.0 if credit_stop else None
        rows = json.loads(self.ledger.read_text()) if self.ledger.is_file() else []
        row = next(r for r in rows if r.get('identity') == identity and r['scope'] == scope)
        row.update(state='KNOWN' if cost is not None else 'UNKNOWN')
        if cost is not None:
            row['cost_usd'] = cost
        atomic_json(self.ledger, rows)
        if cost is None:
            raise ReplayError('API_SPEND_UNKNOWN')


def api_billing(root):
    if os.environ.get('REPLAY_OPUS_BILLING', 'subscription') != 'api':
        return None
    grant = os.environ.get('REPLAY_API_GRANT')
    if not grant:
        raise ReplayError('API_GRANT_NOT_CONFIGURED')
    return ApiBilling(grant, Path(root) / 'api-ledger.json')


class Models:
    def __init__(self, grant, budget, transcripts, *, runner=subprocess.run, clock=now, window=night_remaining,
                 billing=None, scope='replay'):
        self.grant, self.budget, self.runner, self.clock = grant, budget, runner, clock
        self.window = window
        self.transcripts = Path(transcripts)
        self.billing, self.scope = billing, scope

    def call(self, family, prompt, schema, *, tag, timeout=900):
        require_systemd(self.runner)
        if family not in spec.spec()['models']['identities']:
            raise ReplayError('MODEL_NOT_ALLOWED')
        # Bind code/spec/schema: no old decisions can be silently relabeled under a new protocol.
        identity = sha256_canonical([spec.SPEC_HASH, spec.code_hash(), family, prompt, schema, tag])
        path = self.transcripts / (identity + '.json')
        saved = json.loads(path.read_text()) if path.exists() else None
        if saved and saved['state'] == 'INTENT':
            raise ReplayError('CALL_OUTCOME_UNKNOWN', identity)
        if saved and saved['state'] == 'RECEIVED':
            try:
                return self._decode(saved, family, prompt, schema, identity)
            except ReplayError as error:
                if error.code != 'MODEL_USAGE_LIMIT':
                    raise
                saved['state'] = 'USAGE_LIMIT'
                atomic_json(path, saved)
                self.budget.stop_model(family, instant(saved['at']))
        at = self.clock()
        remaining = self.window(at)
        self.grant.check()
        # GPT research authorization is the operator's new task, pinned in the new spec;
        # the original HTTP grant is unchanged and still retires GPT in the live engine.
        limit = min(NIGHT_CAP, self.grant.model_limit('opus')) if family == 'opus' else NIGHT_CAP
        api = self.billing is not None and family == 'opus'
        if api:
            self.billing.check(self.scope)
            key = self.billing.key()
        self.budget.reserve_model(family, limit, at)
        if api:
            api_remaining = self.billing.begin(self.scope, identity)
        record = {'state': 'INTENT', 'family': family, 'at': iso(at), 'tag': tag,
                  'spec_hash': spec.SPEC_HASH, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        atomic_json(path, record)
        append_event(self.transcripts / 'calls.jsonl', dict(record, identity=identity))
        with tempfile.TemporaryDirectory(prefix='replay-duel-') as private:
            private = Path(private)
            cwd = private / 'empty'
            cwd.mkdir()
            schema_file = private / 'schema.json'
            atomic_json(schema_file, schema)
            env = {k: v for k, v in os.environ.items() if k not in (
                'CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT', 'CODEX_THREAD_ID', 'OPENAI_API_KEY',
                'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'OPENAI_BASE_URL', 'ANTHROPIC_BASE_URL')}
            env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY='1', CLAUDE_CODE_DISABLE_CLAUDE_MDS='1', CLAUDE_CODE_SIMPLE='1')
            if api:
                env['ANTHROPIC_API_KEY'] = key
            try:
                args = argv(family, schema, schema_file)
                if api:
                    args += ['--max-budget-usd', str(api_remaining)]
                result = self.runner(args, input=prompt, text=True, capture_output=True,
                                     cwd=str(cwd), env=env, timeout=min(timeout, remaining - 1))
            except (subprocess.TimeoutExpired, OSError):
                raise ReplayError('CALL_OUTCOME_UNKNOWN', identity) from None
        record.update(state='RECEIVED', raw=result.stdout or '', err=result.stderr or '', returncode=result.returncode,
                      billing='api' if api else 'subscription')
        atomic_json(path, record)
        try:
            value = self._decode(record, family, prompt, schema, identity)
        except ReplayError as error:
            if api:
                try:
                    self.billing.record(self.scope, identity, record['raw'], credit_stop=error.code == 'API_CREDIT_EXHAUSTED')
                except ReplayError:
                    pass
            raise
        if api:
            self.billing.record(self.scope, identity, record['raw'])
        return value

    def _decode(self, record, family, prompt, schema, identity):
        try:
            output, reported = decode(record['raw'], record['err'], record['returncode'], family, schema)
        except ReplayError as error:
            flags = error.evidence if isinstance(error.evidence, dict) else {}
            if error.code == 'API_CREDIT_EXHAUSTED' or flags.get('credit_stop'):
                record['state'] = 'CREDIT_STOP'
                atomic_json(self.transcripts / (identity + '.json'), record)
                self.budget.credit_stop(instant(record['at']))
            elif error.code == 'MODEL_USAGE_LIMIT' or flags.get('usage_limit'):
                record['state'] = 'USAGE_LIMIT'
                atomic_json(self.transcripts / (identity + '.json'), record)
                self.budget.stop_model(family, instant(record['at']))
            raise
        return output, {'family': family, 'model': spec.spec()['models']['identities'][family], 'reported': reported,
                        'identity_evidence': 'CLI_REQUEST_ONLY' if not reported else 'CLI_REPORTED',
                        'called_at': record['at'], 'transcript': identity,
                        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
