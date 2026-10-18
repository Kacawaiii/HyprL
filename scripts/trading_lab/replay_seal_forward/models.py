"""Serial no-tool subscription dispatch, separate 60-call caps, durable no-refund intents."""
import hashlib
import json
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


class DuelBudget(Budget):
    """Share the legacy budget file, but each subscription has its own ceiling and stop."""
    def reserve_model(self, family, limit, at, night_limit=60):
        if family not in ('opus', 'gpt'):
            raise ReplayError('MODEL_NOT_ALLOWED')
        with self._locked() as data:
            day = data.setdefault(iso(at)[:10], {})
            if day.get('stopped:' + family) or family == 'opus' and day.get('stopped'):
                raise ReplayError('MODEL_USAGE_LIMIT')
            key = 'model:' + family
            if day.get(key, 0) >= min(limit, night_limit, 60):
                raise ReplayError('BUDGET_EXHAUSTED')
            day[key] = day.get(key, 0) + 1

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
            '-c', 'developer_instructions=' + json.dumps(SYSTEM),
            '-c', 'model_reasoning_effort="medium"',
            '-c', 'model_provider="openai"', '-c', 'forced_login_method="chatgpt"']
    for feature in ('shell_tool', 'code_mode_host', 'multi_agent', 'multi_agent_v2', 'apps', 'plugins',
                    'hooks', 'browser_use', 'computer_use', 'image_generation', 'skill_search',
                    'view_image', 'sleep_tool', 'tool_suggest', 'memories', 'goals',
                    'unbounded_connection_retries', 'remote_plugin'):
        args += ['--disable', feature]
    return args + ['--enable', 'skip_host_skill_discovery', '--output-schema', str(schema_file), '--json', '-']


SYSTEM = 'Answer only the supplied research question using the supplied data. No tools.'


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
    for event in events:
        no_tools(event)
    errors = err + ''.join(json.dumps(e) for e in events if e.get('type') in ('error', 'turn.failed') or e.get('is_error'))
    if not events:
        errors += raw
    if any(x in errors.lower() for x in (*QUOTA, '429', 'quota exceeded', 'codex limit')):
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


class Models:
    def __init__(self, grant, budget, transcripts, *, runner=subprocess.run, clock=now, window=night_remaining):
        self.grant, self.budget, self.runner, self.clock = grant, budget, runner, clock
        self.window = window
        self.transcripts = Path(transcripts)

    def call(self, family, prompt, schema, *, tag, timeout=900):
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
        limit = min(60, self.grant.model_limit('opus')) if family == 'opus' else 60
        self.budget.reserve_model(family, limit, at)
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
            try:
                result = self.runner(argv(family, schema, schema_file), input=prompt, text=True, capture_output=True,
                                     cwd=str(cwd), env=env, timeout=min(timeout, remaining - 1))
            except (subprocess.TimeoutExpired, OSError):
                raise ReplayError('CALL_OUTCOME_UNKNOWN', identity) from None
        record.update(state='RECEIVED', raw=result.stdout or '', err=result.stderr or '', returncode=result.returncode)
        atomic_json(path, record)
        return self._decode(record, family, prompt, schema, identity)

    def _decode(self, record, family, prompt, schema, identity):
        try:
            output, reported = decode(record['raw'], record['err'], record['returncode'], family, schema)
        except ReplayError as error:
            if error.code == 'MODEL_USAGE_LIMIT':
                record['state'] = 'USAGE_LIMIT'
                atomic_json(self.transcripts / (identity + '.json'), record)
                self.budget.stop_model(family, instant(record['at']))
            raise
        return output, {'family': family, 'model': spec.spec()['models']['identities'][family], 'reported': reported,
                        'identity_evidence': 'CLI_REQUEST_ONLY' if not reported else 'CLI_REPORTED',
                        'called_at': record['at'], 'transcript': identity,
                        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
