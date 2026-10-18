"""No-tool CLI with serial dispatch, shared night cap and durable call intents."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from jsonschema import Draft202012Validator

from .grant import ReplayError, instant, iso, now
from .storage import atomic_json, append_event
from . import spec

MODELS = {'opus': 'claude-opus-5-5', 'sonnet': 'claude-sonnet-5-5'}
QUOTA = ('rate_limit', 'rate limit', 'usage limit', 'usage_limit', 'limit reached', 'hit your limit',
         'hit your session limit', 'session limit', 'out of extra usage', 'insufficient_quota', 'too many requests',
         'you\'ve hit your limit', 'usage cap', 'credit balance', 'limit exceeded')


def night_remaining(at):
    start = at.replace(hour=0, minute=30, second=0, microsecond=0)
    end = at.replace(hour=5, minute=30, second=0, microsecond=0)
    if not start <= at < end:
        raise ReplayError('OUTSIDE_NIGHT_WINDOW')
    seconds = int((end - at).total_seconds())
    if seconds < 5:
        raise ReplayError('NIGHT_WINDOW_END')
    return seconds


def argv(model, schema):
    return ['claude', '-p', '--model', model, '--tools', '', '--permission-mode', 'dontAsk',
            '--system-prompt', 'Answer only the supplied research question using the supplied data. No tools.',
            '--setting-sources', '', '--settings', '{"disableAllHooks":true}', '--strict-mcp-config',
            '--mcp-config', '{"mcpServers":{}}', '--disable-slash-commands', '--no-session-persistence',
            '--output-format', 'json', '--json-schema', json.dumps(schema)]


def parse(raw):
    try:
        event = json.loads(raw)
    except ValueError:
        raise ReplayError('MODEL_JSON_INVALID') from None
    if not isinstance(event, dict) or event.get('is_error') or event.get('subtype', 'success') != 'success':
        raise ReplayError('MODEL_FAILED')
    if event.get('permission_denials'):
        raise ReplayError('TAINTED_RUN')
    output = event.get('structured_output')
    if not isinstance(output, dict):
        raise ReplayError('MODEL_JSON_INVALID')
    return output, event


class Claude:
    def __init__(self, grant, budget, transcripts, *, runner=subprocess.run, clock=now):
        self.grant, self.budget, self.runner, self.clock = grant, budget, runner, clock
        self.transcripts = Path(transcripts)
        self.transcripts.mkdir(parents=True, exist_ok=True, mode=0o700)

    def call(self, family, prompt, schema, *, tag, timeout=900):
        if family != 'opus':
            raise ReplayError('MODEL_NOT_ALLOWED')
        identity = hashlib.sha256(json.dumps([family, prompt, schema, tag], sort_keys=True).encode()).hexdigest()
        path = self.transcripts / (identity + '.json')
        previous = json.loads(path.read_text()) if path.exists() else None
        # A saved response is replayed locally, never re-dispatched. An unresolved intent cannot be retried.
        if previous and previous['state'] == 'RECEIVED':
            if self._quota(previous['raw'], previous['err']):
                previous['state'] = 'USAGE_LIMIT'
                atomic_json(path, previous)
                self.budget.mark_stopped('MODEL_USAGE_LIMIT', instant(previous['at']))
            else:
                return self._decode(previous['raw'], previous['err'], previous['returncode'], family,
                                    previous['at'], identity, prompt, schema)
        if previous and previous['state'] == 'INTENT':
            raise ReplayError('CALL_OUTCOME_UNKNOWN', identity)
        at = self.clock()
        remaining = night_remaining(at)
        if self.budget.stopped(at):
            raise ReplayError('MODEL_USAGE_LIMIT')
        if previous and previous['state'] == 'USAGE_LIMIT' and previous['at'][:10] == iso(at)[:10]:
            raise ReplayError('MODEL_USAGE_LIMIT')
        self.grant.check()
        self.budget.reserve_model(family, self.grant.model_limit(family), at,
                                  night_limit=spec.spec()['models']['night']['max_calls'])
        record = {'state': 'INTENT', 'at': iso(at), 'family': family, 'tag': tag,
                  'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        atomic_json(path, record)
        append_event(self.transcripts / 'calls.jsonl', dict(record, identity=identity))
        with tempfile.TemporaryDirectory(prefix='replay-empty-') as cwd:
            env = {k: v for k, v in os.environ.items() if k not in ('CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT')}
            env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY='1', CLAUDE_CODE_DISABLE_CLAUDE_MDS='1', CLAUDE_CODE_SIMPLE='1')
            try:
                result = self.runner(argv(family, schema), input=prompt, text=True, capture_output=True,
                                     timeout=min(timeout, remaining - 1), cwd=cwd, env=env)
            except subprocess.TimeoutExpired:
                # Keep INTENT: a timeout may have consumed a real decision, so no blind retry.
                raise ReplayError('CALL_OUTCOME_UNKNOWN', identity) from None
        raw, err = result.stdout or '', result.stderr or ''
        record.update(state='RECEIVED', raw=raw, err=err, returncode=result.returncode)
        atomic_json(path, record)
        append_event(self.transcripts / 'calls.jsonl', dict(record, identity=identity))
        if self._quota(raw, err):
            record['state'] = 'USAGE_LIMIT'
            atomic_json(path, record)
            self.budget.mark_stopped('MODEL_USAGE_LIMIT', at)
            raise ReplayError('MODEL_USAGE_LIMIT')
        return self._decode(raw, err, result.returncode, family, iso(at), identity, prompt, schema)

    @staticmethod
    def _quota(raw, err):
        try:
            event = json.loads(raw)
        except ValueError:
            event = {}
        if not isinstance(event, dict):
            event = {}
        if event.get('api_error_status') == 429:
            return True
        # Inspect transport errors even if the CLI also returned a structured object.
        message = str(event.get('result', '')) + err
        if event.get('is_error') or not isinstance(event.get('structured_output'), dict):
            message += raw
        return any(marker in message.lower() for marker in QUOTA)

    def _decode(self, raw, err, returncode, family, at, identity, prompt, schema):
        if self._quota(raw, err):
            raise ReplayError('MODEL_USAGE_LIMIT')
        if returncode:
            raise ReplayError('MODEL_FAILED')
        output, event = parse(raw)
        reported = sorted(event.get('modelUsage', {}))
        if reported != [MODELS[family]]:
            raise ReplayError('MODEL_IDENTITY_CHANGED', reported)
        if not Draft202012Validator(schema).is_valid(output):
            raise ReplayError('MODEL_SCHEMA_INVALID')
        meta = {'family': family, 'model': MODELS[family], 'reported': reported, 'called_at': at,
                'cost_usd': event.get('total_cost_usd'), 'duration_ms': event.get('duration_ms'),
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'transcript': identity}
        return output, meta
