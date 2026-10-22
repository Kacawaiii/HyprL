"""Bounded, no-tool CLI calls. Reservations outlive failures; no automatic retries."""
from pathlib import Path
import os
import signal
import subprocess
import tempfile

from .contracts import Blocked, EXTRACTION, JUDGMENT, canonical, digest, iso, strict_json, utc, validate
from .spec import HERE, spec

QUOTA = ('rate limit', 'rate_limit', 'usage limit', 'usage_limit', 'limit reached',
         'hit your limit', 'insufficient_quota', 'too many requests', '429')


def command(role, schema_path):
    if role == 'contradictor':
        argv = ['codex', 'exec', '--ignore-user-config', '--ignore-rules', '--sandbox', 'read-only',
                '--ephemeral', '--skip-git-repo-check', '--model', spec()['gpt_model'],
                '--output-schema', str(schema_path), '--json']
        for feature in ('shell_tool', 'code_mode', 'code_mode_host', 'multi_agent', 'apps', 'plugins',
                        'hooks', 'browser_use', 'computer_use', 'image_generation', 'skill_search',
                        'view_image', 'sleep_tool', 'tool_suggest', 'enable_mcp_apps', 'remote_plugin',
                        'multi_agent_v2', 'skill_mcp_dependency_install'):
            argv += ['--disable', feature]
        return argv + ['--enable', 'skip_host_skill_discovery', '-c', 'web_search="disabled"', '-']
    return ['claude', '-p', '--model', 'opus', '--tools', '', '--permission-mode', 'dontAsk',
            '--setting-sources', '', '--settings', '{"disableAllHooks":true}', '--strict-mcp-config',
            '--mcp-config', '{"mcpServers":{}}', '--disable-slash-commands', '--no-session-persistence',
            '--output-format', 'json', '--json-schema', schema_path.read_text()]


def parse(role, raw):
    if role != 'contradictor':
        value = strict_json(raw)
        if not isinstance(value, dict) or value.get('is_error') or value.get('type') != 'result':
            raise Blocked('MODEL_FAILED')
        identities = sorted(value.get('modelUsage', {}))
        if not identities or any(not name.startswith('claude-opus-') for name in identities):
            raise Blocked('MODEL_IDENTITY_UNPROVEN')
        return value.get('structured_output'), identities
    events = [strict_json(line) for line in raw.splitlines() if line.strip()]
    # Fail closed on every tool, including unknown/new tool types; metadata/assistant output only.
    allowed_events = {'thread.started', 'turn.started', 'turn.completed', 'item.started', 'item.completed'}
    allowed_items = {'reasoning', 'agent_message'}
    if not events or any(not isinstance(e, dict) or e.get('type') not in allowed_events or
            'item' in e and e['item'].get('type') not in allowed_items for e in events):
        raise Blocked('MODEL_TOOL_OR_ERROR')
    messages = [e['item']['text'] for e in events if e.get('type') == 'item.completed'
                and e['item'].get('type') == 'agent_message']
    if not messages or not any(e.get('type') == 'turn.completed' for e in events):
        raise Blocked('MODEL_FAILED')
    # Codex JSON does not attest the served model. Preserve this limitation explicitly.
    return strict_json(messages[-1]), ['requested:' + spec()['gpt_model'], 'served_identity_unreported']


class Models:
    synthetic = False

    def __init__(self, journal, grant, *, clock=utc):
        self.journal, self.grant, self.clock = journal, grant, clock

    def call(self, role, value):
        if role not in ('extraction', 'analyst', 'contradictor'):
            raise Blocked('MODEL_ROLE_INVALID')
        grant = self.grant.check()
        at = self.clock()
        if 15 * 60 + 30 <= at.hour * 60 + at.minute < 17 * 60:
            raise Blocked('V1_PM_PROTECTED_WINDOW')
        if self.journal.binding['mode'] != 'SHADOW_FORWARD':
            raise Blocked('REAL_MODEL_IN_SYNTHETIC')
        family = 'gpt' if role == 'contradictor' else 'claude_opus'
        schema = EXTRACTION if role == 'extraction' else JUDGMENT
        with self.journal.owned():
            rows = [r for r in self.journal.records('model_reservation') if r['at'][:10] == iso(at)[:10]]
            if any(r['at'][:10] == iso(at)[:10] for r in self.journal.records('model_halt')):
                raise Blocked('MODEL_USAGE_LIMIT')
            if sum(r['body']['family'] == family for r in rows) >= grant['models'][family + '_calls_per_day']:
                raise Blocked('MODEL_BUDGET')
            self.journal.append([('model_reservation', {'role': role, 'family': family,
                                  'input_hash': digest(value), 'authorization_hash': digest(grant)})], at)
        prompt = (HERE / 'prompts' / (role + '.txt')).read_text() + '\nDATA:\n' + canonical(value)
        if len(prompt) > 500_000:
            raise Blocked('MODEL_CONTEXT_TOO_LARGE')
        with tempfile.TemporaryDirectory(prefix='quality-model-') as directory:
            schema_path = Path(directory) / 'schema.json'
            schema_path.write_text(canonical(schema))
            env = {k: v for k, v in os.environ.items() if not any(marker in k.upper()
                   for marker in ('APCA_', 'ALPACA', 'CLAUDECODE', 'CODEX_THREAD'))}
            with tempfile.TemporaryFile(mode='w+') as out, tempfile.TemporaryFile(mode='w+') as err:
                try:
                    proc = subprocess.Popen(command(role, schema_path), stdin=subprocess.PIPE, stdout=out,
                                            stderr=err, text=True, cwd=directory, env=env, start_new_session=True)
                    try:
                        proc.communicate(prompt, timeout=180)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.communicate()
                        raise Blocked('MODEL_TIMEOUT') from None
                except OSError:
                    raise Blocked('MODEL_UNAVAILABLE') from None
                out.seek(0)
                err.seek(0)
                raw, errors = out.read(2_000_001), err.read(100_001)
                if len(raw) > 2_000_000 or len(errors) > 100_000:
                    raise Blocked('MODEL_OUTPUT_TOO_LARGE')
                error_text = errors + (raw if proc.returncode else '')
                if role != 'contradictor' and not proc.returncode:
                    envelope = strict_json(raw)
                    if isinstance(envelope, dict) and envelope.get('is_error'):
                        error_text += raw
                if any(word in error_text.lower() for word in QUOTA):
                    with self.journal.owned():
                        self.journal.append([('model_halt', {'reason': 'MODEL_USAGE_LIMIT'})], self.clock())
                    raise Blocked('MODEL_USAGE_LIMIT')
                if proc.returncode:
                    raise Blocked('MODEL_FAILED')
        output, identities = parse(role, raw)
        validate(output, schema)
        with self.journal.owned():
            self.journal.append([('model_result', {'role': role, 'output': output,
                                  'input_hash': digest(value), 'reported_models': identities})], self.clock())
        return output
