"""No-tool Claude extraction and veto-only review, with durable per-runtime budgets."""
import json
import subprocess
from pathlib import Path

from .contracts import Blocked, NEWS, SUPERVISOR, canonical, digest, instant, iso, strict_json, utc, validate
from .spec import HERE, PREREG_HASH, spec


class Claude:
    def __init__(self, ledger, authorization, *, clock=utc, executable='claude'):
        self.ledger, self.clock, self.executable = ledger, clock, executable
        self.path = Path(authorization)

    def grant(self):
        grant = strict_json(self.path.read_text())
        now = self.clock()
        if (grant.get('schema') != 'agent-v2-shadow-authorization-v1'
                or grant.get('operator_signed') is not True or grant.get('mode') != 'SHADOW_FORWARD'
                or grant.get('preregistration_hash') != PREREG_HASH
                or grant.get('operator_decision_hash') != spec()['operator_decision_hash']
                or not instant(grant['granted_at']) <= now < instant(grant['not_after'])
                or type(grant.get('claude_calls_per_day')) is not int
                or not 0 < grant.get('claude_calls_per_day', 0) <= 8
                or grant.get('model') != 'sonnet'):
            raise Blocked('V2_AUTHORIZATION_REQUIRED')
        return grant

    def call(self, role, value):
        grant = self.grant()
        at = self.clock()
        calls = [r for r in self.ledger.records('model_reservation') if r['at'][:10] == iso(at)[:10]]
        if len(calls) >= grant['claude_calls_per_day']:
            raise Blocked('MODEL_BUDGET')
        if any(r['body'].get('reason') == 'MODEL_USAGE_LIMIT' and r['at'][:10] == iso(at)[:10]
               for r in self.ledger.records('blocked')):
            raise Blocked('MODEL_USAGE_LIMIT')
        schema = NEWS if role == 'news' else SUPERVISOR
        prompt = (HERE / 'prompts' / (role + '.txt')).read_text()
        self.ledger.append([('model_reservation', {'role': role, 'input_hash': digest(value),
                                                  'authorization_hash': digest(grant), 'model': 'sonnet'})], at)
        argv = [self.executable, '-p', '--model', 'sonnet', '--tools', '', '--permission-mode', 'dontAsk',
                '--setting-sources', '', '--settings', '{"disableAllHooks":true}', '--strict-mcp-config',
                '--mcp-config', '{"mcpServers":{}}', '--disable-slash-commands', '--no-session-persistence',
                '--output-format', 'json', '--json-schema', json.dumps(schema)]
        # Runtime contains no repo instructions; tools/MCP/hooks/session persistence are disabled.
        try:
            result = subprocess.run(argv, input=prompt + '\nDATA:\n' + canonical(value), text=True,
                                    capture_output=True, timeout=180, cwd=self.ledger.root)
        except (OSError, subprocess.TimeoutExpired):
            raise Blocked('MODEL_UNAVAILABLE') from None
        raw = result.stdout
        if any(term in (raw + result.stderr).lower() for term in
               ('rate limit', 'usage limit', 'limit reached', 'hit your limit', 'insufficient_quota', '429')):
            raise Blocked('MODEL_USAGE_LIMIT')
        if result.returncode:
            raise Blocked('MODEL_FAILED')
        envelope = strict_json(raw)
        if envelope.get('is_error') or envelope.get('type') != 'result':
            raise Blocked('MODEL_FAILED')
        output = envelope.get('structured_output')
        validate(output, schema)
        self.ledger.append([('model_result', {'role': role, 'output': output, 'output_hash': digest(output),
                                            'model': 'sonnet'})], self.clock())
        return output


class SyntheticClaude:
    """Explicit synthetic fixture only, never usable by a forward runtime."""
    def call(self, role, value):
        if role == 'news':
            return {'events': []}
        return {'reviews': [{'plan_id': p['plan_id'], 'verdict': 'ALLOW', 'reason': 'NONE'} for p in value['plans']]}
