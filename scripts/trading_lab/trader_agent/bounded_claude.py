"""Duplex CLI permission gate. Only the host can admit a web operation."""
import json
import os
import select
import selectors
import signal
import subprocess
import tempfile
import time
from urllib.parse import unquote, urlsplit

from .config import TraderError, strict_json
from .work_budget import policy, usage


class Gate:
    def __init__(self, limits, role, allowance, *, probe=False):
        self.limits, self.allowance = limits, allowance
        self.caps = {name: 0 if probe else count for name, count in limits['tools'][role].items()}
        self.counts = dict.fromkeys(self.caps, 0)
        self.request_ids, self.tool_ids, self.messages = set(), set(), {}
        self.result = None
        self.error_text = ''

    def event(self, event):
        from .runners import tainted
        if not isinstance(event, dict):
            raise TraderError('MODEL_PERMISSION_PROTOCOL_INVALID')
        if tainted(event):
            raise TraderError('TAINTED_RUN')
        if event.get('type') in {'error', 'turn.failed', 'rate_limit_event'} or event.get('is_error'):
            self.error_text += json.dumps(event)
        if event.get('type') == 'assistant':
            message = event.get('message', {})
            key = message.get('id')
            if not isinstance(key, str):
                raise TraderError('MODEL_PERMISSION_PROTOCOL_INVALID')
            # A message can be re-emitted with additional blocks: count its ID once.
            amounts = message.get('usage', {})
            total = 0
            for name in ('input_tokens', 'cache_creation_input_tokens', 'output_tokens'):
                value = amounts.get(name, 0)
                if type(value) is not int or value < 0:
                    raise TraderError('MODEL_USAGE_INVALID')
                total += value
            self.messages[key] = max(self.messages.get(key, 0), total)
            if len(self.messages) > self.limits['max_turns']:
                raise TraderError('MODEL_TURN_BUDGET_EXHAUSTED')
            if sum(self.messages.values()) > self.allowance:
                raise TraderError('WORK_BUDGET_OVERRUN')
        if event.get('type') == 'control_request':
            request, key = event.get('request', {}), event.get('request_id')
            if not isinstance(key, str) or key in self.request_ids or request.get('subtype') != 'can_use_tool':
                raise TraderError('MODEL_PERMISSION_PROTOCOL_INVALID')
            self.request_ids.add(key)
            tool, tool_id = request.get('tool_name'), request.get('tool_use_id')
            if tool not in self.caps or not isinstance(tool_id, str) or tool_id in self.tool_ids:
                raise TraderError('MODEL_PERMISSION_PROTOCOL_INVALID')
            self.tool_ids.add(tool_id)
            if self.counts[tool] >= self.caps[tool]:
                raise TraderError('MODEL_TOOL_BUDGET_EXHAUSTED')
            inputs = request.get('input')
            if not isinstance(inputs, dict):
                raise TraderError('MODEL_PERMISSION_PROTOCOL_INVALID')
            if tool == 'WebFetch':
                try:
                    url = urlsplit(inputs.get('url', ''))
                    host = unquote(url.hostname or '').encode('idna').decode('ascii').lower().rstrip('.')
                    official = any(host == domain or host.endswith('.' + domain)
                                   for domain in ('sec.gov', 'federalreserve.gov'))
                    if url.scheme != 'https' or not host or url.username or url.password or official:
                        raise ValueError()
                except (ValueError, TypeError, UnicodeError):
                    raise TraderError('TAINTED_RUN') from None
            self.counts[tool] += 1  # reserve before sending ALLOW, including failed web operations
            return {'type': 'control_response', 'response': {'subtype': 'success', 'request_id': key,
                'response': {'behavior': 'allow', 'updatedInput': inputs}}}
        if event.get('type') == 'result':
            if self.result is not None:
                raise TraderError('MODEL_PERMISSION_PROTOCOL_INVALID')
            self.result = event
            if event.get('subtype') == 'error_max_turns':
                raise TraderError('MODEL_TURN_BUDGET_EXHAUSTED')
            turns = event.get('num_turns')
            if type(turns) is not int or turns > self.limits['max_turns'] or turns < 0:
                raise TraderError('MODEL_TURN_BUDGET_EXHAUSTED')
            if not event.get('is_error'):
                new, _ = usage(event)
                if new > self.allowance:
                    raise TraderError('WORK_BUDGET_OVERRUN')
        return None


def once(runner, role, prompt, limits, *, probe=False):
    from .runners import QUOTA, command, parse_claude
    ledger = runner.ledger
    if len(prompt.encode('utf-8')) > limits['max_prompt_bytes']:
        raise TraderError('WORK_BUDGET_PROMPT_TOO_LARGE')
    args = command(role, ledger.grant, limits=limits)
    version = subprocess.run([args[0], '--version'], capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    transcript_role = 'rehearsal_' + role if probe else role
    seq = ledger.reserve(transcript_role)
    allowance = limits['admission']['probe_per_call' if probe else 'per_call']
    gate = Gate(limits, role, allowance, probe=probe)
    folder = ledger.root / 'transcripts'
    folder.mkdir(exist_ok=True, mode=0o700)
    new, cached, outcome = 0, 0, 'MODEL_FAILED'
    try:
        with tempfile.TemporaryDirectory(prefix='trader-empty-') as cwd:
            env = dict(os.environ)
            for name in ('CLAUDECODE', 'CODEX_THREAD_ID'):
                env.pop(name, None)
            env.update(CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(limits['max_output_tokens']),
                       CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION=str(gate.caps['WebSearch']),
                       CLAUDE_CODE_MAX_RETRIES='0')
            ledger.grant.check_slot(ledger.slot, runner.clock(), preflight=probe)
            policy(ledger.grant, runner.clock())
            deadline = getattr(runner, 'deadline', None)
            if ledger.paused:
                raise TraderError('PAUSED')
            if deadline and runner.clock() >= deadline:
                raise TraderError('MISSED_DECISION_DEADLINE')
            timeout = ledger.grant.payload['external_models'][role]['timeout_minutes'] * 60
            if probe:
                timeout = min(timeout, 120)
            if deadline:
                timeout = min(timeout, max(.1, (deadline - runner.clock()).total_seconds()))
            expires = time.monotonic() + timeout
            with (folder / f'{seq}-{transcript_role}.stdout').open('wb') as out, \
                 (folder / f'{seq}-{transcript_role}.stderr').open('w+b') as err:
                proc = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=err, start_new_session=True)
                try:
                    os.set_blocking(proc.stdin.fileno(), False)
                    def send(event):
                        pending = memoryview((json.dumps(event, ensure_ascii=False) + '\n').encode())
                        while pending:
                            remaining = expires - time.monotonic()
                            if remaining <= 0:
                                raise TraderError('MODEL_TIMEOUT')
                            if not select.select([], [proc.stdin], [], min(remaining, 1))[1]:
                                continue
                            try:
                                count = os.write(proc.stdin.fileno(), pending)
                            except BlockingIOError:
                                continue
                            if not count:
                                raise BrokenPipeError()
                            pending = pending[count:]

                    send({'type': 'control_request', 'request_id': 'work-budget-init',
                          'request': {'subtype': 'initialize'}})
                    initialized, buffer, size = False, b'', 0
                    with selectors.DefaultSelector() as selector:
                        selector.register(proc.stdout, selectors.EVENT_READ)
                        while True:
                            remaining = expires - time.monotonic()
                            if remaining <= 0:
                                raise TraderError('MODEL_TIMEOUT')
                            if not selector.select(min(remaining, 1)):
                                if ledger.paused:
                                    raise TraderError('PAUSED')
                                continue
                            chunk = os.read(proc.stdout.fileno(), 65536)
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > (64000 if probe else 8000000):
                                raise TraderError('MODEL_OUTPUT_TOO_LARGE')
                            out.write(chunk)
                            buffer += chunk
                            while b'\n' in buffer:
                                line, buffer = buffer.split(b'\n', 1)
                                if not line.strip():
                                    continue
                                event = strict_json(line.decode('utf-8'))
                                if event.get('type') == 'control_response':
                                    response = event.get('response', {})
                                    if (initialized or response.get('request_id') != 'work-budget-init'
                                            or response.get('subtype') != 'success'):
                                        raise TraderError('MODEL_PERMISSION_PROTOCOL_INVALID')
                                    initialized = True
                                    send({'type': 'user', 'session_id': '', 'parent_tool_use_id': None,
                                          'message': {'role': 'user', 'content': prompt}})
                                    continue
                                if not initialized:
                                    # The installed CLI can emit UI/system notifications
                                    # before acknowledging SDK initialization. No user
                                    # prompt or tool permission is sent until the ACK.
                                    if event.get('type') == 'system':
                                        gate.event(event)
                                        continue
                                    raise TraderError('MODEL_PERMISSION_PROTOCOL_INVALID')
                                reply = gate.event(event)
                                if event.get('type') == 'result':
                                    proc.stdin.close()  # one turn only; do not wait for another SDK user message
                                if reply:
                                    # Recheck capability immediately before every web tool ALLOW.
                                    ledger.grant.check_slot(ledger.slot, runner.clock(), preflight=probe)
                                    policy(ledger.grant, runner.clock())
                                    if ledger.paused:
                                        raise TraderError('PAUSED')
                                    if deadline and runner.clock() >= deadline:
                                        raise TraderError('MISSED_DECISION_DEADLINE')
                                    send(reply)
                    if buffer.strip():
                        raise TraderError('MODEL_PERMISSION_PROTOCOL_INVALID')
                    proc.wait(timeout=max(.1, expires - time.monotonic()))
                    err.seek(0)
                    errors = err.read(200001).decode('utf-8', errors='replace')
                    if len(errors) > 200000:
                        raise TraderError('MODEL_OUTPUT_TOO_LARGE')
                    result = gate.result or {}
                    error_text = errors + gate.error_text
                    if any(marker in error_text.lower() for marker in QUOTA):
                        raise TraderError('SKIPPED_QUOTA')
                    if not result or proc.returncode:
                        raise TraderError('MODEL_FAILED')
                    new, cached = usage(result)
                    output = parse_claude(json.dumps(result))
                finally:
                    if proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    proc.wait()
                    proc.stdin.close()
                    proc.stdout.close()
        runner.metadata[role] = {'model': ledger.grant.payload['external_models'][role]['model'],
            'cli_version': version, 'reported_version': ','.join(sorted(gate.result['modelUsage']))}
        runner.web_searches[role] = gate.counts['WebSearch']
        outcome = 'COMPLETE'
        return output
    except TraderError as error:
        outcome = error.code
        raise
    except (ValueError, TypeError, UnicodeError, BrokenPipeError) as error:
        outcome = 'MODEL_PERMISSION_PROTOCOL_INVALID'
        raise TraderError(outcome) from error
    except subprocess.TimeoutExpired:
        outcome = 'MODEL_TIMEOUT'
        raise TraderError(outcome) from None
    finally:
        complete = False
        if gate.result:
            try:
                new, cached = usage(gate.result)
                complete = True
            except TraderError:
                outcome = 'MODEL_USAGE_INVALID'
        else:
            # This is only observed main-model work; helper totals may be missing after a kill.
            new = sum(gate.messages.values())
        ledger.record_work(seq, new, cached, outcome, usage_complete=complete)
