import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

from scripts.trading_lab.replay_seal_forward import spec


def fixture(tmp_path, monkeypatch):
    definition = importlib.util.spec_from_file_location('seal_timers', spec.ROOT / 'deploy/research/install_seal_forward_timers.py')
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    target = tmp_path / '.config/systemd/user'
    target.mkdir(parents=True)
    service = target / 'hyprl-replay-bench.service'
    service.write_text('ExecStart=python -m scripts.trading_lab.replay_duel.cli night --expected-commit old\n')
    (target / 'hyprl-replay-bench.timer').write_text('Unit=hyprl-replay-bench.service\n')
    env = tmp_path / 'research.env'; env.write_text('REPLAY_DATA_ROOT=synthetic\n'); os.chmod(env, 0o600)
    for name in ('claude', 'codex'):
        file = tmp_path / name; file.write_text('synthetic'); os.chmod(file, 0o700)
    def read(args, **kwargs):
        if args[0] == str(tmp_path/'codex'):
            return 'codex-cli 0.160.0\n'
        if args[0] == 'systemctl':
            return 'inactive\n'
        return 'sha\n' if 'rev-parse' in args else ''
    monkeypatch.setattr(subprocess, 'check_output', read)
    args = ['--environment', str(env), '--claude', str(tmp_path/'claude'), '--codex', str(tmp_path/'codex'),
            '--backup-dir', str(tmp_path/'backup')]
    return module, args, target, service


def test_verify_before_replace_all_four_research_timers_with_named_timezone(tmp_path, monkeypatch):
    module, args, target, service = fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda args, **kwargs: calls.append(args))
    module.main(args)
    assert calls[0][:3] == ['systemd-analyze', '--user', 'verify'] and len(calls[0][3:]) == 16
    assert (tmp_path/'backup'/service.name).is_file()
    assert 'replay_seal_forward.cli night --expected-commit sha' in service.read_text()
    assert '09:05:00 America/New_York' in (target/'hyprl-research-forward.timer').read_text()
    assert '17:15:00 America/New_York' in (target/'hyprl-research-forward-weekly.timer').read_text()
    assert all('Persistent=false' in p.read_text() for p in target.glob('*.timer'))
    assert len([c for c in calls if c[:4] == ['systemctl', '--user', 'enable', '--now']]) == 8
    assert '22:45:00 UTC' in (target/'hyprl-research-forward-drill.timer').read_text()
    assert 'RuntimeMaxSec=infinity' in service.read_text()
    assert '*:00/15:00 UTC' in (target/'hyprl-replay-bench.timer').read_text()


def test_verification_failure_and_unrelated_units_are_preserved(tmp_path, monkeypatch):
    module, args, target, service = fixture(tmp_path, monkeypatch)
    original = service.read_bytes()
    def fail(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.main(args)
    assert service.read_bytes() == original and not (tmp_path/'backup').exists()
    service.write_text('ExecStart=unrelated\n')
    with pytest.raises(SystemExit, match='unrelated unit'):
        module.main(args)


def test_refresh_keeps_existing_calendars_and_adds_0400_once(tmp_path, monkeypatch):
    module, _, _, _ = fixture(tmp_path, monkeypatch)
    old = '[Timer]\nOnCalendar=*-*-* 00:20:00 UTC\nOnCalendar=*-*-* 08:55:00 America/New_York\n'
    result = module.refresh_timer_text(old)
    assert 'OnCalendar=*-*-* 04:00:00 UTC' in result
    assert '08:55:00 America/New_York' in result
    assert module.refresh_timer_text(result) == result
    with pytest.raises(SystemExit, match='Unrecognized'):
        module.refresh_timer_text('[Timer]\nOnCalendar=daily\n')
