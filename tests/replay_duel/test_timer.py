"""Only fake user units and subprocesses; never mutate a real timer in tests."""
import importlib.util
from pathlib import Path
import os
import subprocess

import pytest

from scripts.trading_lab.replay_duel import spec


def fixture(tmp_path, monkeypatch):
    definition = importlib.util.spec_from_file_location('duel_timer', spec.ROOT / 'deploy/research/install_duel_timer.py')
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    target = tmp_path / '.config/systemd/user'
    target.mkdir(parents=True)
    service, timer = target / 'hyprl-replay-bench.service', target / 'hyprl-replay-bench.timer'
    service.write_text('ExecStart=python -m scripts.trading_lab.replay_bench.cli night --expected-commit old\n')
    timer.write_text('Unit=hyprl-replay-bench.service\n')
    env = tmp_path / 'research.env'
    env.write_text('REPLAY_DATA_ROOT=synthetic\n')
    os.chmod(env, 0o600)
    for name in ('claude', 'codex'):
        file = tmp_path / name
        file.write_text('synthetic executable')
        os.chmod(file, 0o700)
    def read(args, **kwargs):
        if args[0] == str(tmp_path / 'codex'):
            return 'codex-cli 0.160.0\n'
        if args[0] == 'systemctl':
            return 'inactive\n'
        return 'synthetic-sha\n' if 'rev-parse' in args else ''
    monkeypatch.setattr(subprocess, 'check_output', read)
    args = ['--environment', str(env), '--claude', str(tmp_path / 'claude'), '--codex', str(tmp_path / 'codex'),
            '--backup-dir', str(tmp_path / 'backup')]
    return module, args, service, timer


def test_failed_unit_verification_preserves_old_units(tmp_path, monkeypatch):
    module, args, service, timer = fixture(tmp_path, monkeypatch)
    original = service.read_bytes(), timer.read_bytes()
    def fail(args, **kwargs):
        assert args[:3] == ['systemd-analyze', '--user', 'verify']
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.main(args)
    assert (service.read_bytes(), timer.read_bytes()) == original
    assert not (tmp_path / 'backup').exists()


def test_only_replay_units_replaced_with_backup_and_exact_sha(tmp_path, monkeypatch):
    module, args, service, timer = fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda args, **kwargs: calls.append(args))
    module.main(args)
    assert (tmp_path / 'backup' / service.name).is_file()
    assert 'replay_duel.cli night --expected-commit synthetic-sha' in service.read_text()
    assert 'MemoryMax=700M' in service.read_text()
    assert calls[0][:3] == ['systemd-analyze', '--user', 'verify']
    assert calls[-1] == ['systemctl', '--user', 'enable', '--now', timer.name]


def test_unrelated_units_are_never_replaced(tmp_path, monkeypatch):
    module, args, service, timer = fixture(tmp_path, monkeypatch)
    service.write_text('ExecStart=unrelated\n')
    with pytest.raises(SystemExit, match='unrelated units'):
        module.main(args)
    assert service.read_text() == 'ExecStart=unrelated\n' and not (tmp_path / 'backup').exists()
