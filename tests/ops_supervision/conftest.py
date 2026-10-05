import json
from pathlib import Path
import shutil
import socket
import uuid

import pytest

from scripts.trading_lab.ops.control import WORKTREE, load_config, stop


@pytest.fixture
def config(tmp_path):
    root = WORKTREE / "var" / ("ops-test-" + uuid.uuid4().hex)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    path = tmp_path / "private-config.json"
    path.write_text(json.dumps({"schema": "hyprl-ops-v1", "runtime_root": str(root), "port": port}))
    path.chmod(0o600)
    configuration = load_config(path)
    yield configuration
    for name in ("app", "workers", "edgar"):
        try:
            stop(configuration, name, timeout=10)
        except ValueError:
            pass
    if root.exists():
        shutil.rmtree(root)


@pytest.fixture
def backup_target():
    path = WORKTREE / "var" / ("ops-backup-test-" + uuid.uuid4().hex)
    yield path
    if path.exists():
        shutil.rmtree(path)
