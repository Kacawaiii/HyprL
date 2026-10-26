import os

import pytest

from tests.replay_seal_forward.conftest import ledger as base_ledger  # noqa: F401


@pytest.fixture
def ledger(base_ledger):
    original = base_ledger.runner

    def dated(args, **kwargs):
        if args[0] == 'git' and 'commit' in args:
            kwargs['env'] = dict(os.environ, GIT_COMMITTER_DATE=base_ledger.clock().isoformat())
        return original(args, **kwargs)

    base_ledger.runner = dated
    return base_ledger
