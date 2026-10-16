"""Read-only projection of a separately configured v2 shadow ledger."""
from pathlib import Path
import sqlite3

from .contracts import AppApiError, ConflictError, NotFoundError


class UnavailableError(AppApiError):
    status = 503


class AgentV2Views:
    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else None

    def dispatch(self, path, query):
        if path != '/api/v1/agent-v2/summary':
            raise NotFoundError('no such v2 endpoint')
        if query:
            raise AppApiError('v2 summary does not accept query parameters')
        if self.root is None:
            raise UnavailableError('v2 shadow runtime not configured')
        from scripts.trading_lab.agent_v2.contracts import Blocked
        from scripts.trading_lab.agent_v2.ledger import Ledger
        from scripts.trading_lab.agent_v2.reporting import summary
        ledger = None
        try:
            ledger = Ledger(self.root, read_only=True)
            return summary(ledger)
        except Blocked:
            raise ConflictError('v2 ledger or configuration integrity failure') from None
        except (OSError, sqlite3.Error, ValueError, KeyError):
            raise UnavailableError('v2 shadow evidence unavailable') from None
        finally:
            if ledger:
                ledger.close()
