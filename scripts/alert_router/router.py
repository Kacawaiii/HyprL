"""One outbound worker; durable reservations survive failures and restarts."""

from __future__ import annotations

import random
import time

from .destinations import request
from .model import Alert, AlertError, instant
from .security import HTTPS


class Router:
    def __init__(
        self, store, transport=None, *, clock=time.time, jitter=random.random, telegram_token=None
    ):
        self.store = store
        self.transport = transport or HTTPS()
        self.clock = clock
        self.jitter = jitter
        self.telegram_token = telegram_token

    async def run_once(self):
        job = self.store.claim(now=self.clock())
        if job is None:
            return False
        state, code, retry = "dead", "INVALID_JOB", 0
        try:
            alert = Alert.parse(job["event"])
            if not self.store.can_send(job):
                self.store.finish(job, state="pending", code="PAUSED", due=self.clock() + 10)
                return True
            if (
                alert.event()["kind"] == "duel_hashes"
                and self.clock() + 5 >= instant(alert.event()["facts"]["open_at"]).timestamp()
            ):
                raise AlertError("HASH_WINDOW_CLOSED")
            url, body, headers = request(
                job["config"], alert, now=self.clock(), telegram_token=self.telegram_token
            )
            receipt = await self.transport.post(url, body, headers)
            if 200 <= receipt.status < 300:
                state, code = "sent", "DELIVERED"
            else:
                code = f"HTTP_{receipt.status}" if 100 <= receipt.status <= 599 else "HTTP_ERROR"
                if receipt.status in (408, 429) or 500 <= receipt.status < 600:
                    state, retry = "pending", receipt.retry_after
        except AlertError as error:
            # Never propagate arbitrary exception text, URLs or response bodies.
            code = (
                str(error)
                if str(error)
                in {
                    "SSRF_BLOCKED",
                    "UNSAFE_URL",
                    "DNS_EMPTY",
                    "DNS_PIN_MISMATCH",
                    "HASH_WINDOW_CLOSED",
                    "TELEGRAM_NOT_CONFIGURED",
                }
                else "INVALID_JOB"
            )
        except Exception:
            state, code = "pending", "TRANSPORT_FAILED"
        if state == "pending" and job["attempts"] >= 5:
            state = "dead"
        backoff = max(retry, min(3600, 10 * 2 ** (job["attempts"] - 1)) + self.jitter() * 3)
        self.store.finish(
            job,
            state=state,
            code=code,
            due=self.clock() + backoff if state == "pending" else self.clock(),
        )
        return True
