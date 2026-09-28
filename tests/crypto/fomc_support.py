"""Shared offline environment for the FOMC slice tests (simulated clock, local provider, store, collector)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from scripts.trading_lab.fomc import identity, spec
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.collector import Collector
from scripts.trading_lab.fomc.store import FomcStore

EXACT = spec.FEED_TITLE_EXACT
P1 = syn.statement_path("20260617")
SID1 = identity.source_item_id(syn.url(P1))


class Env:
    def __init__(self, tmp_path, *, wall_offset_s=0.0):
        self.clock = syn.SimClock(syn.START, wall_offset_s=wall_offset_s)
        self.provider = syn.LocalProvider(self.clock)
        self.root = tmp_path / "fomc"
        self.store = FomcStore(self.root, wall_clock=self.clock.wall)
        self.collector = Collector(self.store, self.provider.connector(), self.clock)

    def feed(self, items):
        self.provider.routes[syn.FEED_PATH] = syn.feed_response(items)

    def drive(self, seconds, idle=30):
        end = self.clock.true + timedelta(seconds=seconds)
        results = []
        while self.clock.true < end:
            r = self.collector.step()
            if r is None:
                self.clock.sleep(idle)
            else:
                results.append(r)
        return results

    def restart(self, boot_id="boot-2"):
        self.collector.close()
        self.store.close()
        self.store = FomcStore(self.root, wall_clock=self.clock.wall)
        self.collector = Collector(self.store, self.provider.connector(), self.clock, boot_id=boot_id)

    def cycles(self):
        return [c.body["result"] for c in self.store.rows("CYCLE_CONCLUSION")]

    def outcomes(self):
        return [o.body["outcome"] for o in self.store.rows("PROCESSING_OUTCOME")]


def statement_item(path=P1, title=EXACT, guid="g1"):
    return {"title": title, "link": syn.url(path), "guid": guid}
