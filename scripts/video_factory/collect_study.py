"""Collect only the operator-approved hourly BTC study range; no other endpoint."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .delivery import atomic_json
from .study import instant, plan_hash

ID = "video-factory-btc-asia-v1"
ORIGIN = "https://api.exchange.coinbase.com"
PATH = "/products/BTC-USD/candles"
UTC = timezone.utc


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def authorize(path, at):
    raw = Path(path).read_bytes()
    grant = json.loads(raw)
    scope = grant.get("scope", {})
    expected = dict(
        origin=ORIGIN,
        method="GET",
        path=PATH,
        product="BTC-USD",
        granularity_seconds=3600,
        start="2018-01-01T00:00:00Z",
        end_exclusive="2026-09-01T00:00:00Z",
        max_requests_including_retries=400,
        max_requests_per_second=1,
    )
    if (
        grant.get("id") != ID
        or grant.get("schema") != "research-data-authorization-v1"
        or grant.get("status") != "AUTHORIZED"
        or grant.get("operator_signed") is not True
        or scope != expected
        or not isinstance(grant.get("not_after"), str)
        or at >= instant(grant["not_after"])
    ):
        raise ValueError("STUDY_AUTHORIZATION_REQUIRED")
    return sha(raw)


def windows():
    start = instant("2018-01-01T00:00:00Z")
    end = instant("2026-09-01T00:00:00Z")
    while start < end:
        stop = min(end, start + timedelta(hours=288))
        yield start, stop
        start = stop


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def transport(url):
    request = Request(
        url,
        headers={"User-Agent": "HyprL-authorized-Asia-low-study/1.0", "Accept": "application/json"},
        method="GET",
    )
    try:
        with build_opener(NoRedirect()).open(request, timeout=30) as response:
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("STUDY_RESPONSE_TOO_LARGE")
            return response.status, raw
    except HTTPError as error:
        return error.code, b""


def checked_rows(raw, start, end):
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) > 300:
        raise ValueError("STUDY_CANDLE_SHAPE")
    rows = []
    discarded = 0
    safe_start = instant("2018-01-01T00:00:00Z")
    safe_end = instant("2026-09-01T00:00:00Z")
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 6
            or type(row[0]) not in (int, float)
            or not math.isfinite(row[0])
        ):
            raise ValueError("STUDY_CANDLE_SHAPE")
        at = datetime.fromtimestamp(row[0], UTC)
        if at >= safe_end:
            raise ValueError("STUDY_PROTECTED_TIMESTAMP")  # never inspect its OHLCV
        if at < safe_start or not start <= at < end:
            discarded += 1
            continue
        if at.timestamp() % 3600 or not all(
            type(v) in (int, float) and math.isfinite(v) for v in row
        ):
            raise ValueError("STUDY_CANDLE_GRID")
        _, low, high, opened, closed, volume = row
        if not 0 < low <= min(opened, closed) <= max(opened, closed) <= high or volume < 0:
            raise ValueError("STUDY_CANDLE_OHLCV")
        rows.append(row)
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("STUDY_DUPLICATE_CANDLE")
    return rows, discarded


def collect(
    authorization,
    output,
    *,
    bank=None,
    clock=lambda: datetime.now(UTC),
    send=transport,
    pause=time.sleep,
):
    authorization = Path(authorization)
    identity = authorize(authorization, clock())
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or repo in output.parents:
        raise ValueError("PRIVATE_STUDY_OUTPUT_REQUIRED")
    bank = Path(bank) if bank else Path.home() / "data/source-budgets" / ID
    bank.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (bank / "owner.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = bank / "requests.jsonl"
        records = (
            [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        )
        previous = "0" * 64
        for record in records:
            body = {k: v for k, v in record.items() if k != "hash"}
            if (
                body["previous"] != previous
                or sha(encoded(body)) != record["hash"]
                or body["authorization_sha256"] != identity
            ):
                raise ValueError("STUDY_BUDGET_INTEGRITY")
            previous = record["hash"]
        if any(r["kind"] == "receipt" and r.get("status") in (403, 429) for r in records):
            raise ValueError("STUDY_SOURCE_HALTED")

        def append(kind, **fields):
            nonlocal previous
            record = dict(
                kind=kind,
                at=clock().isoformat(),
                authorization_sha256=identity,
                previous=previous,
                **fields,
            )
            record["hash"] = sha(encoded(record))
            previous = record["hash"]
            with log.open("a") as stream:
                os.chmod(log, 0o600)
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            records.append(record)

        manifest_path = output / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else dict(
                product="BTC-USD",
                source="coinbase_exchange",
                plan_sha256=plan_hash(),
                authorization_sha256=identity,
                shards=[],
            )
        )
        if manifest["authorization_sha256"] != identity or manifest["plan_sha256"] != plan_hash():
            raise ValueError("STUDY_MANIFEST_BINDING")
        by_start = {s["start"]: s for s in manifest["shards"]}
        for index, (start, end) in enumerate(windows()):
            stamp = start.strftime("%Y-%m-%dT%H:%M:%SZ")
            if stamp in by_start:
                shard = by_start[stamp]
                if sha((output / shard["file"]).read_bytes()) != shard["sha256"]:
                    raise ValueError("STUDY_SHARD_BINDING")
                continue
            # The final request ends one second before the forbidden interval.
            url = (
                ORIGIN
                + PATH
                + "?"
                + urlencode(
                    dict(
                        granularity=3600,
                        start=stamp,
                        end=(end - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    )
                )
            )
            prior = [r for r in records if r["kind"] == "request" and r["url"] == url]
            if len(prior) >= 2:
                raise ValueError("STUDY_RETRY_EXHAUSTED")
            for attempt in range(len(prior), 2):
                if authorize(authorization, clock()) != identity:
                    raise ValueError("STUDY_AUTHORIZATION_CHANGED")
                requests = [r for r in records if r["kind"] == "request"]
                if len(requests) >= 400:
                    raise ValueError("STUDY_REQUEST_BUDGET")
                if requests:
                    remaining = 1.05 - (clock() - instant(requests[-1]["at"])).total_seconds()
                    if remaining > 0:
                        pause(remaining)
                if authorize(authorization, clock()) != identity:
                    raise ValueError("STUDY_AUTHORIZATION_CHANGED")
                append("request", url=url, number=len(requests) + 1)
                try:
                    status, raw = send(url)
                except OSError:
                    status, raw = 599, b""
                append("receipt", url=url, status=status, body_sha256=sha(raw), bytes=len(raw))
                if status == 200:
                    break
                if status < 500 and status != 408 or attempt == 1:
                    raise ValueError("STUDY_HTTP_" + str(status))
                pause(2)
            rows, discarded = checked_rows(raw, start, end)
            name = f"candles-{index:03d}.json"
            data = encoded(rows)
            (output / name).write_bytes(data)
            shard = dict(
                file=name,
                start=stamp,
                end_exclusive=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                granularity_seconds=3600,
                authorization_id=ID,
                sha256=sha(data),
                rows=len(rows),
                response_sha256=sha(raw),
                discarded_outside_window=discarded,
                received_at=clock().isoformat(),
            )
            manifest["shards"].append(shard)
            atomic_json(manifest_path, manifest)
            print(
                json.dumps(
                    {
                        "completed_shards": len(manifest["shards"]),
                        "requests": sum(r["kind"] == "request" for r in records),
                        "rows": len(rows),
                    }
                ),
                flush=True,
            )
        return dict(
            status="COLLECTED",
            shards=len(manifest["shards"]),
            rows=sum(s["rows"] for s in manifest["shards"]),
            requests=sum(r["kind"] == "request" for r in records),
            manifest_sha256=sha(manifest_path.read_bytes()),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        print(json.dumps(collect(args.authorization, args.output)))
        return 0
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
