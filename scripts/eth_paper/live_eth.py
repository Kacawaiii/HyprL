#!/usr/bin/env python3
"""Run one isolated ETH Alpaca paper account.

Both variants use the exact same long/cash base strategy.  ``news`` may only
reduce exposure after a tagged risk headline; ``control`` never reads news.

Dry-run is the default and writes nothing.  ``--live`` submits paper orders and
records only daily equity, broker orders, and derived news-event metadata.
Credentials are read from the environment and are never persisted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eth_paper.strategy import (  # noqa: E402
    EthStrategyConfig,
    NewsRisk,
    apply_account_risk_caps,
    compute_base_signal,
    evaluate_news_risk,
    plan_rebalance,
    quantize_exposure,
)


PAPER_API = "https://paper-api.alpaca.markets"
DATA_API = "https://data.alpaca.markets"
SYMBOL = "ETH/USD"
NORMALIZED_SYMBOL = "ETHUSD"
OPEN_STATUSES = {
    "accepted",
    "new",
    "pending_new",
    "partially_filled",
    "accepted_for_bidding",
    "pending_replace",
    "pending_cancel",
}
TERMINAL_FAILURES = {"rejected", "canceled", "expired", "suspended"}


class AlpacaError(RuntimeError):
    """Sanitized Alpaca API error."""


def _normalize_symbol(value: object) -> str:
    return "".join(char for char in str(value).upper() if char.isalnum())


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


class AlpacaPaperClient:
    def __init__(self, key: str, secret: str):
        if not key or not secret:
            raise AlpacaError("missing Alpaca paper credentials")
        self._headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json",
            "User-Agent": "HyprL-ETH-Paper/1.0",
        }

    def _request(
        self,
        method: str,
        base: str,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        body: Mapping[str, object] | None = None,
        retries: int = 3,
        allow_404: bool = False,
    ):
        if base not in (PAPER_API, DATA_API):
            raise AlpacaError("non-paper/non-data endpoint blocked")
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode(params)
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"{base}{path}{query}",
            data=payload,
            method=method,
            headers=self._headers,
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as exc:
                if allow_404 and exc.code == 404:
                    return None
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    detail = ""
                if exc.code == 429 and attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise AlpacaError(
                    f"Alpaca {method} {path} -> HTTP {exc.code}: {detail}"
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if method == "GET" and attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise AlpacaError(
                    f"Alpaca {method} {path} did not confirm a response"
                ) from exc
        raise AlpacaError(f"Alpaca {method} {path} exhausted retries")

    def get_account(self) -> dict:
        return self._request("GET", PAPER_API, "/v2/account")

    def get_positions(self) -> list[dict]:
        return self._request("GET", PAPER_API, "/v2/positions")

    def get_open_orders(self) -> list[dict]:
        payload = self._request(
            "GET",
            PAPER_API,
            "/v2/orders",
            params={"status": "open", "limit": 100, "direction": "desc"},
        )
        return list(payload or [])

    def get_recent_orders(self) -> list[dict]:
        payload = self._request(
            "GET",
            PAPER_API,
            "/v2/orders",
            params={
                "status": "all",
                "limit": 100,
                "direction": "desc",
                "nested": "true",
            },
        )
        return list(payload or [])

    def get_asset(self) -> dict:
        path_symbol = urllib.parse.quote(SYMBOL, safe="")
        return self._request("GET", PAPER_API, f"/v2/assets/{path_symbol}")

    def get_portfolio_history(self) -> dict:
        return self._request(
            "GET",
            PAPER_API,
            "/v2/account/portfolio/history",
            params={"period": "1A", "timeframe": "1D"},
        )

    def _paged_data(
        self,
        path: str,
        params: Mapping[str, object],
        *,
        item_key: str,
        max_pages: int = 20,
    ) -> list[dict]:
        rows: list[dict] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        for _ in range(max_pages):
            page_params = dict(params)
            if token:
                page_params["page_token"] = token
            payload = self._request("GET", DATA_API, path, params=page_params)
            data = payload.get(item_key, {})
            if isinstance(data, dict):
                symbol_rows = data.get(SYMBOL) or data.get(NORMALIZED_SYMBOL) or []
                rows.extend(symbol_rows)
            elif isinstance(data, list):
                rows.extend(data)
            token = payload.get("next_page_token")
            if not token:
                return rows
            token = str(token)
            if token in seen_tokens:
                raise AlpacaError(f"repeated page token from {path}")
            seen_tokens.add(token)
        raise AlpacaError(f"page cap exceeded for {path}")

    def get_bars(
        self,
        *,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        return self._paged_data(
            "/v1beta3/crypto/us/bars",
            {
                "symbols": SYMBOL,
                "timeframe": timeframe,
                "start": _iso(start),
                "end": _iso(end),
                "limit": 10000,
                "sort": "asc",
            },
            item_key="bars",
        )

    def get_latest_quote(self) -> tuple[float, float]:
        payload = self._request(
            "GET",
            DATA_API,
            "/v1beta3/crypto/us/latest/quotes",
            params={"symbols": SYMBOL},
        )
        quotes = payload.get("quotes", {})
        quote = quotes.get(SYMBOL) or quotes.get(NORMALIZED_SYMBOL) or {}
        bid = _as_float(quote.get("bp") or quote.get("bid_price"))
        ask = _as_float(quote.get("ap") or quote.get("ask_price"))
        if bid <= 0 or ask <= 0 or ask < bid:
            raise AlpacaError("invalid ETH/USD quote")
        return bid, ask

    def get_news(self, *, start: datetime, end: datetime) -> list[dict]:
        rows: list[dict] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        for _ in range(20):
            params: dict[str, object] = {
                "symbols": "ETHUSD,BTCUSD",
                "start": _iso(start),
                "end": _iso(end),
                "sort": "asc",
                "limit": 50,
            }
            if token:
                params["page_token"] = token
            payload = self._request(
                "GET",
                DATA_API,
                "/v1beta1/news",
                params=params,
            )
            rows.extend(payload.get("news", []))
            token = payload.get("next_page_token")
            if not token:
                return rows
            token = str(token)
            if token in seen_tokens:
                raise AlpacaError("repeated Alpaca news page token")
            seen_tokens.add(token)
        raise AlpacaError("Alpaca news page cap exceeded")

    def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        return self._request(
            "GET",
            PAPER_API,
            "/v2/orders:by_client_order_id",
            params={"client_order_id": client_order_id},
            allow_404=True,
        )

    def get_order(self, order_id: str) -> dict:
        return self._request("GET", PAPER_API, f"/v2/orders/{order_id}")

    def submit_order(self, body: Mapping[str, object]) -> dict:
        client_order_id = str(body["client_order_id"])
        existing = self.get_order_by_client_id(client_order_id)
        if existing is not None:
            return existing
        try:
            return self._request(
                "POST",
                PAPER_API,
                "/v2/orders",
                body=body,
                retries=1,
            )
        except AlpacaError as exc:
            # A timed-out POST has unknown outcome.  Query the idempotency key
            # once rather than blindly sending a duplicate order.
            time.sleep(2)
            existing = self.get_order_by_client_id(client_order_id)
            if existing is not None:
                return existing
            raise exc


def _completed_close_series(
    bars: list[dict],
    *,
    now: datetime,
    timeframe: str,
) -> pd.Series:
    records = []
    if timeframe == "1Day":
        cutoff = pd.Timestamp(now.astimezone(timezone.utc).date(), tz="UTC")
    elif timeframe == "1Hour":
        cutoff = pd.Timestamp(now).floor("h")
    else:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    for bar in bars:
        try:
            timestamp = pd.Timestamp(bar["t"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            close = float(bar["c"])
        except (KeyError, TypeError, ValueError):
            continue
        if timestamp < cutoff and close > 0:
            records.append((timestamp, close))
    if not records:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(records, columns=["timestamp", "close"])
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return frame.set_index("timestamp")["close"]


def _position_snapshot(positions: list[dict]) -> tuple[float, float]:
    eth_qty = 0.0
    eth_market_value = 0.0
    unexpected = []
    for position in positions:
        symbol = _normalize_symbol(position.get("symbol"))
        qty = _as_float(position.get("qty"))
        if symbol == NORMALIZED_SYMBOL:
            if qty < 0:
                raise AlpacaError("short ETH position found on spot-only account")
            eth_qty += qty
            eth_market_value += max(0.0, _as_float(position.get("market_value")))
        elif abs(qty) > 0:
            unexpected.append(symbol)
    if unexpected:
        raise AlpacaError(
            "dedicated ETH account contains unexpected positions: "
            + ",".join(sorted(unexpected))
        )
    return eth_qty, eth_market_value


def _portfolio_peak(history: Mapping[str, object], current_equity: float) -> float:
    values = [
        _as_float(value, default=float("nan"))
        for value in history.get("equity", [])
        if value is not None
    ]
    valid = [value for value in values if value > 0]
    return max(valid + [current_equity])


def _available_cash(account: Mapping[str, object]) -> float:
    cash = max(0.0, _as_float(account.get("cash")))
    non_marginable = max(
        0.0,
        _as_float(account.get("non_marginable_buying_power"), cash),
    )
    return min(cash, non_marginable) if non_marginable > 0 else cash


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _already_logged(path: Path, field: str, value: object) -> bool:
    if not path.exists():
        return False
    for raw in path.read_text(encoding="utf-8").splitlines()[-200:]:
        try:
            if json.loads(raw).get(field) == value:
                return True
        except (json.JSONDecodeError, AttributeError):
            continue
    return False


def _logged_order_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(raw)
            order = payload.get("order", {})
            order_id = str(order.get("id") or "")
        except (json.JSONDecodeError, AttributeError):
            continue
        if order_id:
            result.add(order_id)
    return result


def _is_owned_terminal_order(order: Mapping[str, object], variant: str) -> bool:
    short_variant = "n" if variant == "news" else "c"
    prefix = f"hyprl-eth-{short_variant}-"
    status = str(order.get("status") or "").lower()
    return (
        _normalize_symbol(order.get("symbol")) == NORMALIZED_SYMBOL
        and str(order.get("client_order_id") or "").startswith(prefix)
        and (status == "filled" or status in TERMINAL_FAILURES)
    )


def _order_log_payload(
    snapshot: Mapping[str, object],
    order: Mapping[str, object],
) -> dict[str, object]:
    return {
        **snapshot,
        "order": {
            "id": str(order.get("id") or ""),
            "client_order_id": str(order.get("client_order_id") or ""),
            "status": str(order.get("status") or ""),
            "side": str(order.get("side") or ""),
            "qty": str(order.get("qty") or ""),
            "notional": str(order.get("notional") or ""),
            "filled_qty": str(order.get("filled_qty") or ""),
            "filled_avg_price": str(order.get("filled_avg_price") or ""),
            "submitted_at": str(order.get("submitted_at") or ""),
            "filled_at": str(order.get("filled_at") or ""),
            "failed_at": str(order.get("failed_at") or ""),
            "canceled_at": str(order.get("canceled_at") or ""),
            "expired_at": str(order.get("expired_at") or ""),
        },
    }


def _wait_for_order(
    client: AlpacaPaperClient,
    order: Mapping[str, object],
    *,
    attempts: int = 12,
) -> dict:
    current = dict(order)
    order_id = str(current.get("id") or "")
    if not order_id:
        raise AlpacaError("submitted order has no id")
    for _ in range(attempts):
        status = str(current.get("status") or "").lower()
        if status == "filled" or status in TERMINAL_FAILURES:
            return current
        time.sleep(1)
        current = client.get_order(order_id)
    return current


def _client_order_id(
    variant: str,
    now: datetime,
    action: str,
    target_exposure: float,
) -> str:
    short_variant = "n" if variant == "news" else "c"
    target_bps = int(round(target_exposure * 10_000))
    return (
        f"hyprl-eth-{short_variant}-{now:%Y%m%dT%H}-{action[0]}-{target_bps:04d}"
    )


def _news_risk(
    client: AlpacaPaperClient,
    *,
    now: datetime,
) -> NewsRisk:
    hourly_bars = client.get_bars(
        timeframe="1Hour",
        start=now - timedelta(days=3),
        end=now,
    )
    hourly = _completed_close_series(hourly_bars, now=now, timeframe="1Hour")
    if len(hourly) < 7:
        raise AlpacaError("insufficient completed ETH hourly bars for news confirmation")
    return_1h = float(hourly.iloc[-1] / hourly.iloc[-2] - 1.0)
    return_6h = float(hourly.iloc[-1] / hourly.iloc[-7] - 1.0)
    articles = client.get_news(start=now - timedelta(hours=12), end=now)
    return evaluate_news_risk(
        articles,
        now=now,
        return_1h=return_1h,
        return_6h=return_6h,
    )


def run_once(args: argparse.Namespace) -> dict:
    now = datetime.now(timezone.utc)
    config = EthStrategyConfig()
    client = AlpacaPaperClient(
        os.environ.get("ALPACA_KEY", ""),
        os.environ.get("ALPACA_SECRET", ""),
    )

    account = client.get_account()
    account_id = str(account.get("id") or "")
    if not account_id.endswith(args.expected_account_suffix):
        raise AlpacaError("credentials do not match the expected dedicated account")
    if str(account.get("status") or "").upper() != "ACTIVE":
        raise AlpacaError("Alpaca account is not active")
    if str(account.get("crypto_status") or "").upper() != "ACTIVE":
        raise AlpacaError("crypto is not active on this account")
    if account.get("trading_blocked"):
        raise AlpacaError("trading is blocked on this account")

    asset = client.get_asset()
    if not asset.get("tradable") or asset.get("shortable"):
        raise AlpacaError("ETH/USD asset flags violate the spot-only contract")

    equity = _as_float(account.get("equity"))
    if equity <= 0:
        raise AlpacaError("invalid account equity")
    positions = client.get_positions()
    current_qty, position_market_value = _position_snapshot(positions)

    daily_bars = client.get_bars(
        timeframe="1Day",
        start=now - timedelta(days=1_100),
        end=now,
    )
    daily = _completed_close_series(daily_bars, now=now, timeframe="1Day")
    signal = compute_base_signal(daily, as_of=daily.index[-1], config=config)
    expected_bar_date = now.date() - timedelta(days=1)
    if signal.as_of.date() != expected_bar_date:
        raise AlpacaError(
            "latest completed ETH daily bar does not match the previous UTC day"
        )

    news = NewsRisk(
        multiplier=1.0,
        level="disabled_control",
        reasons=(),
        article_count=0,
        critical_count=0,
        policy_count=0,
        event_sha256="",
    )
    target = signal.target_exposure
    if args.variant == "news":
        news = _news_risk(client, now=now)
        target = quantize_exposure(
            target * news.multiplier,
            step=config.exposure_step,
            maximum=config.max_exposure,
        )

    history = client.get_portfolio_history()
    peak_equity = _portfolio_peak(history, equity)
    previous_day_equity = _as_float(account.get("last_equity"), equity)
    target, risk_reasons = apply_account_risk_caps(
        target,
        current_equity=equity,
        peak_equity=peak_equity,
        previous_day_equity=previous_day_equity,
        config=config,
    )

    bid, ask = client.get_latest_quote()
    mid = (bid + ask) / 2.0
    current_value = position_market_value or current_qty * mid
    # Use the quote-derived value for sizing so stale position marks do not
    # create a false rebalance.
    if current_qty > 0:
        current_value = current_qty * mid
    plan = plan_rebalance(
        equity=equity,
        target_exposure=target,
        current_qty=current_qty,
        current_price=mid,
        available_cash=_available_cash(account),
        config=config,
    )

    open_orders = [
        order
        for order in client.get_open_orders()
        if _normalize_symbol(order.get("symbol")) == NORMALIZED_SYMBOL
        and str(order.get("status") or "").lower() in OPEN_STATUSES
    ]
    order_result: dict | None = None
    if open_orders:
        execution = "existing_open_order"
    elif plan.action == "hold":
        execution = "hold"
    elif not args.live:
        execution = "dry_run"
    else:
        client_order_id = _client_order_id(
            args.variant,
            now,
            plan.action,
            target,
        )
        if plan.action == "buy":
            body = {
                "symbol": SYMBOL,
                "notional": f"{plan.order_notional:.2f}",
                "side": "buy",
                "type": "market",
                "time_in_force": "gtc",
                "client_order_id": client_order_id,
            }
        else:
            body = {
                "symbol": SYMBOL,
                "qty": f"{plan.order_qty:.9f}".rstrip("0").rstrip("."),
                "side": "sell",
                "type": "market",
                "time_in_force": "gtc",
                "client_order_id": client_order_id,
            }
        order_result = _wait_for_order(client, client.submit_order(body))
        status = str(order_result.get("status") or "").lower()
        execution = f"order_{status or 'unknown'}"
        if status in TERMINAL_FAILURES:
            raise AlpacaError(f"ETH paper order ended in status={status}")

    snapshot = {
        "ts": _iso(now),
        "date": now.date().isoformat(),
        "variant": args.variant,
        "account_suffix": args.expected_account_suffix,
        "mode": "live" if args.live else "dry_run",
        "equity": round(equity, 2),
        "peak_equity": round(peak_equity, 2),
        "current_qty": round(current_qty, 9),
        "current_value": round(current_value, 2),
        "price": round(mid, 4),
        "base_target": signal.target_exposure,
        "effective_target": target,
        "composite_score": round(signal.composite_score, 6),
        "realized_volatility": round(signal.realized_volatility, 6),
        "news_level": news.level,
        "news_multiplier": news.multiplier,
        "news_article_count": news.article_count,
        "news_event_sha256": news.event_sha256,
        "risk_reasons": list(risk_reasons),
        "plan": plan.action,
        "plan_reason": plan.reason,
        "execution": execution,
        "open_orders": len(open_orders),
    }

    if args.live:
        out = args.state_dir
        equity_path = out / "equity.jsonl"
        if not _already_logged(equity_path, "date", snapshot["date"]):
            _append_jsonl(equity_path, snapshot)
        if args.variant == "news" and news.multiplier < 1.0:
            events_path = out / "events.jsonl"
            if not _already_logged(
                events_path,
                "news_event_sha256",
                news.event_sha256,
            ):
                _append_jsonl(
                    events_path,
                    {
                        **snapshot,
                        "news_reasons": list(news.reasons),
                        "news_critical_count": news.critical_count,
                        "news_policy_count": news.policy_count,
                    },
                )
        orders_path = out / "orders.jsonl"
        logged_order_ids = _logged_order_ids(orders_path)
        candidates = ([] if order_result is None else [order_result])
        candidates.extend(client.get_recent_orders())
        for broker_order in candidates:
            order_id = str(broker_order.get("id") or "")
            if (
                not order_id
                or order_id in logged_order_ids
                or not _is_owned_terminal_order(broker_order, args.variant)
            ):
                continue
            _append_jsonl(
                orders_path,
                _order_log_payload(snapshot, broker_order),
            )
            logged_order_ids.add(order_id)

    print(json.dumps(snapshot, sort_keys=True))
    return snapshot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("news", "control"), required=True)
    parser.add_argument("--expected-account-suffix", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument(
        "--live",
        action="store_true",
        help="submit orders to the paper endpoint (default: dry-run)",
    )
    args = parser.parse_args(argv)
    if not args.expected_account_suffix.isalnum():
        parser.error("expected account suffix must be alphanumeric")
    try:
        args.state_dir.resolve().relative_to(ROOT.resolve())
    except ValueError:
        parser.error("state-dir must be inside the repository")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_once(args)
    except Exception as exc:
        # Never include request headers or credentials in errors.
        print(
            json.dumps(
                {
                    "variant": args.variant,
                    "ok": False,
                    "error": str(exc)[:500],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
