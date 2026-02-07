"""
HyprL Status API - Provides live trading status for landing page widget
Can be used as:
1. Standalone Flask/FastAPI endpoint
2. Imported by Streamlit for st.experimental_get_query_params() JSON mode
3. Cron job to generate static status.json
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

# Try to import Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import OrderSide, QueryOrderStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _default_status_payload(status: str, *, last_signal: str, error: Optional[str] = None) -> dict:
    payload = {
        "status": status,
        "account_value": 100000.0,
        "today_pnl": 0.0,
        "open_positions": 0,
        "last_signal": last_signal,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        payload["error"] = error
    return payload


def _normalize_status_payload(payload: dict, *, default_status: str) -> dict:
    base = _default_status_payload(
        default_status,
        last_signal=payload.get("last_signal") or "Paper trading starting soon",
        error=payload.get("error"),
    )
    base.update(payload)
    base["status"] = base.get("status") or default_status
    base["account_value"] = _coerce_float(base.get("account_value"), 100000.0)
    base["today_pnl"] = _coerce_float(base.get("today_pnl"), 0.0)
    base["open_positions"] = _coerce_int(base.get("open_positions"), 0)
    base["last_signal"] = base.get("last_signal") or "Paper trading starting soon"
    base["timestamp"] = base.get("timestamp") or datetime.now(timezone.utc).isoformat()
    return base


def get_alpaca_status() -> dict:
    """Fetch live status from Alpaca paper trading account."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        return _normalize_status_payload(
            {
                "status": "offline",
                "error": "Alpaca credentials not configured",
                "last_signal": "Paper trading offline",
            },
            default_status="offline",
        )

    if not ALPACA_AVAILABLE:
        return _normalize_status_payload(
            {
                "status": "offline",
                "error": "Alpaca SDK not installed",
                "last_signal": "Paper trading offline",
            },
            default_status="offline",
        )

    try:
        # Paper trading
        client = TradingClient(api_key, secret_key, paper=True)

        # Get account info
        account = client.get_account()

        # Get positions
        positions = client.get_all_positions()

        # Get today's orders
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        orders_request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=today.isoformat()
        )
        orders = client.get_orders(orders_request)

        # Calculate today's P/L from closed orders
        today_pnl = 0.0
        last_signal = None

        for order in orders:
            if order.status == "filled":
                # Simple P/L approximation
                if order.filled_avg_price and order.filled_qty:
                    # Track last signal
                    side = "LONG" if order.side == OrderSide.BUY else "SHORT"
                    last_signal = f"{order.symbol} {side} @ ${float(order.filled_avg_price):.2f}"

        # Use account's daily P/L if available
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        today_pnl = equity - last_equity

        return _normalize_status_payload(
            {
                "status": "online",
                "account_value": equity,
                "today_pnl": today_pnl,
                "open_positions": len(positions),
                "last_signal": last_signal or "Awaiting signal",
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            default_status="online",
        )

    except Exception as e:
        return _normalize_status_payload(
            {
                "status": "error",
                "error": str(e),
                "last_signal": "Paper trading offline",
            },
            default_status="error",
        )


def get_status_from_state_file(state_path: Optional[str] = None) -> dict:
    """Read status from local state file (for offline mode)."""
    if state_path is None:
        state_path = os.getenv("BRIDGE_STATE_FILE", "/home/ubuntu/HyprL/live/state/bridge_alpaca.json")

    try:
        with open(state_path, "r") as f:
            state = json.load(f)

        return _normalize_status_payload(
            {
                "status": "online",
                "account_value": state.get("account_value", 100000),
                "today_pnl": state.get("today_pnl", 0),
                "open_positions": len(state.get("positions", {})),
                "last_signal": state.get("last_signal", "Awaiting signal"),
                "timestamp": state.get("updated_at", datetime.now(timezone.utc).isoformat()),
            },
            default_status="online",
        )
    except FileNotFoundError:
        return _normalize_status_payload(
            {
                "status": "pending",
                "last_signal": "Paper trading starting soon",
            },
            default_status="pending",
        )
    except Exception as e:
        return _normalize_status_payload(
            {
                "status": "error",
                "error": str(e),
                "last_signal": "Paper trading offline",
            },
            default_status="error",
        )


def get_public_status() -> dict:
    """Return a resilient status payload for public consumption."""
    status = get_alpaca_status()
    if status.get("status") == "online":
        status["source"] = "alpaca"
        return status

    fallback = get_status_from_state_file()
    if fallback.get("status") in {"online", "pending"}:
        fallback["source"] = "state"
        return fallback

    status["source"] = "alpaca"
    if fallback.get("error") and not status.get("error"):
        status["error"] = fallback["error"]
    return status


def generate_status_json(output_path: str = "/tmp/hyprl_status.json") -> dict:
    """Generate status JSON file for static hosting."""
    status = get_public_status()

    # Write to file
    with open(output_path, "w") as f:
        json.dump(status, f, indent=2)

    return status


# Waitlist storage
WAITLIST_FILE = os.getenv("WAITLIST_FILE", "/home/ubuntu/HyprL/live/waitlist.jsonl")


def save_waitlist_entry(data: dict) -> bool:
    """Append a waitlist entry to JSONL file."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": data.get("name", "").strip(),
            "email": data.get("email", "").strip().lower(),
            "experience": data.get("experience", ""),
            "capital": data.get("capital", ""),
            "ip": data.get("ip", ""),
        }

        # Basic validation
        if not entry["email"] or "@" not in entry["email"]:
            return False

        # Append to file
        os.makedirs(os.path.dirname(WAITLIST_FILE), exist_ok=True)
        with open(WAITLIST_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return True
    except Exception:
        return False


# Flask/FastAPI integration
def create_flask_app():
    """Create a minimal Flask app for status endpoint."""
    from flask import Flask, jsonify, request, redirect
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)  # Allow cross-origin for landing page

    @app.route("/api/status")
    def status():
        return jsonify(get_public_status())

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/subscribe", methods=["POST"])
    def subscribe():
        """Handle beta waitlist form submissions."""
        # Get form data (supports both JSON and form-encoded)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Add IP for analytics
        data["ip"] = request.headers.get("X-Forwarded-For", request.remote_addr)

        if save_waitlist_entry(data):
            # If form submission (not AJAX), redirect to thanks page
            if not request.is_json and request.form:
                return redirect("/thanks.html")
            return jsonify({"status": "ok", "message": "Added to waitlist"})
        else:
            return jsonify({"status": "error", "message": "Invalid email"}), 400

    @app.route("/api/waitlist/count")
    def waitlist_count():
        """Return waitlist count (for admin)."""
        try:
            with open(WAITLIST_FILE, "r") as f:
                count = sum(1 for _ in f)
            return jsonify({"count": count})
        except FileNotFoundError:
            return jsonify({"count": 0})

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HyprL Status API")
    parser.add_argument("--mode", choices=["generate", "serve", "print"], default="print")
    parser.add_argument("--output", default="/tmp/hyprl_status.json")
    parser.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    if args.mode == "generate":
        status = generate_status_json(args.output)
        print(f"Status written to {args.output}")
        print(json.dumps(status, indent=2))

    elif args.mode == "serve":
        app = create_flask_app()
        app.run(host="0.0.0.0", port=args.port)

    else:
        status = get_public_status()
        print(json.dumps(status, indent=2))
