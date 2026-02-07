#!/usr/bin/env python3
"""
HyprL Notifications API - Realtime alerts system
Handles: Email, Discord, API signals, Webhooks
"""

import os
import json
import hmac
import hashlib
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, Any, List

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app, origins=["https://hyprlcore.com", "https://app.hyprlcore.com", "http://localhost:*"])

# ============================================================================
# Configuration
# ============================================================================

SIGNALS_FILE = os.environ.get("SIGNALS_FILE", "/home/ubuntu/HyprL/live/logs/live_signals.jsonl")
FIREBASE_CREDS = os.environ.get("FIREBASE_CREDS", "/home/ubuntu/HyprL/secrets/firebase-admin.json")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# Initialize Firebase if not already done
try:
    firebase_admin.get_app()
except ValueError:
    if os.path.exists(FIREBASE_CREDS):
        cred = credentials.Certificate(FIREBASE_CREDS)
        firebase_admin.initialize_app(cred)

db = firestore.client() if os.path.exists(FIREBASE_CREDS) else None

# ============================================================================
# Authentication Helpers
# ============================================================================

def get_user_by_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Look up user by API key in Firestore."""
    if not db or not api_key:
        return None

    # API keys stored as hashed in Firestore
    try:
        users_ref = db.collection("users")
        # Find user with matching API key
        query = users_ref.where("apiKey", "==", api_key).limit(1)
        docs = list(query.stream())
        if docs:
            user_data = docs[0].to_dict()
            user_data["uid"] = docs[0].id
            return user_data
    except Exception as e:
        print(f"Error looking up API key: {e}")
    return None


def require_api_key(allowed_tiers: List[str]):
    """Decorator to require API key authentication with tier check."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
            if not api_key:
                return jsonify({"error": "Missing API key", "code": "missing_api_key"}), 401

            user = get_user_by_api_key(api_key)
            if not user:
                return jsonify({"error": "Invalid API key", "code": "invalid_api_key"}), 401

            user_tier = user.get("tier", "observer")
            if user_tier not in allowed_tiers:
                return jsonify({
                    "error": f"This endpoint requires {', '.join(allowed_tiers)} tier",
                    "code": "insufficient_tier",
                    "your_tier": user_tier
                }), 403

            request.user = user
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def verify_firebase_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify Firebase ID token and return user info."""
    if not db:
        return None
    try:
        from firebase_admin import auth
        decoded = auth.verify_id_token(token)
        uid = decoded.get("uid")
        user_doc = db.collection("users").document(uid).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            user_data["uid"] = uid
            user_data["email"] = decoded.get("email")
            return user_data
        return {"uid": uid, "email": decoded.get("email"), "tier": "observer"}
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None


def require_firebase_auth(f):
    """Decorator to require Firebase auth token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing authorization token"}), 401
        token = auth_header[7:]
        user = verify_firebase_token(token)
        if not user:
            return jsonify({"error": "Invalid or expired token"}), 401
        request.user = user
        return f(*args, **kwargs)
    return decorated


def generate_api_key() -> str:
    """Generate a secure API key."""
    return f"hprl_{secrets.token_urlsafe(32)}"


# ============================================================================
# Signal Helpers
# ============================================================================

def get_latest_signals(limit: int = 10, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Get the latest signals from the JSONL file."""
    signals = []
    try:
        if not os.path.exists(SIGNALS_FILE):
            return []

        # Read last N lines efficiently
        with open(SIGNALS_FILE, "rb") as f:
            # Go to end
            f.seek(0, 2)
            file_size = f.tell()

            # Read last 50KB (should be enough for recent signals)
            read_size = min(50000, file_size)
            f.seek(max(0, file_size - read_size))

            lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")

        # Parse lines in reverse order
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                signal = json.loads(line)
                if symbols and signal.get("symbol") not in symbols:
                    continue
                signals.append(signal)
                if len(signals) >= limit:
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"Error reading signals: {e}")

    return signals


def get_current_signals() -> Dict[str, Dict[str, Any]]:
    """Get the most recent signal for each symbol."""
    all_signals = get_latest_signals(limit=100)
    current = {}

    for signal in all_signals:
        symbol = signal.get("symbol")
        if symbol and symbol not in current:
            current[symbol] = signal

    return current


# ============================================================================
# Email Notifications (Resend)
# ============================================================================

def send_email_alert(to_email: str, signal: Dict[str, Any]) -> bool:
    """Send email alert via Resend API."""
    if not RESEND_API_KEY:
        print("Resend API key not configured")
        return False

    symbol = signal.get("symbol", "UNKNOWN")
    decision = signal.get("decision", "flat").upper()
    probability = signal.get("probability_up", 0)
    timestamp = signal.get("timestamp", "")

    # Determine signal color/emoji
    if decision == "LONG":
        emoji = "🟢"
        color = "#10B981"
    elif decision == "SHORT":
        emoji = "🔴"
        color = "#EF4444"
    else:
        emoji = "⚪"
        color = "#6B7280"

    confidence = probability * 100 if decision == "LONG" else (1 - probability) * 100 if decision == "SHORT" else 50

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #fff; padding: 40px; }}
            .container {{ max-width: 500px; margin: 0 auto; background: #111118; border-radius: 16px; padding: 32px; border: 1px solid #222; }}
            .header {{ text-align: center; margin-bottom: 24px; }}
            .logo {{ font-size: 24px; font-weight: 700; color: #7C3AED; }}
            .signal-box {{ background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(34,211,238,0.05)); border: 1px solid {color}; border-radius: 12px; padding: 24px; text-align: center; }}
            .symbol {{ font-size: 32px; font-weight: 700; margin-bottom: 8px; }}
            .decision {{ font-size: 24px; font-weight: 600; color: {color}; margin-bottom: 16px; }}
            .confidence {{ font-size: 14px; color: #9CA3AF; }}
            .timestamp {{ font-size: 12px; color: #6B7280; margin-top: 24px; text-align: center; }}
            .footer {{ margin-top: 24px; text-align: center; font-size: 12px; color: #6B7280; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">HyprL</div>
                <p style="color: #9CA3AF; margin: 8px 0 0;">Trading Signal Alert</p>
            </div>
            <div class="signal-box">
                <div class="symbol">{emoji} {symbol}</div>
                <div class="decision">{decision}</div>
                <div class="confidence">Confidence: {confidence:.0f}%</div>
            </div>
            <div class="timestamp">Generated: {timestamp}</div>
            <div class="footer">
                <a href="https://hyprlcore.com/dashboard.html" style="color: #7C3AED;">View Dashboard</a>
            </div>
        </div>
    </body>
    </html>
    """

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "from": "HyprL Alerts <alerts@hyprlcore.com>",
                "to": [to_email],
                "subject": f"{emoji} {symbol}: {decision} Signal",
                "html": html_content
            },
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Email send error: {e}")
        return False


# ============================================================================
# Discord Notifications
# ============================================================================

def send_discord_alert(webhook_url: str, signal: Dict[str, Any]) -> bool:
    """Send Discord webhook notification."""
    if not webhook_url:
        return False

    symbol = signal.get("symbol", "UNKNOWN")
    decision = signal.get("decision", "flat").upper()
    probability = signal.get("probability_up", 0)
    timestamp = signal.get("timestamp", "")

    # Determine color
    if decision == "LONG":
        color = 0x10B981  # Green
        emoji = ":green_circle:"
    elif decision == "SHORT":
        color = 0xEF4444  # Red
        emoji = ":red_circle:"
    else:
        color = 0x6B7280  # Gray
        emoji = ":white_circle:"

    confidence = probability * 100 if decision == "LONG" else (1 - probability) * 100 if decision == "SHORT" else 50

    embed = {
        "title": f"{emoji} {symbol} Signal",
        "description": f"**{decision}**",
        "color": color,
        "fields": [
            {"name": "Confidence", "value": f"{confidence:.1f}%", "inline": True},
            {"name": "Symbol", "value": symbol, "inline": True}
        ],
        "timestamp": timestamp,
        "footer": {"text": "HyprL Trading Signals"}
    }

    try:
        response = requests.post(
            webhook_url,
            json={"embeds": [embed]},
            timeout=10
        )
        return response.status_code in [200, 204]
    except Exception as e:
        print(f"Discord webhook error: {e}")
        return False


# ============================================================================
# User Webhook Notifications
# ============================================================================

def send_user_webhook(webhook_url: str, webhook_secret: str, signal: Dict[str, Any]) -> bool:
    """Send webhook to user's configured URL with signature."""
    if not webhook_url:
        return False

    payload = {
        "event": "signal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": signal
    }

    payload_json = json.dumps(payload, separators=(",", ":"))

    # Create signature
    signature = hmac.new(
        webhook_secret.encode(),
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()

    try:
        response = requests.post(
            webhook_url,
            headers={
                "Content-Type": "application/json",
                "X-HyprL-Signature": f"sha256={signature}",
                "X-HyprL-Event": "signal"
            },
            data=payload_json,
            timeout=10
        )
        return response.status_code in [200, 201, 202, 204]
    except Exception as e:
        print(f"User webhook error: {e}")
        return False


# ============================================================================
# API Endpoints
# ============================================================================

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "notifications"})


# --- Signals API (Pro+) ---

@app.route("/api/signals", methods=["GET"])
@require_api_key(allowed_tiers=["pro", "team", "enterprise"])
def get_signals():
    """Get latest trading signals. Requires Pro tier."""
    limit = min(int(request.args.get("limit", 10)), 100)
    symbols = request.args.get("symbols", "").upper().split(",") if request.args.get("symbols") else None
    symbols = [s.strip() for s in symbols if s.strip()] if symbols else None

    signals = get_latest_signals(limit=limit, symbols=symbols)

    return jsonify({
        "success": True,
        "count": len(signals),
        "signals": signals
    })


@app.route("/api/signals/current", methods=["GET"])
@require_api_key(allowed_tiers=["pro", "team", "enterprise"])
def get_current_signals_api():
    """Get current signal for each symbol. Requires Pro tier."""
    current = get_current_signals()

    return jsonify({
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signals": current
    })


# --- API Key Management ---

@app.route("/api/keys/current", methods=["GET"])
@require_firebase_auth
def get_current_api_key():
    """Get user's current API key."""
    user = request.user
    api_key = user.get("apiKey")
    if api_key:
        return jsonify({"key": api_key})
    return jsonify({"key": None})


@app.route("/api/keys/generate", methods=["POST"])
@require_firebase_auth
def generate_user_api_key():
    """Generate a new API key for user."""
    user = request.user
    uid = user.get("uid")
    tier = user.get("tier", "observer")

    if tier not in ["pro", "team", "enterprise"]:
        return jsonify({"error": "API keys require Pro tier or higher"}), 403

    try:
        api_key = generate_api_key()
        db.collection("users").document(uid).update({
            "apiKey": api_key,
            "apiKeyCreated": datetime.now(timezone.utc).isoformat()
        })
        return jsonify({"success": True, "key": api_key})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/keys/revoke", methods=["POST"])
@require_firebase_auth
def revoke_user_api_key():
    """Revoke user's API key."""
    user = request.user
    uid = user.get("uid")

    try:
        db.collection("users").document(uid).update({
            "apiKey": firestore.DELETE_FIELD,
            "apiKeyCreated": firestore.DELETE_FIELD
        })
        return jsonify({"success": True, "message": "API key revoked"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Webhook Configuration ---

@app.route("/api/webhooks", methods=["POST"])
@require_firebase_auth
def configure_webhook():
    """Configure user's webhook URL."""
    user = request.user
    uid = user.get("uid")
    tier = user.get("tier", "observer")
    data = request.get_json() or {}
    webhook_url = data.get("url", "").strip()

    if tier not in ["pro", "team", "enterprise"]:
        return jsonify({"error": "Webhooks require Pro tier or higher"}), 403

    try:
        # Generate webhook secret if setting URL
        webhook_secret = f"whsec_{secrets.token_urlsafe(32)}" if webhook_url else None

        update_data = {
            "webhookUrl": webhook_url if webhook_url else firestore.DELETE_FIELD,
            "webhookSecret": webhook_secret if webhook_secret else firestore.DELETE_FIELD,
            "webhookEnabled": bool(webhook_url)
        }

        db.collection("users").document(uid).update(update_data)

        return jsonify({
            "success": True,
            "url": webhook_url,
            "secret": webhook_secret
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/webhooks/test", methods=["POST"])
@require_firebase_auth
def test_webhook():
    """Send a test webhook to user's URL."""
    user = request.user
    webhook_url = user.get("webhookUrl")
    webhook_secret = user.get("webhookSecret", "")

    if not webhook_url:
        return jsonify({"error": "No webhook URL configured"}), 400

    test_signal = {
        "symbol": "TEST",
        "decision": "long",
        "probability_up": 0.85,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test": True
    }

    try:
        success = send_user_webhook(webhook_url, webhook_secret, test_signal)
        return jsonify({
            "success": success,
            "message": "Test webhook sent" if success else "Webhook delivery failed"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Email Preferences ---

@app.route("/api/notifications/email", methods=["POST"])
@require_firebase_auth
def configure_email_notifications():
    """Configure email notification preferences."""
    user = request.user
    uid = user.get("uid")
    tier = user.get("tier", "observer")
    data = request.get_json() or {}
    enabled = data.get("enabled", False)

    if tier not in ["starter", "pro", "team", "enterprise"]:
        return jsonify({"error": "Email alerts require Starter tier or higher"}), 403

    try:
        db.collection("users").document(uid).update({
            "notifications.email": enabled
        })
        return jsonify({"success": True, "enabled": enabled})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Discord Configuration ---

@app.route("/api/notifications/discord", methods=["POST"])
@require_firebase_auth
def configure_discord_notifications():
    """Configure user's personal Discord webhook."""
    user = request.user
    uid = user.get("uid")
    tier = user.get("tier", "observer")
    data = request.get_json() or {}
    webhook_url = data.get("webhookUrl", "").strip()

    if tier not in ["pro", "team", "enterprise"]:
        return jsonify({"error": "Discord alerts require Pro tier or higher"}), 403

    try:
        db.collection("users").document(uid).update({
            "notifications.discordWebhook": webhook_url if webhook_url else firestore.DELETE_FIELD
        })
        return jsonify({"success": True, "configured": bool(webhook_url)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/notifications/discord/test", methods=["POST"])
def test_discord():
    """Send a test Discord notification."""
    data = request.get_json() or {}
    uid = data.get("uid")

    if not uid:
        return jsonify({"error": "Missing uid"}), 400

    try:
        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404

        user_data = user_doc.to_dict()
        webhook_url = user_data.get("discordWebhook")

        if not webhook_url:
            return jsonify({"error": "No Discord webhook configured"}), 400

        test_signal = {
            "symbol": "TEST",
            "decision": "long",
            "probability_up": 0.85,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        success = send_discord_alert(webhook_url, test_signal)

        return jsonify({
            "success": success,
            "message": "Test notification sent" if success else "Discord notification failed"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Internal: Broadcast signal to all subscribers ---

@app.route("/api/internal/broadcast", methods=["POST"])
def broadcast_signal():
    """
    Internal endpoint to broadcast a new signal to all subscribers.
    Called by the signal generator when a new signal is produced.
    Requires internal auth header.
    """
    internal_key = request.headers.get("X-Internal-Key")
    expected_key = os.environ.get("INTERNAL_API_KEY", "")

    if not expected_key or internal_key != expected_key:
        return jsonify({"error": "Unauthorized"}), 401

    signal = request.get_json()
    if not signal:
        return jsonify({"error": "Missing signal data"}), 400

    # Skip flat signals for notifications
    if signal.get("decision") == "flat":
        return jsonify({"success": True, "skipped": True, "reason": "flat signal"})

    results = {
        "email_sent": 0,
        "email_failed": 0,
        "discord_sent": 0,
        "discord_failed": 0,
        "webhook_sent": 0,
        "webhook_failed": 0
    }

    try:
        # Get all users with notifications enabled
        users_ref = db.collection("users")

        # Email notifications (Starter+)
        email_users = users_ref.where("emailAlertsEnabled", "==", True).stream()
        for user_doc in email_users:
            user = user_doc.to_dict()
            email = user.get("emailAlertsAddress") or user.get("email")
            if email:
                if send_email_alert(email, signal):
                    results["email_sent"] += 1
                else:
                    results["email_failed"] += 1

        # Discord notifications (Pro+)
        discord_users = users_ref.where("discordEnabled", "==", True).stream()
        for user_doc in discord_users:
            user = user_doc.to_dict()
            webhook = user.get("discordWebhook")
            if webhook:
                if send_discord_alert(webhook, signal):
                    results["discord_sent"] += 1
                else:
                    results["discord_failed"] += 1

        # User webhooks (Pro+)
        webhook_users = users_ref.where("webhookEnabled", "==", True).stream()
        for user_doc in webhook_users:
            user = user_doc.to_dict()
            url = user.get("webhookUrl")
            secret = user.get("webhookSecret", "")
            if url:
                if send_user_webhook(url, secret, signal):
                    results["webhook_sent"] += 1
                else:
                    results["webhook_failed"] += 1

    except Exception as e:
        print(f"Broadcast error: {e}")
        return jsonify({"error": str(e), "results": results}), 500

    return jsonify({"success": True, "results": results})


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8082))
    app.run(host="0.0.0.0", port=port, debug=False)
