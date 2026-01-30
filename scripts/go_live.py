#!/usr/bin/env python3
"""
HYPRL GO LIVE - Démarrage Trading Réel
======================================
Script pour passer en production avec capital réel.

CHECKLIST AVANT LANCEMENT:
1. ✅ Walk-Forward Validation passée
2. ✅ Slippage intégré aux backtests
3. ⬜ API Keys Alpaca configurées
4. ⬜ Capital déposé sur compte Alpaca
5. ⬜ Kill switch testé

Usage:
    # Vérification pré-lancement
    python scripts/go_live.py --check

    # Démarrage avec capital minimal
    python scripts/go_live.py --start --capital 2000

    # Monitoring
    python scripts/go_live.py --status
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check for required packages
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from hyprl.monitoring import (
    get_health_checker,
    get_risk_monitor,
    get_notifier,
    get_journal,
)


class GoLiveChecker:
    """Pre-launch checks for live trading."""

    def __init__(self):
        self.checks = {}
        self.warnings = []
        self.errors = []

    def check_all(self) -> bool:
        """Run all pre-launch checks."""
        print("\n" + "=" * 60)
        print("HYPRL GO LIVE - PRÉ-LANCEMENT")
        print("=" * 60 + "\n")

        self._check_environment()
        self._check_alpaca_connection()
        self._check_kill_switch()
        self._check_validation_results()
        self._check_monitoring()

        # Print results
        print("\n" + "-" * 60)
        print("RÉSULTATS DES VÉRIFICATIONS")
        print("-" * 60)

        all_passed = True
        for name, result in self.checks.items():
            status = "✅" if result['passed'] else "❌"
            print(f"{status} {name}: {result['message']}")
            if not result['passed']:
                all_passed = False

        if self.warnings:
            print("\n⚠️ AVERTISSEMENTS:")
            for w in self.warnings:
                print(f"  - {w}")

        if self.errors:
            print("\n❌ ERREURS:")
            for e in self.errors:
                print(f"  - {e}")

        print("\n" + "=" * 60)
        if all_passed:
            print("✅ TOUS LES CHECKS PASSÉS - PRÊT POUR LIVE")
        else:
            print("❌ CHECKS ÉCHOUÉS - CORRIGER AVANT DE CONTINUER")
        print("=" * 60)

        return all_passed

    def _check_environment(self):
        """Check environment variables."""
        api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("APCA_API_BASE_URL", "")

        has_keys = bool(api_key and api_secret)
        is_live = "paper" not in base_url.lower() if base_url else False

        self.checks['environment'] = {
            'passed': has_keys,
            'message': f"API Keys {'configurées' if has_keys else 'MANQUANTES'}"
        }

        if is_live:
            self.warnings.append("Base URL pointe vers LIVE (pas paper)")

        # Check Telegram
        tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        tg_chat = os.getenv("TELEGRAM_CHAT_ID")
        if not (tg_token and tg_chat):
            self.warnings.append("Telegram non configuré (alertes désactivées)")

    def _check_alpaca_connection(self):
        """Check Alpaca API connection."""
        if not ALPACA_AVAILABLE:
            self.checks['alpaca'] = {
                'passed': False,
                'message': "Package alpaca-trade-api non installé"
            }
            return

        try:
            api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
            api_secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
            base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

            if not (api_key and api_secret):
                self.checks['alpaca'] = {
                    'passed': False,
                    'message': "API credentials manquantes"
                }
                return

            api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            account = api.get_account()

            equity = float(account.equity)
            buying_power = float(account.buying_power)
            status = account.status

            self.checks['alpaca'] = {
                'passed': status == 'ACTIVE',
                'message': f"Connecté | Equity: ${equity:,.0f} | Status: {status}"
            }

            if equity < 1000:
                self.warnings.append(f"Equity faible: ${equity:,.0f}")

        except Exception as e:
            self.checks['alpaca'] = {
                'passed': False,
                'message': f"Erreur connexion: {str(e)[:50]}"
            }

    def _check_kill_switch(self):
        """Check kill switch file."""
        kill_file = Path(".kill_switch")

        if kill_file.exists():
            self.checks['kill_switch'] = {
                'passed': False,
                'message': "Kill switch ACTIF - supprimer .kill_switch pour trader"
            }
        else:
            self.checks['kill_switch'] = {
                'passed': True,
                'message': "Kill switch inactif"
            }

    def _check_validation_results(self):
        """Check walk-forward validation results."""
        wf_file = Path("live/logs/wf_validation.json")

        if not wf_file.exists():
            self.checks['validation'] = {
                'passed': False,
                'message': "Walk-Forward validation non exécutée"
            }
            return

        try:
            with open(wf_file) as f:
                data = json.load(f)

            results = data.get('results', {})
            valid_results = {k: v for k, v in results.items() if 'error' not in v}

            if not valid_results:
                self.checks['validation'] = {
                    'passed': False,
                    'message': "Aucun résultat de validation valide"
                }
                return

            avg_oos_sharpe = sum(r['oos_sharpe'] for r in valid_results.values()) / len(valid_results)
            avg_overfit = sum(r['overfit_ratio'] for r in valid_results.values()) / len(valid_results)

            passed = avg_oos_sharpe > 0 and avg_overfit < 2.0

            self.checks['validation'] = {
                'passed': passed,
                'message': f"OOS Sharpe: {avg_oos_sharpe:.2f} | Overfit ratio: {avg_overfit:.2f}"
            }

        except Exception as e:
            self.checks['validation'] = {
                'passed': False,
                'message': f"Erreur lecture: {e}"
            }

    def _check_monitoring(self):
        """Check monitoring systems."""
        try:
            health = get_health_checker()
            journal = get_journal()
            notifier = get_notifier()

            self.checks['monitoring'] = {
                'passed': True,
                'message': f"Health, Journal, Telegram {'activé' if notifier.enabled else 'désactivé'}"
            }
        except Exception as e:
            self.checks['monitoring'] = {
                'passed': False,
                'message': f"Erreur: {e}"
            }


def start_live_trading(capital: float, dry_run: bool = False):
    """Start live trading."""
    print("\n" + "=" * 60)
    print("DÉMARRAGE TRADING LIVE")
    print("=" * 60)

    if dry_run:
        print("⚠️ MODE DRY RUN - Aucun ordre réel ne sera passé")
    else:
        print("🔴 MODE LIVE - Ordres réels!")

    print(f"Capital: ${capital:,.0f}")
    print()

    # Run pre-checks
    checker = GoLiveChecker()
    if not checker.check_all() and not dry_run:
        print("\n❌ Checks échoués. Utilisez --dry-run pour tester sans capital.")
        return False

    # Start the monitored trading script
    import subprocess

    config = "configs/runtime/strategy_mix.yaml"
    cmd = [
        sys.executable,
        "scripts/run_live_monitored.py",
        "--config", config,
        "--equity", str(capital),
    ]

    if dry_run:
        cmd.append("--dry-run")
    else:
        cmd.append("--live")

    print(f"\nLancement: {' '.join(cmd)}")
    print("-" * 60)

    # Execute
    subprocess.run(cmd)

    return True


def get_status():
    """Get current trading status."""
    print("\n" + "=" * 60)
    print("HYPRL STATUS")
    print("=" * 60 + "\n")

    # Check heartbeat
    heartbeat_file = Path("live/state/heartbeat.json")
    if heartbeat_file.exists():
        with open(heartbeat_file) as f:
            hb = json.load(f)
        print(f"Heartbeat: {hb.get('status', 'unknown')}")
        print(f"Last update: {hb.get('timestamp', 'N/A')}")
        print(f"PID: {hb.get('pid', 'N/A')}")

        stats = hb.get('stats', {})
        if stats:
            print(f"\nStats:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
    else:
        print("❌ Bot non démarré (pas de heartbeat)")

    # Check positions
    if ALPACA_AVAILABLE:
        try:
            api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
            api_secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
            base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

            if api_key and api_secret:
                api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
                account = api.get_account()
                positions = api.list_positions()

                print(f"\n--- COMPTE ALPACA ---")
                print(f"Equity: ${float(account.equity):,.2f}")
                print(f"Cash: ${float(account.cash):,.2f}")
                print(f"P/L Today: ${float(account.equity) - float(account.last_equity):+,.2f}")

                if positions:
                    print(f"\nPositions ouvertes: {len(positions)}")
                    for pos in positions:
                        pnl = float(pos.unrealized_pl)
                        print(f"  {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f} | P/L: ${pnl:+,.2f}")
                else:
                    print("\nAucune position ouverte")

        except Exception as e:
            print(f"\n⚠️ Erreur Alpaca: {e}")

    # Check journal
    journal = get_journal()
    stats = journal.get_stats()
    if 'error' not in stats:
        print(f"\n--- JOURNAL ---")
        print(f"Total trades: {stats['total_trades']}")
        print(f"Win rate: {stats['win_rate']:.1%}")
        print(f"Total P/L: ${stats['total_pnl']:,.0f}")


def main():
    parser = argparse.ArgumentParser(description="HYPRL Go Live")
    parser.add_argument("--check", action="store_true", help="Run pre-launch checks")
    parser.add_argument("--start", action="store_true", help="Start live trading")
    parser.add_argument("--status", action="store_true", help="Get current status")
    parser.add_argument("--capital", type=float, default=2000, help="Trading capital")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    args = parser.parse_args()

    if args.check:
        checker = GoLiveChecker()
        checker.check_all()

    elif args.start:
        start_live_trading(args.capital, args.dry_run)

    elif args.status:
        get_status()

    else:
        # Default: show help and run check
        parser.print_help()
        print("\n")
        checker = GoLiveChecker()
        checker.check_all()


if __name__ == "__main__":
    main()
