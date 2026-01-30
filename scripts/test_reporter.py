#!/usr/bin/env python3
"""
Test du système de reporting HyprL v1.1
========================================
Simule des trades et affiche les rapports formatés.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.monitoring.reporter import HyprLReporter


def main():
    print("=" * 70)
    print("TEST REPORTING HYPRL v1.1")
    print("=" * 70)

    # Create reporter (will print to console if no Telegram)
    reporter = HyprLReporter()

    # Initialize portfolio
    symbols = ["NVDA", "META", "MSFT", "PLTR", "TSLA", "QQQ", "SMCI", "SHOP"]
    reporter.init_portfolio(equity=10000, symbols=symbols)

    print("\n[1] Simulating trades...")

    # Simulate some trades
    trades = [
        # Symbol, direction, shares, entry, stop, tp, exit, pnl, reason
        ("NVDA", "long", 5, 180.00, 175.00, 190.00, 188.50, 42.50, "tp"),
        ("PLTR", "long", 30, 160.00, 155.00, 170.00, 172.00, 360.00, "tp"),
        ("META", "long", 3, 640.00, 620.00, 680.00, 615.00, -75.00, "stop"),
        ("TSLA", "short", 4, 450.00, 470.00, 420.00, 425.00, 100.00, "tp"),
        ("SMCI", "long", 50, 32.00, 30.00, 36.00, 35.50, 175.00, "tp"),
        ("NVDA", "long", 6, 185.00, 180.00, 195.00, 192.00, 42.00, "trailing"),
        ("QQQ", "long", 3, 615.00, 600.00, 640.00, 638.00, 69.00, "tp"),
        ("SHOP", "long", 15, 135.00, 130.00, 145.00, 128.00, -105.00, "stop"),
        ("PLTR", "long", 25, 165.00, 160.00, 175.00, 178.00, 325.00, "tp"),
        ("MSFT", "long", 4, 445.00, 435.00, 465.00, 462.00, 68.00, "tp"),
    ]

    for symbol, direction, shares, entry, stop, tp, exit_price, pnl, reason in trades:
        # Record entry
        reporter.record_trade_entry(
            symbol=symbol,
            direction=direction,
            shares=shares,
            price=entry,
            stop_price=stop,
            tp_price=tp
        )

        # Record exit
        reporter.record_trade_exit(
            symbol=symbol,
            exit_price=exit_price,
            pnl=pnl,
            exit_reason=reason
        )

        print(f"  {symbol}: {direction.upper()} {shares} @ ${entry:.2f} → ${exit_price:.2f} = ${pnl:+,.0f}")

    # Update final equity
    total_pnl = sum(t[7] for t in trades)
    reporter.update_equity(10000 + total_pnl)

    print(f"\n  Total P/L: ${total_pnl:+,.0f}")

    # Generate reports
    print("\n" + "=" * 70)
    print("[2] RAPPORT GLOBAL")
    print("=" * 70)
    print(reporter.generate_global_summary().replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "").replace("<i>", "").replace("</i>", ""))

    print("\n" + "=" * 70)
    print("[3] RÉSUMÉ PAR SYMBOLE")
    print("=" * 70)
    print(reporter.generate_all_symbols_summary().replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "").replace("<i>", "").replace("</i>", ""))

    print("\n" + "=" * 70)
    print("[4] DÉTAIL NVDA")
    print("=" * 70)
    print(reporter.generate_symbol_report("NVDA").replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "").replace("<i>", "").replace("</i>", ""))

    print("\n" + "=" * 70)
    print("[5] RAPPORT HORAIRE")
    print("=" * 70)
    print(reporter.generate_hourly_report().replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "").replace("<i>", "").replace("</i>", ""))

    # Test Telegram sending (will print if no token)
    print("\n" + "=" * 70)
    print("[6] TEST ENVOI TELEGRAM")
    print("=" * 70)

    if reporter.enabled:
        print("Telegram activé - Envoi des rapports...")
        reporter.send_full_report()
        print("✓ Rapports envoyés!")
    else:
        print("⚠️ Telegram non configuré")
        print("Pour activer, définir dans .env:")
        print("  TELEGRAM_BOT_TOKEN=xxx")
        print("  TELEGRAM_CHAT_ID=xxx")

    print("\n" + "=" * 70)
    print("TEST TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()
