# HyprL — Quantitative Trading Intelligence

**Algorithmic trading strategy with transparent performance tracking**

[Join Beta Waitlist] [View Live Track Record]

---

## The Challenge

Retail traders lack:
- ✗ **Transparent performance data** (most services hide real results)
- ✗ **Reproducible strategies** (black boxes, no audit trail)
- ✗ **Risk-adjusted execution** (overleveraged, no position sizing)
- ✗ **Real-time decision support** (delayed signals, manual execution)

**Result:** 90% of retail algorithmic traders lose money.

---

## HyprL Solution

A **fully audited**, **risk-first** quantitative trading system for US equities (1h timeframe).

### What You Get

#### 📊 Live Track Record Dashboard
- Real-time equity curve
- Transparent P&L (no cherry-picking)
- Risk metrics (MaxDD, Sharpe, Win Rate)
- Full trade history

**Current Performance (Paper Trading):**
- Equity: $12,345.67
- Return: +23.46%
- Max Drawdown: -2.50%
- Profit Factor: 2.23
- Win Rate: 73.42%

*⚠️ Disclaimer: Paper trading results. Past performance is not indicative of future results.*

---

#### 🎯 Signal Intelligence
- **NVDA, MSFT, QQQ** coverage (1h bars)
- XGBoost probability models (19 features)
- Kelly-sized positions (1% base risk)
- ATR-based stop-loss (1.5x)
- Trailing profit capture

**Entry Logic:**
- Long/Short thresholds per ticker
- Multi-timeframe confirmation
- Risk guards (max DD, consecutive losses)

---

#### 🔐 API Access (Pro)
- Real-time signal stream (webhooks)
- Historical backtest data
- Custom portfolio weights
- Position sizing calculator

**Example:**
```json
{
  "ticker": "NVDA",
  "signal": "LONG",
  "probability": 0.87,
  "entry_price": 450.00,
  "stop_loss": 442.50,
  "position_size": 22,
  "risk_pct": 1.0
}
```

---

#### ⚡ Execution Bridge (Coming Soon)
- Alpaca broker integration
- Automated order routing
- Kill-switch protection
- Trade logging + reconciliation

---

## Pricing

### 🎯 Beta Early Access
**€19/month**
- Live track record dashboard
- Discord community
- Weekly strategy updates
- **Limited to 30 spots**

[Join Beta] ← **First 10 get 50% off first month**

---

### 🚀 Starter (Public Launch)
**€39/month**
- Everything in Beta
- Email alerts on signals
- Monthly performance reports
- Full trade history export

---

### 💎 Pro (Q1 2026)
**€99/month**
- Everything in Starter
- Real-time API access
- Custom webhook integrations
- Priority support
- Advanced analytics

---

## Why HyprL is Different

| Feature | HyprL | Typical "Signal Services" |
|---------|-------|---------------------------|
| **Live Track Record** | ✅ Public, auditable | ❌ Hidden or fake |
| **Risk Management** | ✅ Kelly sizing + guards | ❌ Overleveraged |
| **Code Audit** | ✅ Full reproducibility | ❌ Black box |
| **Execution** | ✅ Broker integration | ❌ Manual only |
| **Transparency** | ✅ All trades logged | ❌ Cherry-picked winners |

---

## Technology Stack

- **Models:** XGBoost (scikit-learn)
- **Features:** 19 technical indicators (SMA, RSI, ATR, volatility)
- **Backtest:** 474 trades, 1+ year validation
- **Execution:** Alpaca Trading API (paper + live)
- **Monitoring:** FastAPI + Streamlit dashboards

**Open Core Philosophy:**
- Strategy logic: proprietary
- Execution framework: open-source ready
- Research: reproducible (freeze tags + SHA256 bundles)

---

## Roadmap

### ✅ Phase 0 (Now)
- Core v3 strategy frozen
- Paper trading live
- Track record portal online

### 🔄 Phase 1 (Q1 2026)
- 60+ days paper track record
- Beta launch (30 users)
- API v1 release

### 📈 Phase 2 (Q2 2026)
- Live micro-capital ($500 real account)
- Pro tier launch
- Multi-ticker portfolio expansion

### 🌐 Phase 3 (Q3 2026)
- White-label partnerships
- Institutional API access
- Managed accounts (qualified investors)

---

## Legal & Risk Disclosure

### ⚠️ Important Disclaimers

**HyprL is NOT:**
- ❌ Financial advice (we are not registered advisors)
- ❌ A guarantee of profit
- ❌ Suitable for everyone (trading involves risk of loss)

**You should:**
- ✅ Understand trading risks before subscribing
- ✅ Only trade with capital you can afford to lose
- ✅ Consult a licensed financial advisor
- ✅ Read our full [Terms of Service] and [Risk Disclosure]

**Performance Disclaimers:**
- Past performance ≠ future results
- Paper trading results may not reflect live execution
- Actual results may vary due to slippage, commissions, market conditions

**Jurisdiction:**
- HyprL operates under French law
- Service may not be available in all countries
- US residents: Check FINRA/SEC regulations

---

## FAQ

### Is HyprL a registered investment advisor?
**No.** HyprL provides trading signals and analytics tools for educational purposes. We do not manage funds or provide personalized financial advice.

### Can I automate execution?
**Yes (Pro tier).** Our Alpaca bridge allows automated order routing. You maintain full control over your brokerage account.

### What's the minimum capital?
We recommend **$5,000+** for proper position sizing. Smaller accounts may not be able to follow signals effectively.

### Do you guarantee profits?
**No.** All trading involves risk. Our track record is transparent, but past performance does not guarantee future results.

### Can I cancel anytime?
**Yes.** All subscriptions are month-to-month with no lock-in.

### What brokers do you support?
Currently **Alpaca Markets** (paper + live). Interactive Brokers support planned for Q2 2026.

---

## Join Beta Waitlist

**Limited to 30 early access spots**

[Email Sign-up Form]
- Name
- Email
- Trading experience (beginner/intermediate/advanced)
- Capital range (<$5k / $5k-$20k / $20k-$100k / $100k+)
- Primary interest (Dashboard / API / Execution automation)

**What happens next:**
1. You'll receive an invite within 7 days
2. Beta access: €19/mo (first 10 get 50% off)
3. Exclusive Discord community
4. Direct feedback channel to development team

[Submit Beta Request]

---

## Contact

**Questions?** Email: [email protected]

**Twitter/X:** [@HyprLQuant](https://twitter.com/HyprLQuant)

**GitHub:** [github.com/Kacawaiii/HyprL](https://github.com/Kacawaiii/HyprL) (execution framework only)

**Discord:** Join our beta community after signup

---

## Footer

© 2025 HyprL. All rights reserved.

[Terms of Service] | [Privacy Policy] | [Risk Disclosure] | [Refund Policy]

**Risk Warning:** Trading involves substantial risk of loss. HyprL is not a registered investment advisor. Signals are for educational purposes only. You are solely responsible for your trading decisions.

**Regulatory:** HyprL operates under French jurisdiction. This service may not be available in all countries. Check local regulations before subscribing.

---

*Built with transparency. Powered by data.*
