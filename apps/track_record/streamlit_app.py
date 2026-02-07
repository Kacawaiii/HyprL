import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import sys
import numpy as np

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from apps.track_record.lib.constants import TRACK_RECORD_DIR, STYLE_CSS_PATH
from apps.track_record.lib.io import load_track_record_json, load_track_record_md, check_artifacts_exist
from apps.track_record.lib.metrics import extract_kpis
from apps.track_record.lib.auth import get_current_user_level

# Page Config
st.set_page_config(
    page_title="HyprL Track Record",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load and inject CSS
def load_css():
    if STYLE_CSS_PATH.exists():
        css = STYLE_CSS_PATH.read_text()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# Data path
DATA_ROOT = TRACK_RECORD_DIR


def create_equity_chart(curve_data: dict, key: str = "equity"):
    """Create beautiful equity curve with area fill."""
    if not curve_data or "equity" not in curve_data:
        return None

    timestamps = curve_data.get("timestamps", [])
    equity = curve_data.get("equity", [])

    if not timestamps or not equity:
        return None

    # Convert timestamps
    if isinstance(timestamps[0], (int, float)):
        dates = pd.to_datetime(timestamps, unit='s')
    else:
        dates = pd.to_datetime(timestamps)

    df = pd.DataFrame({"Date": dates, "Equity": equity})
    df = df.sort_values("Date")

    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Equity"],
        mode='lines',
        name='Equity',
        line=dict(color='#22D3EE', width=2),
        fill='tozeroy',
        fillcolor='rgba(34, 211, 238, 0.1)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Equity: $%{y:,.2f}<extra></extra>'
    ))

    # Starting line
    start_equity = equity[0] if equity else 100000
    fig.add_hline(y=start_equity, line_dash="dash", line_color="rgba(148, 163, 184, 0.3)",
                  annotation_text=f"Start: ${start_equity:,.0f}", annotation_position="right")

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        height=400,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8'),
            showline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8'),
            tickprefix="$",
            showline=False
        ),
        hovermode="x unified",
        showlegend=False
    )

    return fig


def create_pnl_bars(curve_data: dict, key: str = "pnl"):
    """Create daily P&L bar chart with red/green colors."""
    if not curve_data or "equity" not in curve_data:
        return None

    timestamps = curve_data.get("timestamps", [])
    equity = curve_data.get("equity", [])

    if not timestamps or not equity or len(equity) < 2:
        return None

    if isinstance(timestamps[0], (int, float)):
        dates = pd.to_datetime(timestamps, unit='s')
    else:
        dates = pd.to_datetime(timestamps)

    df = pd.DataFrame({"Date": dates, "Equity": equity})
    df = df.sort_values("Date")
    df["PnL"] = df["Equity"].diff().fillna(0)
    df["Color"] = df["PnL"].apply(lambda x: '#34D399' if x >= 0 else '#ef4444')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["PnL"],
        marker_color=df["Color"].tolist(),
        opacity=0.9,
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>P&L: $%{y:,.2f}<extra></extra>'
    ))

    fig.add_hline(y=0, line_color="rgba(148, 163, 184, 0.5)")

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        height=250,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8'),
            tickprefix="$"
        ),
        showlegend=False
    )

    return fig


def create_monthly_heatmap(curve_data: dict):
    """Create monthly returns heatmap."""
    if not curve_data or "equity" not in curve_data:
        return None

    timestamps = curve_data.get("timestamps", [])
    equity = curve_data.get("equity", [])

    if not timestamps or not equity or len(equity) < 2:
        return None

    if isinstance(timestamps[0], (int, float)):
        dates = pd.to_datetime(timestamps, unit='s')
    else:
        dates = pd.to_datetime(timestamps)

    df = pd.DataFrame({"Date": dates, "Equity": equity})
    df = df.sort_values("Date")
    df["Return"] = df["Equity"].pct_change() * 100
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    monthly = df.groupby(["Year", "Month"])["Return"].sum().reset_index()

    if len(monthly) < 2:
        return None

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    pivot = monthly.pivot(index="Year", columns="Month", values="Return").fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[months[i-1] for i in pivot.columns],
        y=pivot.index,
        colorscale=[
            [0, '#ef4444'],
            [0.5, '#1e293b'],
            [1, '#34D399']
        ],
        zmid=0,
        hovertemplate='<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        height=200,
        xaxis=dict(tickfont=dict(color='#94a3b8')),
        yaxis=dict(tickfont=dict(color='#94a3b8'))
    )

    return fig


def create_drawdown_chart(curve_data: dict):
    """Create drawdown chart."""
    if not curve_data or "equity" not in curve_data:
        return None

    timestamps = curve_data.get("timestamps", [])
    equity = curve_data.get("equity", [])

    if not timestamps or not equity:
        return None

    if isinstance(timestamps[0], (int, float)):
        dates = pd.to_datetime(timestamps, unit='s')
    else:
        dates = pd.to_datetime(timestamps)

    df = pd.DataFrame({"Date": dates, "Equity": equity})
    df = df.sort_values("Date")

    # Calculate drawdown
    df["Peak"] = df["Equity"].cummax()
    df["Drawdown"] = (df["Equity"] - df["Peak"]) / df["Peak"] * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Drawdown"],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color='#ef4444', width=1),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        height=200,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8'),
            ticksuffix="%"
        ),
        showlegend=False
    )

    return fig


def main():
    # Header
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown("""
        <h1 style="margin-bottom: 0;">HyprL <span class="gradient-text-brand">Track Record</span></h1>
        """, unsafe_allow_html=True)
        st.caption("Live Automated Trading Performance • ML-Powered • Alpaca Markets")

    with col2:
        st.markdown("""
        <div style="text-align: right; padding-top: 1rem;">
            <span class="status-live">LIVE</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Check data
    if not check_artifacts_exist(DATA_ROOT):
        st.warning("No track record data found. Run the snapshot script first.")
        return

    data = load_track_record_json(DATA_ROOT)
    if not data:
        st.error("Failed to load track record data.")
        return

    kpis = extract_kpis(data)
    curve = data.get("equity_curve", {})

    # Calculate additional stats
    equity_list = curve.get("equity", [])
    if len(equity_list) > 1:
        pnl_list = [equity_list[i] - equity_list[i-1] for i in range(1, len(equity_list))]
        winning_days = sum(1 for p in pnl_list if p > 0)
        losing_days = sum(1 for p in pnl_list if p < 0)
        best_day = max(pnl_list) if pnl_list else 0
        worst_day = min(pnl_list) if pnl_list else 0
        avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
        avg_loss = np.mean([p for p in pnl_list if p < 0]) if any(p < 0 for p in pnl_list) else 0
    else:
        winning_days = losing_days = 0
        best_day = worst_day = avg_win = avg_loss = 0

    # KPI Row 1 - Main metrics
    st.markdown("### Performance Metrics")

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        delta_color = "normal" if kpis['return_pct'] >= 0 else "inverse"
        st.metric("Total Return", f"{kpis['return_pct']:+.2f}%", delta=f"Since inception", delta_color="off")

    with k2:
        st.metric("Current Equity", f"${kpis['equity']:,.0f}")

    with k3:
        st.metric("Sharpe Ratio", f"{kpis['sharpe']:.2f}")

    with k4:
        st.metric("Win Rate", f"{kpis['win_rate']*100:.1f}%", delta=f"{kpis['n_trades']} trades", delta_color="off")

    with k5:
        st.metric("Max Drawdown", f"{kpis['max_drawdown_pct']:.2f}%")

    st.markdown("")

    # KPI Row 2 - Additional metrics
    k6, k7, k8, k9, k10 = st.columns(5)

    with k6:
        st.metric("Profit Factor", f"{kpis['profit_factor']:.2f}")

    with k7:
        st.metric("Winning Days", f"{winning_days}", delta=f"+${best_day:,.0f} best", delta_color="off")

    with k8:
        st.metric("Losing Days", f"{losing_days}", delta=f"${worst_day:,.0f} worst", delta_color="off")

    with k9:
        st.metric("Avg Win", f"${avg_win:,.0f}")

    with k10:
        st.metric("Avg Loss", f"${abs(avg_loss):,.0f}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Main Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity Curve", "📊 Daily P&L", "🗓️ Monthly Returns", "📋 Analysis"])

    with tab1:
        equity_fig = create_equity_chart(curve, "main_equity")
        if equity_fig:
            st.plotly_chart(equity_fig, use_container_width=True, key="equity_main_chart")

        # Drawdown below
        st.markdown("#### Drawdown")
        dd_fig = create_drawdown_chart(curve)
        if dd_fig:
            st.plotly_chart(dd_fig, use_container_width=True, key="drawdown_chart")

    with tab2:
        pnl_fig = create_pnl_bars(curve, "main_pnl")
        if pnl_fig:
            st.plotly_chart(pnl_fig, use_container_width=True, key="pnl_main_chart")

        # Stats row
        if equity_list and len(equity_list) > 1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                streak_wins = 0
                current_streak = 0
                for p in pnl_list:
                    if p > 0:
                        current_streak += 1
                        streak_wins = max(streak_wins, current_streak)
                    else:
                        current_streak = 0
                st.metric("Best Win Streak", f"{streak_wins} days")
            with col2:
                streak_losses = 0
                current_streak = 0
                for p in pnl_list:
                    if p < 0:
                        current_streak += 1
                        streak_losses = max(streak_losses, current_streak)
                    else:
                        current_streak = 0
                st.metric("Worst Loss Streak", f"{streak_losses} days")
            with col3:
                volatility = np.std(pnl_list) if pnl_list else 0
                st.metric("Daily Volatility", f"${volatility:,.0f}")
            with col4:
                total_pnl = sum(pnl_list) if pnl_list else 0
                st.metric("Total P&L", f"${total_pnl:,.0f}")

    with tab3:
        heatmap_fig = create_monthly_heatmap(curve)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True, key="monthly_heatmap")
        else:
            st.info("Not enough data for monthly heatmap.")

        # Monthly table
        if curve and "equity" in curve:
            timestamps = curve.get("timestamps", [])
            equity = curve.get("equity", [])
            if timestamps and equity:
                if isinstance(timestamps[0], (int, float)):
                    dates = pd.to_datetime(timestamps, unit='s')
                else:
                    dates = pd.to_datetime(timestamps)
                df = pd.DataFrame({"Date": dates, "Equity": equity})
                df["Return"] = df["Equity"].pct_change() * 100
                df["Month"] = df["Date"].dt.strftime("%Y-%m")
                monthly_returns = df.groupby("Month")["Return"].sum().reset_index()
                monthly_returns.columns = ["Month", "Return %"]
                monthly_returns["Return %"] = monthly_returns["Return %"].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(monthly_returns, use_container_width=True, hide_index=True)

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Risk Metrics")
            risk_data = {
                "Metric": ["Max Drawdown", "Sharpe Ratio", "Profit Factor", "Win Rate", "Risk/Reward"],
                "Value": [
                    f"{kpis['max_drawdown_pct']:.2f}%",
                    f"{kpis['sharpe']:.2f}",
                    f"{kpis['profit_factor']:.2f}",
                    f"{kpis['win_rate']*100:.1f}%",
                    f"{abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A"
                ],
                "Status": ["Good" if kpis['max_drawdown_pct'] > -15 else "Warning",
                          "Good" if kpis['sharpe'] > 1 else "Moderate",
                          "Good" if kpis['profit_factor'] > 1.5 else "Moderate",
                          "Good" if kpis['win_rate'] > 0.5 else "Moderate",
                          "Good" if avg_loss != 0 and abs(avg_win/avg_loss) > 1 else "Moderate"]
            }
            st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### System Info")
            st.markdown(f"""
            - **Mode:** Paper Trading
            - **Last Update:** {kpis['asof']}
            - **Total Trades:** {kpis['n_trades']}
            - **Strategy:** XGBoost ML + Technical
            - **Timeframe:** 1H
            - **Symbols:** NVDA, MSFT, QQQ + 13 more
            """)

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
        <p>Past performance does not guarantee future results. Paper trading only.</p>
        <p>Built with Python, XGBoost, and Alpaca API</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
