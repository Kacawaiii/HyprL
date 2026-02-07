import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, Optional
from plotly.subplots import make_subplots


def render_kpi_card(label: str, value: str, delta: str = None, help_text: str = None):
    """Renders a KPI card using Streamlit's native metric with custom styling."""
    # Use native st.metric which renders cleanly
    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help_text
    )


def render_equity_chart(curve_data: Dict[str, Any], chart_key: str = "equity_main"):
    """Renders the equity curve using Plotly with unique key."""
    if not curve_data or "equity" not in curve_data:
        st.warning("No equity curve data available.")
        return

    timestamps = curve_data.get("timestamps", [])
    equity = curve_data.get("equity", [])

    if not timestamps or not equity:
        st.info("Equity curve data is empty.")
        return

    df = pd.DataFrame({"Date": pd.to_datetime(timestamps), "Equity": equity})
    df = df.sort_values("Date")

    # Calculate daily returns for the bars
    df["PnL"] = df["Equity"].diff().fillna(0)
    df["Color"] = df["PnL"].apply(lambda x: "#34D399" if x >= 0 else "#F87171")

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Daily P&L")
    )

    # Main Equity Line (top)
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#22D3EE", width=2),
            fill='tozeroy',
            fillcolor='rgba(34, 211, 238, 0.1)'
        ),
        row=1, col=1
    )

    # Daily P&L Bars (bottom) - Red/Green
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["PnL"],
            name="Daily P&L",
            marker_color=df["Color"].tolist(),
            opacity=0.8
        ),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        height=550,
        hovermode="x unified",
        showlegend=False,
        font=dict(color="#94a3b8")
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(148, 163, 184, 0.1)',
        tickfont=dict(color='#94a3b8')
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(148, 163, 184, 0.1)',
        tickfont=dict(color='#94a3b8'),
        tickprefix="$",
        row=1, col=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(148, 163, 184, 0.1)',
        tickfont=dict(color='#94a3b8'),
        tickprefix="$",
        row=2, col=1
    )

    # Update subplot titles color
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#f0f4f8', size=14)

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_pnl_bars(curve_data: Dict[str, Any], chart_key: str = "pnl_bars"):
    """Renders only the daily P&L bars (red/green)."""
    if not curve_data or "equity" not in curve_data:
        st.warning("No equity data available for P&L bars.")
        return

    timestamps = curve_data.get("timestamps", [])
    equity = curve_data.get("equity", [])

    if not timestamps or not equity or len(equity) < 2:
        st.info("Not enough data for P&L bars.")
        return

    df = pd.DataFrame({"Date": pd.to_datetime(timestamps), "Equity": equity})
    df = df.sort_values("Date")
    df["PnL"] = df["Equity"].diff().fillna(0)
    df["Color"] = df["PnL"].apply(lambda x: "#34D399" if x >= 0 else "#F87171")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["PnL"],
        name="Daily P&L",
        marker_color=df["Color"].tolist(),
        opacity=0.9,
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>"
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.5)")

    fig.update_layout(
        title=dict(text="Daily P&L", font=dict(color="#f0f4f8", size=18)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8'),
            tickprefix="$"
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=350,
        hovermode="x unified",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_equity_line_only(curve_data: Dict[str, Any], chart_key: str = "equity_line"):
    """Renders just the equity line chart without P&L bars."""
    if not curve_data or "equity" not in curve_data:
        st.warning("No equity curve data available.")
        return

    timestamps = curve_data.get("timestamps", [])
    equity = curve_data.get("equity", [])

    if not timestamps or not equity:
        st.info("Equity curve data is empty.")
        return

    df = pd.DataFrame({"Date": pd.to_datetime(timestamps), "Equity": equity})
    df = df.sort_values("Date")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Equity"],
        mode="lines",
        name="Equity",
        line=dict(color="#22D3EE", width=3),
        fill='tozeroy',
        fillcolor='rgba(34, 211, 238, 0.1)',
        hovertemplate="<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Equity Curve", font=dict(color="#f0f4f8", size=18)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8'),
            tickprefix="$"
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)
