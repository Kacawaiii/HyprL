import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPORT_JSON = Path("docs/reports/TRACK_RECORD.json")
LATEST_JSON = Path("docs/reports/track_record/latest.json")
DISCLAIMER_PATH = Path("docs/legal/DISCLAIMER.md")


st.set_page_config(page_title="HyprL Track Record", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg: #f6f4ef;
  --card: #ffffff;
  --ink: #1d1c1a;
  --accent: #0b6e4f;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--ink);
}
div[data-testid="stMetric"] {
  background: var(--card);
  padding: 0.75rem 1rem;
  border-radius: 10px;
  border: 1px solid #e6e1d8;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("HyprL Track Record")

if DISCLAIMER_PATH.exists():
    st.info("Disclaimer: " + DISCLAIMER_PATH.read_text(encoding="utf-8").splitlines()[0])

if not REPORT_JSON.exists() or not LATEST_JSON.exists():
    st.error("Track record artifacts not found. Run snapshot + report first.")
    st.stop()

report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
latest = json.loads(LATEST_JSON.read_text(encoding="utf-8"))

st.caption(f"Last update: {report.get('generated_ts')}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Return %", f"{report.get('return_pct', 0):.2f}")
col2.metric("Max DD %", f"{report.get('max_drawdown_pct', 0):.2f}")
col3.metric("Equity start", f"{report.get('start_equity', 0):,.2f}")
col4.metric("Equity end", f"{report.get('end_equity', 0):,.2f}")

curve = latest.get("equity_curve", {})
values = curve.get("equity", [])
labels = curve.get("timestamps", [])

if values:
    df = pd.DataFrame({"timestamp": labels, "equity": values})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["equity"], mode="lines", name="Equity"))
    fig.update_layout(
        title="Equity Curve",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No equity curve data available in latest snapshot.")

st.subheader("Snapshots")

snapshots = report.get("snapshots", [])
if snapshots:
    df_snap = pd.DataFrame(snapshots)
    st.dataframe(df_snap, use_container_width=True)
else:
    st.write("No snapshot history found.")
