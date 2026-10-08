import json
import os
import subprocess
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="QuantConnect Backtest Dashboard", layout="wide")
st.title("📈 Swing Strategy Backtest Dashboard")

# Knop om de LEAN backtest te starten
if st.button("Start LEAN Backtest"):
    with st.spinner("Backtest wordt uitgevoerd via LEAN Docker..."):
        # Voer de LEAN CLI uit vanuit Python
        result = subprocess.run(
            ["lean", "backtest", "MijnStrategie"], capture_output=True, text=True
        )

    if result.returncode == 0:
        st.success("Backtest voltooid!")
    else:
        st.error(f"Fout bij uitvoeren backtest:\n{result.stderr}")

# Resultaten inladen uit het LEAN output-bestand
results_file = "MijnStrategie/backtests/latest/result.json"

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        data = json.load(f)

    statistics = data.get("Statistics", {})
    charts = data.get("Charts", {})

    # Key Performance Indicators (KPI's)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Compounded Annual Return", statistics.get("Compounded Annual Return", "N/B"))
    col2.metric("Sharpe Ratio", statistics.get("Sharpe Ratio", "N/B"))
    col3.metric("Drawdown", statistics.get("Max Drawdown", "N/B"))
    col4.metric("Total Trades", statistics.get("Total Trades", "N/B"))

    # Portfolio Waarde Grafiek
    if "Strategy Equity" in charts:
        equity_data = charts["Strategy Equity"]["Series"]["Equity"]["Values"]
        df_equity = pd.DataFrame(equity_data, columns=["Timestamp", "Equity"])
        df_equity["Date"] = pd.to_datetime(df_equity["Timestamp"], unit="s")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_equity["Date"], y=df_equity["Equity"], mode="lines", name="Equity"))
        fig.update_layout(title="Portfolio Equity Curve", xaxis_title="Datum", yaxis_title="Waarde ($)")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Nog geen backtest-resultaten gevonden. Klik op de knop om een backtest uit te voeren.")
