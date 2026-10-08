import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Swing Strategy Backtest Dashboard", layout="wide"
)

st.title("📈 Swing Strategy Backtest & Signal Dashboard")
st.write(
    "Voer hieronder je gewenste aandelen in om de strategie direct te analyseren."
)

# 1. Invoer van aandelen
tickers_input = st.text_input(
    "Aandelen (komma-gescheiden):", "AAPL, MSFT, NVDA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

selected_ticker = st.selectbox("Selecteer een aandeel om de grafiek te bekijken:", tickers)

if selected_ticker:
    with st.spinner(f"Data ophalen voor {selected_ticker}..."):
        # Koersdata van de afgelopen 2 jaar ophalen
        df = yf.download(selected_ticker, period="2y", interval="1d")

    if not df.empty:
        # MultiIndex kolommen opschonen indien aanwezig
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Indicatoren berekenen (conform QuantConnect logica)
        df["EMA_20"] = ta.ema(df["Close"], length=20)
        df["EMA_50"] = ta.ema(df["Close"], length=50)
        df["RSI_14"] = ta.rsi(df["Close"], length=14)
        df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        df["ROC_5"] = ta.roc(df["Close"], length=5)
        df["Resistance_20"] = df["High"].rolling(20).max()
        df["Support_20"] = df["Low"].rolling(20).min()

        # Laatste bar analyseren
        latest = df.iloc[-1]
        prev_resistance = df["Resistance_20"].shift(1).iloc[-1]
        prev_support = df["Support_20"].shift(1).iloc[-1]

        # Score bepalen (-4 tot +4)
        score = 0
        if latest["EMA_20"] > latest["EMA_50"]:
            score += 1
        elif latest["EMA_20"] < latest["EMA_50"]:
            score -= 1

        if latest["ROC_5"] > 0:
            score += 1
        elif latest["ROC_5"] < 0:
            score -= 1

        if latest["RSI_14"] > 50:
            score += 1
        elif latest["RSI_14"] < 50:
            score -= 1

        if latest["Close"] > prev_resistance:
            score += 1
        elif latest["Close"] < prev_support:
            score -= 1

        # Verdict bepalen
        verdict = "HOLD"
        if score >= 2 and latest["RSI_14"] > 50:
            verdict = "BUY"
        elif score <= -2 and latest["RSI_14"] < 50:
            verdict = "SELL"

        # 3. KPI / Metrics Dashboard
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Signaal", verdict)
        col2.metric("Score", f"{score} / 4")
        col3.metric("RSI (14)", f"{latest['RSI_14']:.1f}")
        col4.metric("ATR (14)", f"${latest['ATR_14']:.2f}")
        col5.metric("Sluitkoers", f"${latest['Close']:.2f}")

        st.divider()

        # Target & Stop-loss berekening bij een BUY signaal
        if verdict == "BUY":
            entry = latest["Close"]
            atr = latest["ATR_14"]
            stop_loss = entry - (1.5 * atr)
            take_profit = entry + (3.0 * atr)
            st.info(
                f"🎯 **Handelsniveaus (BUY):** Entry: **${entry:.2f}** | Stop Loss (1.5x ATR): **${stop_loss:.2f}** | Take Profit (3x ATR): **${take_profit:.2f}**"
            )

        # 4. Interactive Plotly Grafiek
        fig = go.Figure()

        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Koers",
            )
        )

        # Moving Averages & Resistance/Support
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_20"],
                line=dict(color="blue", width=1.5),
                name="EMA 20",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_50"],
                line=dict(color="orange", width=1.5),
                name="EMA 50",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Resistance_20"],
                line=dict(color="green", width=1, dash="dash"),
                name="Resistance (20D)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Support_20"],
                line=dict(color="red", width=1, dash="dash"),
                name="Support (20D)",
            )
        )

        fig.update_layout(
            title=f"Technische Analyse: {selected_ticker}",
            xaxis_title="Datum",
            yaxis_title="Prijs ($)",
            xaxis_rangeslider_visible=False,
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(
            f"Geen gegevens gevonden voor {selected_ticker}. Controleer of de ticker correct is."
        )
