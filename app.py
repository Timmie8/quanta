import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Multi-Timeframe Trading Dashboard", layout="wide"
)

st.title("📈 Multi-Timeframe Trading & Signal Dashboard")
st.write(
    "Bekijk de trend en signalen over meerdere timeframes: **Swing (Daily)**, **1 Uur**, **15 Minuten** en **5 Minuten**."
)

# 1. Invoer van aandelen
tickers_input = st.text_input(
    "Aandelen (komma-gescheiden):", "AAPL, MSFT, NVDA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

selected_ticker = st.selectbox(
    "Selecteer een aandeel om te analyseren:", tickers
)


# Hulpfunctie voor het berekenen van indicatoren en scores per timeframe
def analyze_timeframe(df, tf_type="intraday"):
    if df.empty:
        return None, 0, "GEEN DATA", {}

    # MultiIndex opschonen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()

    # MAs en RSI
    if tf_type == "swing":
        df["EMA_FAST"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA_SLOW"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["ROC"] = df["Close"].pct_change(5) * 100
    else:  # Intraday / Kortere timeframes
        df["EMA_FAST"] = df["Close"].ewm(span=5, adjust=False).mean()
        df["EMA_SLOW"] = df["Close"].ewm(span=15, adjust=False).mean()
        df["ROC"] = df["Close"].pct_change(3) * 100

        # VWAP berekenen voor intraday
        v = df["Volume"]
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (tp * v).cumsum() / (v.cumsum() + 1e-9)

    # RSI 14
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Resistance & Support (20 periodes)
    df["Resistance"] = df["High"].rolling(20).max()
    df["Support"] = df["Low"].rolling(20).min()

    latest = df.iloc[-1]
    prev_res = df["Resistance"].shift(1).iloc[-1]
    prev_sup = df["Support"].shift(1).iloc[-1]

    # Score berekening (-4 tot +4)
    score = 0
    if latest["EMA_FAST"] > latest["EMA_SLOW"]:
        score += 1
    elif latest["EMA_FAST"] < latest["EMA_SLOW"]:
        score -= 1

    if tf_type == "intraday" and "VWAP" in latest:
        if latest["Close"] > latest["VWAP"]:
            score += 1
        elif latest["Close"] < latest["VWAP"]:
            score -= 1
    else:
        if latest["ROC"] > 0:
            score += 1
        elif latest["ROC"] < 0:
            score -= 1

    if latest["RSI_14"] > 50:
        score += 1
    elif latest["RSI_14"] < 50:
        score -= 1

    if latest["Close"] > prev_res:
        score += 1
    elif latest["Close"] < prev_sup:
        score -= 1

    # Verdict bepalen
    if score >= 3:
        verdict = "STRONG BUY"
    elif score >= 1:
        verdict = "BUY"
    elif score <= -3:
        verdict = "STRONG SELL"
    elif score <= -1:
        verdict = "SELL"
    else:
        verdict = "NEUTRAAL"

    metrics = {
        "score": score,
        "verdict": verdict,
        "price": latest["Close"],
        "rsi": latest["RSI_14"],
        "ema_fast": latest["EMA_FAST"],
        "ema_slow": latest["EMA_SLOW"],
        "vwap": latest.get("VWAP", None),
    }

    return df, score, verdict, metrics


if selected_ticker:
    with st.spinner(f"Alle timeframes ophalen voor {selected_ticker}..."):
        # Data ophalen voor diverse timeframes
        df_daily = yf.download(selected_ticker, period="1y", interval="1d")
        df_1h = yf.download(selected_ticker, period="1mo", interval="1h")
        df_15m = yf.download(selected_ticker, period="1mo", interval="15m")
        df_5m = yf.download(selected_ticker, period="5d", interval="5m")

    # Timeframe analyses uitvoeren
    df_daily, score_daily, verdict_daily, m_daily = analyze_timeframe(
        df_daily, tf_type="swing"
    )
    df_1h, score_1h, verdict_1h, m_1h = analyze_timeframe(
        df_1h, tf_type="intraday"
    )
    df_15m, score_15m, verdict_15m, m_15m = analyze_timeframe(
        df_15m, tf_type="intraday"
    )
    df_5m, score_5m, verdict_5m, m_5m = analyze_timeframe(
        df_5m, tf_type="intraday"
    )

    # =========================================================
    # BOVENKANT: MULTI-TIMEFRAME SCORE MATRIX & OVERZICHT
    # =========================================================
    st.subheader(f"📊 Multi-Timeframe Signaal Matrix - {selected_ticker}")

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)

    t_col1.metric(
        label="🌊 Swing (Daily)",
        value=verdict_daily,
        delta=f"Score: {score_daily} / 4",
    )
    t_col2.metric(
        label="⏱️ 1 Uur (Trend)",
        value=verdict_1h,
        delta=f"Score: {score_1h} / 4",
    )
    t_col3.metric(
        label="⚡ 15 Minuten (Setup)",
        value=verdict_15m,
        delta=f"Score: {score_15m} / 4",
    )
    t_col4.metric(
        label="🎯 5 Minuten (Entry)",
        value=verdict_5m,
        delta=f"Score: {score_5m} / 4",
    )

    # Totale Confluentie Check
    total_score = score_daily + score_1h + score_15m + score_5m
    st.write(f"**Totale Confluentie Score:** `{total_score} / 16`")

    if total_score >= 10:
        st.success(
            "🔥 **Sterke Bullish Confluentie:** Alle timeframes wijzen in dezelfde opwaartse richting!"
        )
    elif total_score <= -10:
        st.error(
            "⚠️ **Sterke Bearish Confluentie:** Alle timeframes wijzen in dezelfde neerwaartse richting!"
        )
    else:
        st.info(
            "ℹ️ **Gemengde Trend:** De timeframes spreken elkaar deels tegen. Wees voorzichtig met instappen."
        )

    st.divider()

    # =========================================================
    # TABS VOOR ELKE DETAILED TIMEFRAME GRAFIEK
    # =========================================================
    tab_s, tab_1h, tab_15m, tab_5m = st.tabs(
        [
            "🌊 Swing (Daily)",
            "⏱️ 1 Uur Trend",
            "⚡ 15 Minuten Setup",
            "🎯 5 Minuten Entry",
        ]
    )

    # 1. SWING TAB
    with tab_s:
        st.write(
            f"**Swing Score:** `{score_daily}/4` | **RSI:** `{m_daily['rsi']:.1f}`"
        )
        fig_s = go.Figure()
        fig_s.add_trace(
            go.Candlestick(
                x=df_daily.index,
                open=df_daily["Open"],
                high=df_daily["High"],
                low=df_daily["Low"],
                close=df_daily["Close"],
                name="Koers",
            )
        )
        fig_s.add_trace(
            go.Scatter(
                x=df_daily.index,
                y=df_daily["EMA_FAST"],
                line=dict(color="blue", width=1.5),
                name="EMA 20",
            )
        )
        fig_s.add_trace(
            go.Scatter(
                x=df_daily.index,
                y=df_daily["EMA_SLOW"],
                line=dict(color="orange", width=1.5),
                name="EMA 50",
            )
        )
        fig_s.update_layout(
            title="Daily Swing Chart",
            xaxis_rangeslider_visible=False,
            height=450,
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # 2. 1 UUR TAB
    with tab_1h:
        st.write(
            f"**1H Score:** `{score_1h}/4` | **RSI:** `{m_1h['rsi']:.1f}` | **VWAP:** `${m_1h['vwap']:.2f}`"
        )
        fig_1h = go.Figure()
        fig_1h.add_trace(
            go.Candlestick(
                x=df_1h.index,
                open=df_1h["Open"],
                high=df_1h["High"],
                low=df_1h["Low"],
                close=df_1h["Close"],
                name="Koers",
            )
        )
        fig_1h.add_trace(
            go.Scatter(
                x=df_1h.index,
                y=df_1h["EMA_FAST"],
                line=dict(color="cyan", width=1.5),
                name="EMA 5",
            )
        )
        fig_1h.add_trace(
            go.Scatter(
                x=df_1h.index,
                y=df_1h["EMA_SLOW"],
                line=dict(color="purple", width=1.5),
                name="EMA 15",
            )
        )
        fig_1h.add_trace(
            go.Scatter(
                x=df_1h.index,
                y=df_1h["VWAP"],
                line=dict(color="magenta", width=2),
                name="VWAP",
            )
        )
        fig_1h.update_layout(
            title="1 Uur Trend Chart",
            xaxis_rangeslider_visible=False,
            height=450,
        )
        st.plotly_chart(fig_1h, use_container_width=True)

    # 3. 15 MINUTEN TAB
    with tab_15m:
        st.write(
            f"**15M Score:** `{score_15m}/4` | **RSI:** `{m_15m['rsi']:.1f}` | **VWAP:** `${m_15m['vwap']:.2f}`"
        )
        fig_15m = go.Figure()
        fig_15m.add_trace(
            go.Candlestick(
                x=df_15m.index,
                open=df_15m["Open"],
                high=df_15m["High"],
                low=df_15m["Low"],
                close=df_15m["Close"],
                name="Koers",
            )
        )
        fig_15m.add_trace(
            go.Scatter(
                x=df_15m.index,
                y=df_15m["EMA_FAST"],
                line=dict(color="cyan", width=1.5),
                name="EMA 5",
            )
        )
        fig_15m.add_trace(
            go.Scatter(
                x=df_15m.index,
                y=df_15m["EMA_SLOW"],
                line=dict(color="purple", width=1.5),
                name="EMA 15",
            )
        )
        fig_15m.add_trace(
            go.Scatter(
                x=df_15m.index,
                y=df_15m["VWAP"],
                line=dict(color="magenta", width=2),
                name="VWAP",
            )
        )
        fig_15m.update_layout(
            title="15 Minuten Setup Chart",
            xaxis_rangeslider_visible=False,
            height=450,
        )
        st.plotly_chart(fig_15m, use_container_width=True)

    # 4. 5 MINUTEN TAB
    with tab_5m:
        st.write(
            f"**5M Score:** `{score_5m}/4` | **RSI:** `{m_5m['rsi']:.1f}` | **VWAP:** `${m_5m['vwap']:.2f}`"
        )
        fig_5m = go.Figure()
        fig_5m.add_trace(
            go.Candlestick(
                x=df_5m.index,
                open=df_5m["Open"],
                high=df_5m["High"],
                low=df_5m["Low"],
                close=df_5m["Close"],
                name="Koers",
            )
        )
        fig_5m.add_trace(
            go.Scatter(
                x=df_5m.index,
                y=df_5m["EMA_FAST"],
                line=dict(color="cyan", width=1.5),
                name="EMA 5",
            )
        )
        fig_5m.add_trace(
            go.Scatter(
                x=df_5m.index,
                y=df_5m["EMA_SLOW"],
                line=dict(color="purple", width=1.5),
                name="EMA 15",
            )
        )
        fig_5m.add_trace(
            go.Scatter(
                x=df_5m.index,
                y=df_5m["VWAP"],
                line=dict(color="magenta", width=2),
                name="VWAP",
            )
        )
        fig_5m.update_layout(
            title="5 Minuten Entry Chart",
            xaxis_rangeslider_visible=False,
            height=450,
        )
        st.plotly_chart(fig_5m, use_container_width=True)
