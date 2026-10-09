import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Trading Strategy & Signal Dashboard", layout="wide"
)

st.title("📈 Trading Strategy & Signal Dashboard")
st.write(
    "Analyseer aandelen voor zowel **Swing Trading** als **Daytrading**."
)

# 1. Invoer van aandelen
tickers_input = st.text_input(
    "Aandelen (komma-gescheiden):", "AAPL, MSFT, NVDA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

selected_ticker = st.selectbox(
    "Selecteer een aandeel om te analyseren:", tickers
)

if selected_ticker:
    with st.spinner(f"Data ophalen voor {selected_ticker}..."):
        # 1-minuut data voor Daytrading (laatste 7 dagen) & Dagelijkse data voor Swing
        df_daily = yf.download(selected_ticker, period="1y", interval="1d")
        df_intraday = yf.download(selected_ticker, period="5d", interval="5m")

    if not df_daily.empty:
        # MultiIndex kolommen opschonen
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.get_level_values(0)
        if isinstance(df_intraday.columns, pd.MultiIndex):
            df_intraday.columns = df_intraday.columns.get_level_values(0)

        # Tabs aanmaken voor Swing en Daytrade
        tab1, tab2 = st.tabs(["🌊 Swing Trading (3-5 dagen)", "⚡ Daytrading (Intraday)"])

        # ==========================================
        # TAB 1: SWING TRADING LOGICA
        # ==========================================
        with tab1:
            st.subheader(f"Swing Trade Analyse - {selected_ticker}")

            # Indicatoren Swing (20/50 EMA, 20D Resistance/Support)
            df_daily["EMA_20"] = df_daily["Close"].ewm(span=20, adjust=False).mean()
            df_daily["EMA_50"] = df_daily["Close"].ewm(span=50, adjust=False).mean()
            df_daily["ROC_5"] = df_daily["Close"].pct_change(5) * 100

            delta = df_daily["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_daily["RSI_14"] = 100 - (100 / (1 + rs))

            high_low = df_daily["High"] - df_daily["Low"]
            high_close = np.abs(df_daily["High"] - df_daily["Close"].shift())
            low_close = np.abs(df_daily["Low"] - df_daily["Close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df_daily["ATR_14"] = tr.rolling(14).mean()

            df_daily["Resistance_20"] = df_daily["High"].rolling(20).max()
            df_daily["Support_20"] = df_daily["Low"].rolling(20).min()

            latest_s = df_daily.iloc[-1]
            prev_res_s = df_daily["Resistance_20"].shift(1).iloc[-1]
            prev_sup_s = df_daily["Support_20"].shift(1).iloc[-1]

            # Swing Score (-4 tot +4)
            score_s = 0
            if latest_s["EMA_20"] > latest_s["EMA_50"]: score_s += 1
            elif latest_s["EMA_20"] < latest_s["EMA_50"]: score_s -= 1

            if latest_s["ROC_5"] > 0: score_s += 1
            elif latest_s["ROC_5"] < 0: score_s -= 1

            if latest_s["RSI_14"] > 50: score_s += 1
            elif latest_s["RSI_14"] < 50: score_s -= 1

            if latest_s["Close"] > prev_res_s: score_s += 1
            elif latest_s["Close"] < prev_sup_s: score_s -= 1

            verdict_s = "HOLD"
            if score_s >= 2 and latest_s["RSI_14"] > 50: verdict_s = "BUY"
            elif score_s <= -2 and latest_s["RSI_14"] < 50: verdict_s = "SELL"

            # Metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Swing Signaal", verdict_s)
            c2.metric("Score", f"{score_s} / 4")
            c3.metric("RSI (14)", f"{latest_s['RSI_14']:.1f}")
            c4.metric("ATR (14)", f"${latest_s['ATR_14']:.2f}")
            c5.metric("Koers", f"${latest_s['Close']:.2f}")

            if verdict_s == "BUY":
                entry = latest_s["Close"]
                atr = latest_s["ATR_14"]
                st.info(f"🎯 **Swing Trading Levels (BUY):** Entry: **${entry:.2f}** | Stop Loss (1.5x ATR): **${entry - 1.5*atr:.2f}** | Take Profit (3x ATR): **${entry + 3.0*atr:.2f}**")

            # Grafiek Swing
            fig_s = go.Figure()
            fig_s.add_trace(go.Candlestick(x=df_daily.index, open=df_daily["Open"], high=df_daily["High"], low=df_daily["Low"], close=df_daily["Close"], name="Koers"))
            fig_s.add_trace(go.Scatter(x=df_daily.index, y=df_daily["EMA_20"], line=dict(color="blue", width=1.5), name="EMA 20"))
            fig_s.add_trace(go.Scatter(x=df_daily.index, y=df_daily["EMA_50"], line=dict(color="orange", width=1.5), name="EMA 50"))
            fig_s.add_trace(go.Scatter(x=df_daily.index, y=df_daily["Resistance_20"], line=dict(color="green", width=1, dash="dash"), name="Resistance (20D)"))
            fig_s.add_trace(go.Scatter(x=df_daily.index, y=df_daily["Support_20"], line=dict(color="red", width=1, dash="dash"), name="Support (20D)"))
            fig_s.update_layout(title="Swing Chart (Daily)", xaxis_title="Datum", yaxis_title="Prijs ($)", xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig_s, use_container_width=True)

        # ==========================================
        # TAB 2: DAYTRADING LOGICA (INTRADAY)
        # ==========================================
        with tab2:
            st.subheader(f"Daytrading Analyse (5-minuten Intraday) - {selected_ticker}")

            if not df_intraday.empty:
                # Indicatoren Daytrading (EMA 5/15, Intraday VWAP, 14-period RSI op 5m)
                df_intraday["EMA_5"] = df_intraday["Close"].ewm(span=5, adjust=False).mean()
                df_intraday["EMA_15"] = df_intraday["Close"].ewm(span=15, adjust=False).mean()

                # VWAP berekening
                v = df_intraday["Volume"]
                tp = (df_intraday["High"] + df_intraday["Low"] + df_intraday["Close"]) / 3
                df_intraday["VWAP"] = (tp * v).cumsum() / v.cumsum()

                # Momentum (ROC 3 op 5m)
                df_intraday["ROC_3"] = df_intraday["Close"].pct_change(3) * 100

                # RSI 14 op intraday
                delta_d = df_intraday["Close"].diff()
                gain_d = (delta_d.where(delta_d > 0, 0)).rolling(window=14).mean()
                loss_d = (-delta_d.where(delta_d < 0, 0)).rolling(window=14).mean()
                rs_d = gain_d / loss_d
                df_intraday["RSI_14"] = 100 - (100 / (1 + rs_d))

                latest_d = df_intraday.iloc[-1]

                # Daytrade Score (-4 tot +4)
                score_d = 0
                # 1. Trend: EMA 5 vs EMA 15
                if latest_d["EMA_5"] > latest_d["EMA_15"]: score_d += 1
                elif latest_d["EMA_5"] < latest_d["EMA_15"]: score_d -= 1

                # 2. VWAP Positie (cruciaal voor daytraders)
                if latest_d["Close"] > latest_d["VWAP"]: score_d += 1
                elif latest_d["Close"] < latest_d["VWAP"]: score_d -= 1

                # 3. Korte Termijn Momentum
                if latest_d["ROC_3"] > 0: score_d += 1
                elif latest_d["ROC_3"] < 0: score_d -= 1

                # 4. Intraday RSI Filter
                if latest_d["RSI_14"] > 50: score_d += 1
                elif latest_d["RSI_14"] < 50: score_d -= 1

                verdict_d = "HOLD / NEUTRAAL"
                if score_d >= 3 and latest_d["Close"] > latest_d["VWAP"]: verdict_d = "STRONG BUY (SCALP)"
                elif score_d == 2: verdict_d = "BUY (INTRADAY)"
                elif score_d <= -3 and latest_d["Close"] < latest_d["VWAP"]: verdict_d = "STRONG SHORT"
                elif score_d <= -2: verdict_d = "SHORT (INTRADAY)"

                # Metrics Daytrading
                dc1, dc2, dc3, dc4, dc5 = st.columns(5)
                dc1.metric("Daytrade Signaal", verdict_d)
                dc2.metric("Daytrade Score", f"{score_d} / 4")
                dc3.metric("RSI (5m)", f"{latest_d['RSI_14']:.1f}")
                dc4.metric("VWAP", f"${latest_d['VWAP']:.2f}")
                dc5.metric("Laatste Koers", f"${latest_d['Close']:.2f}")

                st.divider()

                # Grafiek Daytrading (5-minuten kaarten van de laatste trading sessie)
                df_today = df_intraday.tail(78)  # Laatste 6,5 uur aan 5-minuten bars
                fig_d = go.Figure()
                fig_d.add_trace(go.Candlestick(x=df_today.index, open=df_today["Open"], high=df_today["High"], low=df_today["Low"], close=df_today["Close"], name="5m Koers"))
                fig_d.add_trace(go.Scatter(x=df_today.index, y=df_today["EMA_5"], line=dict(color="cyan", width=1.5), name="EMA 5"))
                fig_d.add_trace(go.Scatter(x=df_today.index, y=df_today["EMA_15"], line=dict(color="purple", width=1.5), name="EMA 15"))
                fig_d.add_trace(go.Scatter(x=df_today.index, y=df_today["VWAP"], line=dict(color="magenta", width=2), name="VWAP"))
                fig_d.update_layout(title="Daytrade Chart (5-Minuten Intraday)", xaxis_title="Tijd", yaxis_title="Prijs ($)", xaxis_rangeslider_visible=False, height=500)
                st.plotly_chart(fig_d, use_container_width=True)
            else:
                st.error("Geen intraday data beschikbaar (markt gesloten of geen data).")

    else:
        st.error(f"Geen gegevens gevonden voor {selected_ticker}.")
