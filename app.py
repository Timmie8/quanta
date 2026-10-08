# region imports
from AlgorithmImports import *
from datetime import time
from datetime import date
# endregion


class SwingTrade:
    """State of one open trade (long or short)."""

    def __init__(self, symbol: Symbol, direction: int, entry_price: float,
                 stop_price: float, target_price: float, entry_date: date):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price
        self.entry_date = entry_date
        self.days_held = 0
        self.exiting = False


class MomentumTrendSrSystem(QCAlgorithm):
    """
    Day-trading / max-3-day swing system driven by momentum, trend, and
    support-resistance breakouts.

    Symbol input, two modes:
    - Set the "tickers" project parameter (comma-separated, e.g. "AAPL,MSFT,NVDA")
      to score and trade exactly those stocks. Each gets a daily pre-open score:
      BUY / HOLD / SELL.
    - Leave "tickers" empty to use the automatic universe: SPY constituents
      capped to the 20 most liquid names (daily selection).

    Signals (daily resolution unless noted):
    - Trend    : EMA(20) vs EMA(50).
    - Momentum : 5-day rate of change; intraday price vs 30-min SMA(10).
    - RSI      : daily RSI(14) must be above 50 for every BUY.
    - S/R      : prior 20-day daily high (resistance) and low (support); enter
                 on intraday close breaking through the level with trend,
                 momentum and RSI aligned.

    Exits: ATR stop (1.5x) and take profit (3x) checked on every 30-min bar;
    hard exit at the close on the 3rd trading day (market-on-close); immediate
    liquidation when the name leaves the universe. Trade state is only cleared
    when the exit fill is confirmed, so a trade can never be entered twice and
    no position is left unmanaged.
    """

    def initialize(self) -> None:
        self.set_start_date(2023, 1, 1)
        self.set_cash(100_000)
        self.settings.seed_initial_prices = True
        self.settings.minimum_order_margin_portfolio_percentage = 0
        self.settings.free_portfolio_value_percentage = 0.05

        self._max_positions = 8
        self._universe_size = 20
        self._sr_lookback = 20
        self._atr_period = 14
        self._intraday_sma_period = 10
        self._stop_atr_mult = 1.5
        self._target_atr_mult = 3.0
        self._max_days_held = 3
        self._position_weight = 1.0 / self._max_positions
        self._entry_cutoff = time(15, 0)

        self._trades_by_symbol: dict[Symbol, SwingTrade] = {}
        self._intraday_consolidators: dict[Symbol, IDataConsolidator] = {}
        self._daily_indicators: dict[Symbol, object] = {}

        tickers_param = self.get_parameter("tickers")
        self._manual_tickers: list[str] = (
            [t.strip().upper() for t in tickers_param.split(",") if t.strip()]
            if tickers_param else ["AAPL", "MSFT", "NVDA"]
        )
        self._manual_symbols: list[Symbol] = []

        if self._manual_tickers:
            self._universe = None
            for ticker in self._manual_tickers:
                security = self.add_equity(ticker, Resolution.MINUTE)
                self._manual_symbols.append(security.symbol)
                self._setup_symbol(security)
        else:
            self._universe = self.add_universe(
                self.universe.etf("SPY"),
                self._select_symbols,
            )

        # Daily pre-open score for every manually entered stock.
        if self._manual_tickers:
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.at(8, 0),
                self._score_tickers,
            )

        # Close-of-day maintenance (increment holding-days, force max-hold exits
        # at the official close via market-on-close submitted >= 15.5 min early).
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 16),
            self._on_day_close,
        )

    def _select_symbols(self, fundamentals: list[Fundamental]) -> list[Symbol]:
        top = sorted(
            fundamentals,
            key=lambda f: f.dollar_volume,
            reverse=True,
        )[: self._universe_size]
        return [f.symbol for f in top]

    def on_securities_changed(self, changes: SecurityChanges) -> None:
        for removed in changes.removed_securities:
            symbol = removed.symbol
            # Liquidate positions whose signal basis has left the universe
            # (keep the trade state until the exit fill confirms).
            if self.portfolio[symbol].invested:
                self.liquidate(symbol, tag="universe exit")
            consolidator = self._intraday_consolidators.pop(symbol, None)
            if consolidator is not None:
                self.subscription_manager.remove_consolidator(symbol, consolidator)
            for ind in self._daily_indicators.pop(symbol, []):
                self.deregister_indicator(ind)
            if not self.portfolio[symbol].invested:
                self._trades_by_symbol.pop(symbol, None)

        for added in changes.added_securities:
            self._setup_symbol(added)

    # ------------------------------------------------------------- symbol setup

    def _setup_symbol(self, security: Security) -> None:
        symbol = security.symbol
        # Daily trend / momentum / RSI / ATR indicators.
        security.ema_fast = ExponentialMovingAverage(20)
        security.ema_slow = ExponentialMovingAverage(50)
        security.mom = RateOfChangePercent(5)
        security.rsi = RelativeStrengthIndex(14)
        security.atr = AverageTrueRange(self._atr_period)
        # Support / resistance: prior 20-day high and low of daily bars.
        security.resistance = Maximum(self._sr_lookback)
        security.support = Minimum(self._sr_lookback)
        # Intraday momentum smoothing on 30-min bars.
        security.intraday_sma = SimpleMovingAverage(self._intraday_sma_period)

        daily = [security.ema_fast, security.ema_slow, security.mom,
                 security.rsi, security.atr, security.resistance,
                 security.support]
        for ind in (security.ema_fast, security.ema_slow, security.mom,
                    security.rsi, security.atr):
            self.warm_up_indicator(symbol, ind, Resolution.DAILY)
        self.warm_up_indicator(symbol, security.resistance,
                               Resolution.DAILY, lambda b: b.high)
        self.warm_up_indicator(symbol, security.support,
                               Resolution.DAILY, lambda b: b.low)

        for ind in (security.ema_fast, security.ema_slow, security.mom,
                    security.rsi, security.atr):
            self.register_indicator(symbol, ind, Resolution.DAILY)
        self.register_indicator(symbol, security.resistance,
                                Resolution.DAILY, lambda b: b.high)
        self.register_indicator(symbol, security.support,
                                Resolution.DAILY, lambda b: b.low)
        self._daily_indicators[symbol] = daily

        # One 30-min bar handler drives both entries and intraday exits.
        self._intraday_consolidators[symbol] = self.consolidate(
            symbol, timedelta(minutes=30),
            lambda b, s=symbol: self._on_intraday_bar(s, b))

    def on_order_event(self, order_event: OrderEvent) -> None:
        symbol = order_event.symbol
        if symbol not in self._trades_by_symbol:
            return
        if order_event.status in (OrderStatus.INVALID, OrderStatus.CANCELED,
                                  OrderStatus.FILLED):
            if self.portfolio[symbol].quantity == 0:
                self._trades_by_symbol.pop(symbol, None)

    # ------------------------------------------------------------------ scoring

    def _score_tickers(self) -> None:
        for symbol in self._manual_symbols:
            security = self.securities[symbol]
            if not self._indicators_ready(security):
                continue
            score, verdict = self._score_security(security)
            parts = (f"trend {security.ema_fast.current.value:.2f} vs "
                     f"{security.ema_slow.current.value:.2f}, "
                     f"ROC5 {security.mom.current.value:.2%}, "
                     f"RSI14 {security.rsi.current.value:.1f}, "
                     f"close {security.price:.2f}, "
                     f"resistance {security.resistance.current.value:.2f}, "
                     f"support {security.support.current.value:.2f}")
            self.log(f"SCORE {symbol.value}: {verdict} ({score:+d}/4 | {parts})")

    def _indicators_ready(self, security: Security) -> bool:
        return all(
            ind.is_ready for ind in (security.ema_fast, security.ema_slow,
                                     security.mom, security.rsi, security.atr,
                                     security.resistance, security.support)
        )

    def _score_security(self, security: Security) -> tuple[int, str]:
        trend = (security.ema_fast.current.value
                 - security.ema_slow.current.value)
        roc = security.mom.current.value
        rsi = security.rsi.current.value
        price = security.price
        resistance = security.resistance.current.value
        support = security.support.current.value

        score = 0
        if trend > 0:
            score += 1
        elif trend < 0:
            score -= 1
        if roc > 0:
            score += 1
        elif roc < 0:
            score -= 1
        if rsi > 50:
            score += 1
        elif rsi < 50:
            score -= 1
        if price > resistance:
            score += 1
        elif price < support:
            score -= 1

        # A BUY additionally requires RSI(14) above 50; a SELL below 50.
        if score >= 2 and rsi > 50:
            return score, "BUY"
        if score <= -2 and rsi < 50:
            return score, "SELL"
        return score, "HOLD"

    # ------------------------------------------------------------------ entries

    def _on_intraday_bar(self, symbol: Symbol, bar: TradeBar) -> None:
        if self.is_warming_up:
            return
        security = self.securities[symbol]
        security.intraday_sma.update(bar.end_time, bar.close)

        trade = self._trades_by_symbol.get(symbol)
        if trade is not None:
            self._manage_open_trade(trade, bar)
            return

        if self.portfolio[symbol].invested:
            return
        if bar.end_time.time() >= self._entry_cutoff:
            return
        if len(self._trades_by_symbol) >= self._max_positions:
            return
        if (self._universe is not None
                and symbol not in self._universe.selected):
            return
        if not self._indicators_ready(security) \
                or not security.intraday_sma.is_ready:
            return

        price = bar.close
        sma = security.intraday_sma.current.value
        atr = security.atr.current.value
        if atr <= 0:
            return

        direction = 0
        if (security.ema_fast.current.value > security.ema_slow.current.value
                and security.mom.current.value > 0
                and security.rsi.current.value > 50
                and price > sma
                and price > security.resistance.current.value):
            direction = 1
        elif (security.ema_fast.current.value < security.ema_slow.current.value
                and security.mom.current.value < 0
                and security.rsi.current.value < 50
                and price < sma
                and price < security.support.current.value):
            direction = -1

        if direction == 0:
            return

        self.set_holdings(symbol, direction * self._position_weight,
                          tag="breakout entry")
        self._trades_by_symbol[symbol] = SwingTrade(
            symbol, direction, price,
            price - direction * self._stop_atr_mult * atr,
            price + direction * self._target_atr_mult * atr,
            bar.end_time.date(),
        )

    # -------------------------------------------------------------------- exits

    def _manage_open_trade(self, trade: SwingTrade, bar: TradeBar) -> None:
        if trade.exiting:
            return
        price = bar.close
        if trade.direction > 0:
            hit_stop = price <= trade.stop_price
            hit_target = price >= trade.target_price
        else:
            hit_stop = price >= trade.stop_price
            hit_target = price <= trade.target_price
        if hit_stop:
            trade.exiting = True
            self.liquidate(trade.symbol, tag="stop loss")
        elif hit_target:
            trade.exiting = True
            self.liquidate(trade.symbol, tag="take profit")

    def _on_day_close(self) -> None:
        self.plot("Risk", "Open Trades", len(self._trades_by_symbol))
        today = self.time.date()
        for trade in list(self._trades_by_symbol.values()):
            if trade.exiting:
                continue
            if trade.entry_date != today:
                trade.days_held += 1
            if trade.days_held >= self._max_days_held:
                holding = self.portfolio[trade.symbol]
                if holding.quantity != 0:
                    trade.exiting = True
                    self.market_on_close_order(
                        trade.symbol, -holding.quantity, tag="max 3-day hold")
