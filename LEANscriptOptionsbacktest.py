"""
Cathie's put credit spread rules, expressed as a QuantConnect LEAN algorithm.

PURPOSE: answer one question cheaply -- do these entry/exit rules make money
over ~8 years, after spread costs? Nothing else. This is not a port of the
trading floor project; it deliberately drops the agents, the LLM layer, the
MCP server and the database. Just the mechanical rules.

Run on QuantConnect's free tier (unlimited backtesting), or locally via
`lean backtest` with the LEAN CLI.

PARAMETERS BELOW ARE PLACEHOLDERS. Replace with Cathie's actual values from
templates.py / options_trading_server.py before drawing any conclusion.
"""

from AlgorithmImports import *


class CathieCreditSpreads(QCAlgorithm):

    # ---- knobs (mirror these to Cathie's config) -------------------------
    TARGET_DELTA        = 0.30    # short leg delta
    DELTA_TOLERANCE     = 0.07    # accept 0.23 - 0.37
    SPREAD_WIDTH        = 5.0     # dollars between strikes
    MIN_DTE             = 25
    MAX_DTE             = 45
    PROFIT_TARGET       = 0.75    # close at 50% of credit captured
    DEFENSIVE_DTE       = 07     # close regardless at this DTE
    SHORT_BREACH_DELTA  = 0.50    # close if short leg delta blows past this
    MIN_OPEN_INTEREST   = 100
    MAX_SPREAD_PCT      = 0.10    # bid/ask width as fraction of mid
    CONTRACTS           = 1
    MAX_CONCURRENT      = 3       # open spreads at once

    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2026, 6, 30)
        self.set_cash(100_000)

        # Spread costs are the whole point of testing this properly.
        self.set_brokerage_model(
            BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN
        )

        equity = self.add_equity("SPY", Resolution.MINUTE)
        self.underlying = equity.symbol

        option = self.add_option("SPY", Resolution.MINUTE)
        self.option_symbol = option.symbol
        option.set_filter(self._contract_filter)

        # open spreads: {expiry_key: {"short": Symbol, "long": Symbol,
        #                             "credit": float, "opened": datetime}}
        self.open_spreads = {}

        # Only act once a day, near the open -- matches how Cathie actually
        # runs. Avoids the minute-bar loop generating hundreds of decisions.
        self.schedule.on(
            self.date_rules.every_day(self.underlying),
            self.time_rules.after_market_open(self.underlying, 30),
            self._rebalance,
        )

    def _contract_filter(self, universe):
        """Narrow the chain before it hits on_data -- huge speed difference."""
        return (universe
                .include_weeklys()
                .puts_only()
                .strikes(-25, +5)
                .expiration(self.MIN_DTE, self.MAX_DTE))

    # ---- main loop -------------------------------------------------------
    def _rebalance(self):
        chain = self.current_slice.option_chains.get(self.option_symbol)
        if chain is None or not chain:
            return

        self._manage_open(chain)

        if len(self.open_spreads) < self.MAX_CONCURRENT:
            self._try_open(chain)

    # ---- exits -----------------------------------------------------------
    def _manage_open(self, chain):
        by_symbol = {c.symbol: c for c in chain}

        for key in list(self.open_spreads.keys()):
            pos = self.open_spreads[key]
            short_c = by_symbol.get(pos["short"])
            long_c = by_symbol.get(pos["long"])

            # Contract fell out of the filtered chain (usually near expiry).
            if short_c is None or long_c is None:
                if not self.portfolio[pos["short"]].invested:
                    del self.open_spreads[key]
                continue

            dte = (short_c.expiry.date() - self.time.date()).days

            # Cost to close = buy back short, sell the long.
            debit = self._mid(short_c) - self._mid(long_c)
            captured = (pos["credit"] - debit) / pos["credit"] if pos["credit"] > 0 else 0

            reason = None
            if captured >= self.PROFIT_TARGET:
                reason = "profit target"
            elif dte <= self.DEFENSIVE_DTE:
                reason = "defensive DTE"
            elif abs(short_c.greeks.delta) >= self.SHORT_BREACH_DELTA:
                reason = "short strike breached"

            if reason:
                self._close(pos, reason)
                del self.open_spreads[key]

    # ---- entry -----------------------------------------------------------
    def _try_open(self, chain):
        expiries = sorted({c.expiry for c in chain})
        if not expiries:
            return

        for expiry in expiries:
            key = expiry.strftime("%Y%m%d")
            if key in self.open_spreads:
                continue

            puts = [c for c in chain if c.expiry == expiry]

            # 1. short leg: closest to target delta, within tolerance
            candidates = [
                c for c in puts
                if abs(abs(c.greeks.delta) - self.TARGET_DELTA) <= self.DELTA_TOLERANCE
                and self._liquid(c)
            ]
            if not candidates:
                continue

            short_c = min(
                candidates,
                key=lambda c: (abs(abs(c.greeks.delta) - self.TARGET_DELTA),
                               -c.open_interest),
            )

            # 2. long leg: SPREAD_WIDTH below, must ALSO be liquid.
            #    A liquid short with an illiquid wing is a trap.
            target_strike = short_c.strike - self.SPREAD_WIDTH
            wings = [c for c in puts
                     if abs(c.strike - target_strike) < 0.01 and self._liquid(c)]
            if not wings:
                continue
            long_c = wings[0]

            credit = self._mid(short_c) - self._mid(long_c)
            if credit <= 0.05:     # near-zero premium -- skip, don't crash
                continue

            self._open(short_c, long_c, credit, key)
            return   # one new spread per day

    # ---- order helpers ---------------------------------------------------
    def _open(self, short_c, long_c, credit, key):
        """Submit as ONE combo order. Never leg in."""
        legs = [
            Leg.create(short_c.symbol, -1),
            Leg.create(long_c.symbol, +1),
        ]
        self.combo_market_order(legs, self.CONTRACTS)

        self.open_spreads[key] = {
            "short": short_c.symbol,
            "long": long_c.symbol,
            "credit": credit,
            "opened": self.time,
        }
        self.log(f"OPEN {short_c.strike}/{long_c.strike} "
                 f"exp {key} credit {credit:.2f} "
                 f"delta {short_c.greeks.delta:.2f}")

    def _close(self, pos, reason):
        legs = [
            Leg.create(pos["short"], +1),
            Leg.create(pos["long"], -1),
        ]
        self.combo_market_order(legs, self.CONTRACTS)
        self.log(f"CLOSE {pos['short'].value} -- {reason}")

    # ---- filters ---------------------------------------------------------
    def _liquid(self, c):
        if c.open_interest < self.MIN_OPEN_INTEREST:
            return False
        if c.bid_price <= 0 or c.ask_price <= 0:
            return False
        mid = self._mid(c)
        if mid <= 0:
            return False
        return (c.ask_price - c.bid_price) / mid <= self.MAX_SPREAD_PCT

    @staticmethod
    def _mid(c):
        return (c.bid_price + c.ask_price) / 2.0

    # ---- assignment ------------------------------------------------------
    def on_assignment_order_event(self, event):
        """Early assignment on the short leg. In a backtest LEAN handles the
        mechanics; log it so you can count how often it happens -- that
        frequency is a real input to whether this is tradeable."""
        self.log(f"ASSIGNED {event.symbol} at {self.time}")
        for key, pos in list(self.open_spreads.items()):
            if pos["short"] == event.symbol:
                self.liquidate(pos["long"])
                del self.open_spreads[key]