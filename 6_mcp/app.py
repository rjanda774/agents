import gradio as gr  # type: ignore
from util import css, js, Color
import pandas as pd  # type: ignore
from trading_floor import names, lastnames, short_model_names
import plotly.express as px  # type: ignore
from accounts import Account
from database import read_log

mapper = {
    "trace": Color.WHITE,
    "agent": Color.CYAN,
    "function": Color.GREEN,
    "generation": Color.YELLOW,
    "response": Color.MAGENTA,
    "account": Color.RED,
    "live": Color.CYAN,  # real Schwab trades: staged, sent, filled, cancelled (live_trading.py)
    "scan": Color.WHITE,  # spread_scanner.py's per-cycle summary
}


class Trader:
    def __init__(self, name: str, lastname: str, model_name: str):
        self.name = name
        self.lastname = lastname
        self.model_name = model_name
        self.account = Account.get(name)
        self._last_log_html = None
        self._real_cache = (0.0, None)

    def reload(self):
        self.account = Account.get(self.name)

    def get_title(self) -> str:
        return f"<div style='text-align: center;font-size:34px;'>{self.name}<span style='color:#ccc;font-size:24px;'> ({self.model_name}) - {self.lastname}</span></div>"

    def get_strategy(self) -> str:
        return self.account.get_strategy()

    def get_portfolio_value_df(self) -> pd.DataFrame:
        if self.name == "Cathie":
            # The real Schwab account's Day Net Liq (live_trading.record_account_value,
            # one point per 15 min while the dashboard or the trading floor is running).
            import live_trading
            points = live_trading.account_value_history()
        else:
            points = self.account.portfolio_value_time_series
        df = pd.DataFrame(points, columns=["datetime", "value"])
        # ISO strings with a UTC offset (real history) -> naive ET clock times for the axis.
        df["datetime"] = pd.to_datetime(df["datetime"].str[:19])
        return df

    def get_portfolio_value_chart(self):
        df = self.get_portfolio_value_df()
        if df.empty:
            # Return empty chart if no data
            fig = px.line()
        else:
            fig = px.line(df, x="datetime", y="value")
        margin = dict(l=40, r=20, t=20, b=40)
        fig.update_layout(
            height=200,
            margin=margin,
            xaxis_title=None,
            yaxis_title=None,
            paper_bgcolor="#bbb",
            plot_bgcolor="#dde",
        )
        fig.update_xaxes(tickformat="%m/%d", tickangle=45, tickfont=dict(size=8))
        if self.name == "Cathie":
            # Market hours only: no flat lines across nights and weekends.
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16.25, 9.5], pattern="hour")])
        # Force Y-axis to auto-scale based on data range
        fig.update_yaxes(
            tickfont=dict(size=8), 
            tickformat=",.0f",
            autorange=True  # Force auto-scaling
        )
        return fig

    def get_holdings_df(self) -> pd.DataFrame:
        """Convert holdings to DataFrame for display"""
        holdings = self.account.get_holdings()
        if not holdings:
            return pd.DataFrame(columns=["Symbol", "Quantity"])

        df = pd.DataFrame(
            [{"Symbol": symbol, "Quantity": quantity} for symbol, quantity in holdings.items()]
        )
        return df

    def get_transactions_df(self) -> pd.DataFrame:
        """Convert transactions to DataFrame for display"""
        transactions = self.account.list_transactions()
        if not transactions:
            return pd.DataFrame(columns=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"])

        return pd.DataFrame(transactions)

    def get_portfolio_value(self) -> str:
        """Headline value. For Cathie: the real Schwab account's Day Net Liquidating Value
        and its change since the last value recorded before today."""
        if self.name == "Cathie":
            return self.get_real_headline()
        portfolio_value = self.account.calculate_portfolio_value() or 0.0
        pnl = self.account.calculate_profit_loss(portfolio_value) or 0.0
        color = "green" if pnl >= 0 else "red"
        emoji = "⬆" if pnl >= 0 else "⬇"
        return f"<div style='text-align: center;background-color:{color};'><span style='font-size:32px'>${portfolio_value:,.0f}</span><span style='font-size:24px'>&nbsp;&nbsp;&nbsp;{emoji}&nbsp;${pnl:,.0f}</span></div>"

    def get_real_headline(self) -> str:
        if not self.has_real_account():
            return ("<div style='text-align:center;background-color:#7f8c8d;font-size:20px;padding:4px;'>"
                    "Schwab not connected: Cathie isn't trading</div>")
        try:
            _, sm = self._real_data()
            net_liq = sm.get("net_liq")
        except Exception:
            net_liq = None
        if net_liq is None:
            return ("<div style='text-align:center;background-color:#7f8c8d;font-size:20px;padding:4px;'>"
                    "Real account value unavailable (Schwab can't be read right now)</div>")
        import live_trading
        today = live_trading._now_et().date().isoformat()
        before = [v for t, v in live_trading.account_value_history() if t[:10] < today]
        change = net_liq - before[-1] if before else None
        color = "green" if (change or 0) >= 0 else "red"
        tail = "" if change is None else (
            f"<span style='font-size:22px'>&nbsp;&nbsp;&nbsp;{'⬆' if change >= 0 else '⬇'}&nbsp;"
            f"${abs(change):,.0f} today</span>")
        return (f"<div style='text-align:center;background-color:{color};'>"
                f"<span style='font-size:14px'>Schwab Day Net Liq&nbsp;</span>"
                f"<span style='font-size:32px'>${net_liq:,.0f}</span>{tail}</div>")

    def get_cash_breakdown(self) -> str:
        """Display cash available vs cash in holdings (stock traders only)."""
        cash_available = self.account.balance

        # Standard stock traders
        holdings_value = 0.0
        for symbol, quantity in self.account.holdings.items():
            from market import get_share_price
            price = get_share_price(symbol)
            holdings_value += price * quantity

        return f"""<div style='text-align: center; font-size: 14px; padding: 5px;'>
            <div style='color: #2ecc71; font-weight: bold;'>💵 Cash Available: ${cash_available:,.2f}</div>
            <div style='color: #3498db; font-weight: bold;'>&#128202; Cash in Holdings: ${holdings_value:,.2f}</div>
        </div>"""

    def get_logs(self):
        # Bumped from 50 -- a Cathie cycle alone can make 20-30+ tool calls (each
        # now a single "function"-type row, see tracers.py), which would otherwise crowd
        # a whole cycle's worth of other entries out of a 50-row window.
        logs = list(read_log(self.name, last_n=150))
        # Only show meaningful entries - skip noisy span/trace/agent/mcp_tools chatter.
        # "function" IS shown -- it's every tool call (get_stock_screener, get_market_regime,
        # sell_credit_spread, the Researcher sub-agent, etc.), the main way to verify from
        # this panel which tools a trader actually used in a given cycle.
        # "live" is every real-trade event (live_trading.py, CATHIE_EXECUTION_MODE=approve).
        # "scan" is the spread scanner's one-line summary each cycle (spread_scanner.py).
        SHOW_TYPES = {"account", "generation", "response", "function", "live", "scan"}
        response = ""
        for log in logs:
            timestamp, type, message = log
            if type not in SHOW_TYPES:
                continue
            color = mapper.get(type, Color.WHITE).value
            response += f"<span style='color:{color}'>{timestamp} : [{type}] {message}</span><br/>"
        response = f"<div style='height:220px; overflow-y:auto;'>{response}</div>"
        # Only replace the DOM content if it actually changed -- the browser resets scroll
        # position on every innerHTML replacement, so re-rendering unchanged content every
        # 2s makes the panel feel like it's "overwriting itself" if you're mid-scroll. Track
        # the previous value on the instance instead of round-tripping it through Gradio as
        # a component input -- that was unreliable (some Timer/session-timing edge case sent
        # 0 inputs instead of 1, crashing the app on every load).
        if response != self._last_log_html:
            self._last_log_html = response
            return response
        return gr.update()

    # ------------------------------------------------ real Schwab account (approve mode)

    def has_real_account(self) -> bool:
        if self.name != "Cathie":
            return False
        try:
            import schwab_client
            return schwab_client.is_configured()
        except Exception:
            return False

    def _real_data(self):
        """One Schwab read shared by the summary and the table (cached 60s)."""
        import time
        import real_dashboard
        stamp, data = self._real_cache
        if data is None or time.time() - stamp > 60:
            data = real_dashboard.fetch()
            self._real_cache = (time.time(), data)
        return data

    def get_real_summary(self) -> str:
        try:
            from datetime import datetime
            _, sm = self._real_data()
        except Exception as e:
            return f"<div style='text-align:center;color:#111;background:#fafafa;font-weight:600;'>Real account error: {e}</div>"

        def money(x):
            return "?" if x is None else f"${x:,.2f}"

        def pnl(x):
            color = "#1e8449" if x >= 0 else "#c0392b"
            return f"<span style='color:{color};font-weight:800;'>{'+' if x >= 0 else '-'}${abs(x):,.2f}</span>"
        err = f"<br/><span style='color:#c0392b'>{sm['error']}</span>" if sm.get("error") else ""
        auto = (f", auto-closed by Cathie {pnl(sm['manual_auto_closed_pnl'])}"
                if sm.get("manual_auto_closed_pnl") else "")
        return (
            "<div style='text-align:center;font-size:14px;font-weight:600;color:#111;background:#fafafa;"
            "border-radius:4px;padding:4px;line-height:1.6;'>"
            "<b>REAL SCHWAB ACCOUNT</b> "
            f"<span style='color:#444'>(as of {datetime.now().strftime('%H:%M')})</span><br/>"
            f"<span style='color:#1e8449'>Funds Available {money(sm['funds_available'])}</span> &nbsp;"
            f"<span style='color:#1f618d'>Day Net Liq {money(sm['net_liq'])}</span><br/>"
            f"CATHIE: {sm['cathie_open']} open, unrealized {pnl(sm['cathie_open_pnl'])}, "
            f"realized {pnl(sm['cathie_realized'])} &nbsp;|&nbsp; orders today {sm['orders_today']}/{sm['daily_limit']}<br/>"
            f"MANUAL: {sm['manual_open']} open, unrealized {pnl(sm['manual_open_pnl'])}"
            f"{auto}{err}</div>"
        )

    def get_real_positions_html(self) -> str:
        """Every option position at Schwab, CATHIE (hers) or MANUAL (the user's), plus
        her recent real trades that aren't open (real_dashboard.py)."""
        import real_dashboard
        try:
            rows, _ = self._real_data()
        except Exception as e:
            return f"<div style='text-align:center;color:#111;background:#fafafa;font-weight:600;'>Real positions error: {e}</div>"
        return real_dashboard.to_html(rows)


class TraderView:
    def __init__(self, trader: Trader):
        self.trader = trader
        self.portfolio_value = None
        self.cash_breakdown = None
        self.chart = None
        self.holdings_table = None
        self.transactions_table = None
        self.real_summary = None
        self.real_table = None

    def make_ui(self):
        # Cathie trades real money only (paper trading removed 2026-10-14): her column is
        # the real Schwab account -- Day Net Liq and its chart, the summary line, every
        # option position at Schwab -- then her log.
        is_cathie = self.trader.name == "Cathie"
        with gr.Column(scale=1, min_width=300):
            gr.HTML(self.trader.get_title())
            with gr.Row():
                self.portfolio_value = gr.HTML(self.trader.get_portfolio_value)
            with gr.Row():
                self.chart = gr.Plot(
                    self.trader.get_portfolio_value_chart, container=True, show_label=False
                )
            if is_cathie:
                if self.trader.has_real_account():
                    with gr.Row():
                        self.real_summary = gr.HTML(self.trader.get_real_summary)
                    with gr.Row():
                        self.real_table = gr.HTML(self.trader.get_real_positions_html)
                with gr.Row(variant="panel"):
                    self.log = gr.HTML(self.trader.get_logs())
            else:
                with gr.Row():
                    self.cash_breakdown = gr.HTML(self.trader.get_cash_breakdown)
                with gr.Row(variant="panel"):
                    self.log = gr.HTML(self.trader.get_logs())
                with gr.Row():
                    self.holdings_table = gr.Dataframe(
                        value=self.trader.get_holdings_df,
                        label="Holdings",
                        headers=["Symbol", "Quantity"],
                        column_count=2,
                        max_height=150,
                        elem_classes=["dataframe-fix-small"],
                    )
                with gr.Row():
                    self.transactions_table = gr.Dataframe(
                        value=self.trader.get_transactions_df,
                        label="Recent Transactions",
                        headers=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"],
                        column_count=5,
                        max_height=150,
                        elem_classes=["dataframe-fix"],
                        elem_id=f"transactions-{self.trader.name.lower()}",
                    )

        timer = gr.Timer(value=120)
        timer.tick(
            fn=self.refresh,
            inputs=[],
            outputs=self._refresh_outputs(),
            show_progress="hidden",
            queue=False,
        )
        log_timer = gr.Timer(value=2)
        log_timer.tick(
            fn=self.trader.get_logs,
            inputs=[],
            outputs=[self.log],
            show_progress="hidden",
            queue=False,
        )

    def _refresh_outputs(self):
        outputs = [self.portfolio_value, self.chart]
        if self.real_table:
            outputs += [self.real_summary, self.real_table]
        if self.holdings_table:
            outputs += [self.cash_breakdown, self.holdings_table, self.transactions_table]
        return outputs

    def refresh(self):
        self.trader.reload()
        results = [self.trader.get_portfolio_value(), self.trader.get_portfolio_value_chart()]
        if self.real_table:
            results += [self.trader.get_real_summary(), self.trader.get_real_positions_html()]
        if self.holdings_table:
            results += [self.trader.get_cash_breakdown(), self.trader.get_holdings_df(),
                        self.trader.get_transactions_df()]
        return tuple(results)


# Main UI construction
def create_ui():
    """Create the main Gradio UI for the trading simulation"""

    traders = [
        Trader(trader_name, lastname, model_name)
        for trader_name, lastname, model_name in zip(names, lastnames, short_model_names)
    ]
    trader_views = [TraderView(trader) for trader in traders]

    with gr.Blocks(title="Traders", fill_width=True) as ui:
        with gr.Row():
            for trader_view in trader_views:
                trader_view.make_ui()

    return ui


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(
        inbrowser=True,
        css=css,
        js=js,
        theme=gr.themes.Ocean(),
    )
