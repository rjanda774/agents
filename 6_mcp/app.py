import gradio as gr  # type: ignore
from util import css, js, Color
import pandas as pd  # type: ignore
from trading_floor import names, lastnames, short_model_names
import plotly.express as px  # type: ignore
from accounts import Account
from database import read_log, read_account

mapper = {
    "trace": Color.WHITE,
    "agent": Color.CYAN,
    "function": Color.GREEN,
    "generation": Color.YELLOW,
    "response": Color.MAGENTA,
    "account": Color.RED,
}


class Trader:
    def __init__(self, name: str, lastname: str, model_name: str):
        self.name = name
        self.lastname = lastname
        self.model_name = model_name
        self.account = Account.get(name)
        self._last_log_html = None

    def reload(self):
        self.account = Account.get(self.name)

    def get_title(self) -> str:
        return f"<div style='text-align: center;font-size:34px;'>{self.name}<span style='color:#ccc;font-size:24px;'> ({self.model_name}) - {self.lastname}</span></div>"

    def get_strategy(self) -> str:
        return self.account.get_strategy()

    def get_portfolio_value_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.account.portfolio_value_time_series, columns=["datetime", "value"])
        df["datetime"] = pd.to_datetime(df["datetime"])
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
        """Calculate total portfolio value. For Cathie, tracks her options account's live
        cash directly, not the untouched stock account -- see get_cash_breakdown."""
        if self.name == "Cathie":
            # Her real capital lives entirely in the options account -- self.account (the
            # stock account) is a separate, untouched $10,000 pool she never actually
            # trades, so blending its starting balance in would double-count capital and
            # not match the "Cash: $X" figure shown just below.
            starting_cash = 10_000.0
            options_data = read_account(f"{self.name.lower()}_options")
            portfolio_value = options_data.get("cash", starting_cash) if options_data else starting_cash
            pnl = portfolio_value - starting_cash
        else:
            portfolio_value = self.account.calculate_portfolio_value() or 0.0
            pnl = self.account.calculate_profit_loss(portfolio_value) or 0.0

        color = "green" if pnl >= 0 else "red"
        emoji = "⬆" if pnl >= 0 else "⬇"
        return f"<div style='text-align: center;background-color:{color};'><span style='font-size:32px'>${portfolio_value:,.0f}</span><span style='font-size:24px'>&nbsp;&nbsp;&nbsp;{emoji}&nbsp;${pnl:,.0f}</span></div>"

    def get_cash_breakdown(self) -> str:
        """Display cash available vs cash in holdings. For Cathie, shows options premium."""
        cash_available = self.account.balance

        if self.name == "Cathie":
            options_data = read_account(f"{self.name.lower()}_options")
            if options_data:
                # Her real cash lives in the options account, not the untouched stock
                # account -- she never buys/sells stocks, so self.account.balance would
                # always just show the starting $10,000 regardless of trading activity.
                options_cash = options_data.get("cash", cash_available)
                collected = options_data.get("total_premium_collected", 0.0)
                open_risk = sum(p.get("max_loss", 0) for p in options_data.get("open_positions", []))
                return f"""<div style='text-align: center; font-size: 14px; padding: 5px;'>
                    <div style='color: #2ecc71; font-weight: bold;'>💵 Cash: ${options_cash:,.2f}</div>
                    <div style='color: #f39c12; font-weight: bold;'>&#128176; Premium Collected: ${collected:,.2f}</div>
                    <div style='color: #e74c3c; font-weight: bold;'>&#9888; Open Risk: ${open_risk:,.2f}</div>
                </div>"""

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
        logs = list(read_log(self.name, last_n=50))
        # Only show meaningful entries - skip noisy span/trace/agent/function/mcp_tools chatter
        SHOW_TYPES = {"account", "generation", "response"}
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

    def get_options_positions_df(self) -> pd.DataFrame:
        """Get all of Cathie's options positions (open, closed, expired) as a log."""
        COLS = ["Sym", "Type", "Strikes", "Qty", "Expiry", "Opened", "Status", "Closed", "Premium", "P&L"]
        empty = pd.DataFrame(columns=COLS)
        try:
            from datetime import date, datetime
            options_data = read_account(f"{self.name.lower()}_options")
            if not options_data:
                return empty

            def fmt_dt(ts):
                """Format ISO timestamp to MM/DD HH:MM"""
                if not ts:
                    return ""
                try:
                    dt = datetime.fromisoformat(ts)
                    return dt.strftime("%m/%d %H:%M")
                except Exception:
                    return str(ts)[:10]

            def fmt_exp(exp):
                try:
                    return date.fromisoformat(exp).strftime("%m/%d/%Y")
                except Exception:
                    return exp

            rows = []
            today = date.today()

            for pos in options_data.get("open_positions", []):
                exp = pos.get("expiration_date", "")
                expired = False
                try:
                    expired = date.fromisoformat(exp) < today
                except Exception:
                    pass
                status = "Expired" if expired else "Open"
                prem = pos.get("net_premium_collected", 0)
                pnl = f"+${prem:.2f}" if expired else ""
                rows.append({
                    "Sym":     pos["symbol"],
                    "Type":    "BP" if pos.get("spread_type") == "bull_put" else "BC",
                    "Strikes": f"{pos['short_leg']['strike']}/{pos['long_leg']['strike']}",
                    "Qty":     pos["short_leg"]["contracts"],
                    "Expiry":  fmt_exp(exp),
                    "Opened":  fmt_dt(pos.get("opened_at")),
                    "Status":  status,
                    "Closed":  "",
                    "Premium": f"${prem:.2f}",
                    "P&L":     pnl,
                })

            for pos in options_data.get("closed_positions", []):
                exp = pos.get("expiration_date", "")
                status = pos.get("status", "closed").capitalize()
                prem = pos.get("net_premium_collected", 0)
                paid = pos.get("closing_premium_paid")
                if status == "Expired":
                    pnl_val = prem
                elif paid is not None:
                    pnl_val = prem - paid
                else:
                    pnl_val = pos.get("realized_pnl", 0) or prem
                pnl_str = f"+${pnl_val:.2f}" if pnl_val >= 0 else f"-${abs(pnl_val):.2f}"
                rows.append({
                    "Sym":     pos["symbol"],
                    "Type":    "BP" if pos.get("spread_type") == "bull_put" else "BC",
                    "Strikes": f"{pos['short_leg']['strike']}/{pos['long_leg']['strike']}",
                    "Qty":     pos["short_leg"]["contracts"],
                    "Expiry":  fmt_exp(exp),
                    "Opened":  fmt_dt(pos.get("opened_at")),
                    "Status":  status,
                    "Closed":  fmt_dt(pos.get("closed_at")),
                    "Premium": f"${prem:.2f}",
                    "P&L":     pnl_str,
                })

            if not rows:
                return empty

            # Sort: open/expired-open first (no closed_at), then closed most-recent first
            open_rows   = [r for r in rows if r["Closed"] == ""]
            closed_rows = sorted([r for r in rows if r["Closed"] != ""], key=lambda r: r["Closed"], reverse=True)
            return pd.DataFrame(open_rows + closed_rows)

        except Exception:
            return empty

    def get_options_summary(self) -> str:
        """Get Cathie's options account summary with unrealized P&L."""
        try:
            import yfinance as yf
            import concurrent.futures
            from datetime import date
            options_data = read_account(f"{self.name.lower()}_options")
            if not options_data:
                return "<div style='text-align:center;color:#aaa;padding:5px;'>No options positions yet</div>"
            open_pos   = options_data.get("open_positions", [])
            closed_pos = len(options_data.get("closed_positions", []))
            collected  = options_data.get("total_premium_collected", 0)
            realized   = options_data.get("total_realized_pnl", 0)
            today      = date.today()

            def fetch_unrealized(pos):
                """Fetch current closing cost for one position, return unrealized P&L."""
                exp = pos.get("expiration_date", "")
                premium = pos.get("net_premium_collected", 0)
                try:
                    if date.fromisoformat(exp) < today:
                        return premium  # expired worthless, keep all
                except Exception:
                    pass
                try:
                    t = yf.Ticker(pos["symbol"])
                    chain = t.option_chain(exp)
                    df = chain.puts if pos["spread_type"] == "bull_put" else chain.calls
                    strikes = df["strike"].values
                    sh = strikes[abs(strikes - pos["short_leg"]["strike"]).argmin()]
                    lg = strikes[abs(strikes - pos["long_leg"]["strike"]).argmin()]
                    sh_row = df[df["strike"] == sh]
                    lg_row = df[df["strike"] == lg]
                    if not sh_row.empty and not lg_row.empty:
                        sh_mid = (sh_row.iloc[0]["bid"] + sh_row.iloc[0]["ask"]) / 2
                        lg_mid = (lg_row.iloc[0]["bid"] + lg_row.iloc[0]["ask"]) / 2
                        contracts = pos["short_leg"]["contracts"]
                        current_cost = (sh_mid - lg_mid) * 100 * contracts
                        return premium - current_cost
                except Exception:
                    pass
                return None  # could not fetch - exclude from total

            # Fetch all positions in parallel with a 20s total timeout
            unrealized = 0.0
            fetched_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                futures = {ex.submit(fetch_unrealized, pos): pos for pos in open_pos}
                for fut in concurrent.futures.as_completed(futures, timeout=20):
                    try:
                        val = fut.result()
                        if val is not None:
                            unrealized += val
                            fetched_count += 1
                    except Exception:
                        pass

            fetch_note = "" if fetched_count == len(open_pos) else f" ({fetched_count}/{len(open_pos)} fetched)"
            total_pnl = realized + unrealized
            pnl_color = "#2ecc71" if total_pnl >= 0 else "#e74c3c"
            unr_color = "#2ecc71" if unrealized >= 0 else "#e74c3c"
            return (
                f"<div style='text-align:center;font-size:11px;padding:3px;line-height:1.6;'>"
                f"<span style='color:#2ecc71'>&#128203; Open: {len(open_pos)}</span> &nbsp;"
                f"<span style='color:#aaa'>&#9989; Closed: {closed_pos}</span> &nbsp;"
                f"<span style='color:#f39c12'>&#128176; Collected: ${collected:.2f}</span><br/>"
                f"<span style='color:{unr_color}'>Unrealized P&amp;L: ${unrealized:.2f}{fetch_note}</span> &nbsp;"
                f"<span style='color:{pnl_color}'><b>Total P&amp;L: ${total_pnl:.2f}</b></span>"
                f"</div>"
            )
        except Exception as e:
            return f"<div style='text-align:center;color:#aaa;'>Options summary error: {e}</div>"


class TraderView:
    def __init__(self, trader: Trader):
        self.trader = trader
        self.portfolio_value = None
        self.cash_breakdown = None
        self.chart = None
        self.holdings_table = None
        self.transactions_table = None

    def make_ui(self):
        is_cathie = self.trader.name == "Cathie"
        with gr.Column(scale=1, min_width=300):
            gr.HTML(self.trader.get_title())
            with gr.Row():
                self.portfolio_value = gr.HTML(self.trader.get_portfolio_value)
            with gr.Row():
                self.chart = gr.Plot(
                    self.trader.get_portfolio_value_chart, container=True, show_label=False
                )
            with gr.Row():
                self.cash_breakdown = gr.HTML(self.trader.get_cash_breakdown)
            with gr.Row(variant="panel"):
                self.log = gr.HTML(self.trader.get_logs())
            if is_cathie:
                with gr.Row():
                    self.options_summary = gr.HTML(self.trader.get_options_summary)
                with gr.Row():
                    self.holdings_table = gr.Dataframe(
                        value=self.trader.get_options_positions_df,
                        label="Options Positions Log",
                        headers=["Sym", "Type", "Strikes", "Qty", "Expiry", "Opened", "Status", "Closed", "Premium", "P&L"],
                        column_count=10,
                        max_height=200,
                        elem_classes=["dataframe-fix"],
                        elem_id="options-cathie",
                    )
                self.transactions_table = None
            else:
                self.options_summary = None
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
        refresh_outputs = [
            self.portfolio_value,
            self.cash_breakdown,
            self.chart,
            self.holdings_table,
        ]
        if self.options_summary:
            refresh_outputs.append(self.options_summary)
        if self.transactions_table:
            refresh_outputs.append(self.transactions_table)
        timer.tick(
            fn=self.refresh,
            inputs=[],
            outputs=refresh_outputs,
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

    def refresh(self):
        self.trader.reload()
        results = [
            self.trader.get_portfolio_value(),
            self.trader.get_cash_breakdown(),
            self.trader.get_portfolio_value_chart(),
            self.trader.get_options_positions_df() if self.trader.name == "Cathie" else self.trader.get_holdings_df(),
        ]
        if self.options_summary:
            results.append(self.trader.get_options_summary())
        if self.transactions_table:
            results.append(self.trader.get_transactions_df())
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
