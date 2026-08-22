from agents import TracingProcessor, Trace, Span
from database import write_log
import secrets
import string

ALPHANUM = string.ascii_lowercase + string.digits 

def make_trace_id(tag: str) -> str:
    """
    Return a string of the form 'trace_<tag><random>',
    where the total length after 'trace_' is 32 chars.
    """
    tag += "0"
    pad_len = 32 - len(tag)
    random_suffix = ''.join(secrets.choice(ALPHANUM) for _ in range(pad_len))
    return f"trace_{tag}{random_suffix}"

class LogTracer(TracingProcessor):

    def get_name(self, trace_or_span: Trace | Span) -> str | None:
        trace_id = trace_or_span.trace_id
        name = trace_id.split("_")[1]
        if '0' in name:
            return name.split("0")[0]
        else:
            return None

    def on_trace_start(self, trace) -> None:
        name = self.get_name(trace)
        if name:
            write_log(name, "trace", f"Started: {trace.name}")

    def on_trace_end(self, trace) -> None:
        name = self.get_name(trace)
        if name:
            write_log(name, "trace", f"Ended: {trace.name}")

    def on_span_start(self, span) -> None:
        name = self.get_name(span)
        type = span.span_data.type if span.span_data else "span"
        if not name:
            return
        # Tool calls (span type "function" -- every MCP tool call, e.g. get_stock_screener,
        # get_market_regime, sell_credit_spread, plus the Researcher sub-agent itself, which
        # is exposed to the trader as a callable tool) are logged once, on completion, by
        # on_span_end below with a dedicated one-line message -- the "Started" half adds no
        # signal for verifying which tools actually got called, and doubling the row count
        # for what's already the highest-volume span type would crowd everything else out of
        # the dashboard's fixed-size log window.
        if type == "function":
            return
        message = "Started"
        if span.span_data:
            if span.span_data.type:
                message += f" {span.span_data.type}"
            if hasattr(span.span_data, "name") and span.span_data.name:
                message += f" {span.span_data.name}"
            if hasattr(span.span_data, "server") and span.span_data.server:
                message += f" {span.span_data.server}"
        if span.error:
            message += f" {span.error}"
        write_log(name, type, message)

    def on_span_end(self, span) -> None:
        name = self.get_name(span)
        type = span.span_data.type if span.span_data else "span"
        if not name:
            return
        if type == "function" and span.span_data:
            # One clear row per tool call: "Called <tool_name>", flagged if the tool itself
            # errored (raised) -- note this is separate from a tool returning an ordinary
            # {"error": ...} JSON payload, e.g. a rejected sell_credit_spread, which isn't
            # visible here since that's a normal return value, not a span error.
            tool_name = getattr(span.span_data, "name", None) or "tool"
            message = f"Called {tool_name}"
            if span.error:
                message += f" -- ERROR: {span.error}"
            write_log(name, type, message)
            return
        message = "Ended"
        if span.span_data:
            if span.span_data.type:
                message += f" {span.span_data.type}"
            if hasattr(span.span_data, "name") and span.span_data.name:
                message += f" {span.span_data.name}"
            if hasattr(span.span_data, "server") and span.span_data.server:
                message += f" {span.span_data.server}"
        if span.error:
            message += f" {span.error}"
        write_log(name, type, message)

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass