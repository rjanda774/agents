import asyncio

from mcp.server.fastmcp import FastMCP
from regime_signal import get_regime_signal

mcp = FastMCP("regime_server")

# get_regime_signal is a plain synchronous function -- it calls Polygon through
# market.py's rate limiter, which can legitimately block for tens of seconds waiting
# for a free slot (see market._polygon_rate_limited_call). Calling it directly from
# this async tool handler would block the server's single-threaded event loop for
# that whole wait, leaving the server unable to respond to anything else (including
# the eventual response to this very call) until it woke up -- which is exactly what
# turned a slow-but-recoverable rate-limit wait into a hard 120s MCP client timeout,
# and then cascaded into every other in-flight call timing out too, once queued waits
# started exceeding what the client would tolerate. Running it in a worker thread via
# asyncio.to_thread keeps the event loop free while it waits; wait_for bounds the
# total time so a congested queue fails cleanly instead of hanging indefinitely.
GET_REGIME_TIMEOUT_SECONDS = 100


@mcp.tool()
async def get_market_regime(symbol: str) -> dict:
    """Returns the current Bull/Sideways/Bear trend regime for a symbol, its historical
    persistence ('stickiness'), and forecasted regime probabilities for tomorrow and 5 days out.

    This is a LAGGING trend-context signal derived from the trailing 20-day return -- it
    confirms a trend already in motion, it does not predict a turn, and it is not a
    standalone buy/sell instruction or a profitability guarantee. Weigh it alongside your
    other research, and use the confidence field to judge how much history backs the numbers.

    Args:
        symbol: the ticker symbol to analyze
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(get_regime_signal, symbol),
            timeout=GET_REGIME_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return {
            "symbol": symbol,
            "error": (
                f"Regime signal timed out after {GET_REGIME_TIMEOUT_SECONDS}s, likely "
                "Polygon rate-limit congestion from checking many candidates this cycle. "
                "Skip the regime check for this candidate and continue with your other research."
            ),
        }
    except Exception as e:
        return {"symbol": symbol, "error": f"Regime signal failed: {e}"}


if __name__ == "__main__":
    mcp.run(transport='stdio')
