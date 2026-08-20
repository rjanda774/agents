from mcp_compat import FastMCP
from regime_signal import get_regime_signal

mcp = FastMCP("regime_server")


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
    return get_regime_signal(symbol)


if __name__ == "__main__":
    mcp.run(transport='stdio')
