"""
Markov regime-signal tool.

Classifies an asset's trend into Bull / Sideways / Bear based on its trailing
20-day return, fits a transition matrix from history, and forecasts near-term
regime probabilities.

Design notes (see conversation / video critique this addresses):
- Today's regime is labeled using a ROLLING 20-day window, because we need a
  live, up-to-date label every day.
- The transition matrix used to estimate stickiness/persistence is fit from
  NON-OVERLAPPING (strided) samples, to avoid pseudo-replication: adjacent
  rolling windows share 19/20 days of data, so counting daily transitions
  naively overstates how much independent evidence supports the stickiness
  number.
- Multi-day forecasts use numpy.linalg.matrix_power on the full matrix, NOT
  scalar-squaring a single probability -- squaring a scalar silently assumes
  zero autocorrelation, which contradicts "stickiness" existing at all.
- This is a LAGGING signal: the regime label reflects a move that has already
  happened over the last 20 days. It's context for a trend filter, not a
  leading/entry signal, and it says nothing about profitability on its own.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
from dotenv import load_dotenv
from polygon import RESTClient

load_dotenv(override=True)

polygon_api_key = os.getenv("POLYGON_API_KEY")

STATES = ["Bear", "Sideways", "Bull"]
LOOKBACK_DAYS = 20
BULL_THRESHOLD = 0.05
BEAR_THRESHOLD = -0.05
STRIDE_DAYS = LOOKBACK_DAYS  # non-overlapping samples for matrix fitting
MIN_HISTORY_DAYS = LOOKBACK_DAYS * 25  # want a real amount of history before trusting this


class InsufficientHistoryError(Exception):
    pass


def get_historical_closes(symbol: str, lookback_years: int = 8) -> list[float]:
    """Daily closing prices for `symbol`, oldest first."""
    if not polygon_api_key:
        raise RuntimeError("POLYGON_API_KEY not set; cannot compute a regime signal without historical data.")
    client = RESTClient(polygon_api_key)
    end = datetime.now().date()
    start = end - timedelta(days=365 * lookback_years)
    aggs = client.get_aggs(symbol, 1, "day", start.isoformat(), end.isoformat(), adjusted=True, limit=50000)
    return [a.close for a in aggs]


def classify_state(window_return: float) -> str:
    if window_return >= BULL_THRESHOLD:
        return "Bull"
    if window_return <= BEAR_THRESHOLD:
        return "Bear"
    return "Sideways"


def compute_daily_states(closes: list[float]) -> list[str]:
    """Rolling LOOKBACK_DAYS cumulative return -> state label, one per day."""
    states = []
    for i in range(LOOKBACK_DAYS, len(closes)):
        window_start, window_end = closes[i - LOOKBACK_DAYS], closes[i]
        cum_return = (window_end - window_start) / window_start
        states.append(classify_state(cum_return))
    return states


def build_transition_matrix(states: list[str], stride: int = STRIDE_DAYS) -> tuple[np.ndarray, int, np.ndarray]:
    """Fit the transition matrix from non-overlapping samples (avoids pseudo-replication).

    Also returns per-row sample counts (transitions observed FROM each state) so callers
    can detect a state that was never sampled as a source -- its row is all zeros, which
    is a stochastic-matrix violation (persistence would read as a misleading "0.0" instead
    of "no data", and matrix_power on that row produces a forecast that doesn't sum to 1).
    """
    sampled = states[::stride]
    idx = {s: i for i, s in enumerate(STATES)}
    counts = np.zeros((3, 3))
    for a, b in zip(sampled, sampled[1:]):
        counts[idx[a]][idx[b]] += 1
    sample_count = int(counts.sum())
    row_sums = counts.sum(axis=1)
    matrix = np.divide(counts, row_sums[:, None], out=np.zeros_like(counts), where=row_sums[:, None] != 0)
    return matrix, sample_count, row_sums


def forecast(matrix: np.ndarray, current_state: str, days_ahead: int) -> dict[str, float]:
    """N-day-ahead regime probabilities via matrix exponentiation (matrix_power), not scalar-squaring."""
    powered = np.linalg.matrix_power(matrix, days_ahead)
    row = powered[STATES.index(current_state)]
    return dict(zip(STATES, (float(p) for p in row)))


def confidence_label(sample_count: int) -> str:
    if sample_count < 15:
        return "low (thin history -- treat with real skepticism)"
    if sample_count < 40:
        return "moderate"
    return "reasonable"


def get_regime_signal(symbol: str, lookback_years: int = 8) -> dict:
    """
    Returns today's regime, its historical persistence/stickiness, and near-term
    forecasts for `symbol`. This is trend CONTEXT, not a buy/sell instruction.
    """
    # Guard: get_historical_closes hits Polygon's live API and can raise for reasons that
    # have nothing to do with the symbol -- a missing key, a network blip, or (confirmed
    # live, see CLAUDE.md) a free-tier key's ~5-calls/minute rate limit tripping mid-cycle
    # if Cathie checks regime for several candidates back to back. Every other failure mode
    # in this function returns a clean {"error": ...} dict instead of guessing; letting a
    # raw urllib3/polygon exception escape here instead would hand the LLM (or whoever
    # calls get_market_regime) a stack trace instead of an actionable message.
    try:
        closes = get_historical_closes(symbol, lookback_years)
    except Exception as e:
        return {
            "symbol": symbol,
            "error": (
                f"Could not fetch price history for {symbol}: {e}. Often transient (a "
                "free-tier Polygon key's rate limit, or a network blip) -- wait a moment "
                "and try again, or try a different symbol."
            ),
        }
    if len(closes) < MIN_HISTORY_DAYS:
        return {
            "symbol": symbol,
            "error": (
                f"Only {len(closes)} days of price history available; need at least "
                f"{MIN_HISTORY_DAYS} to fit a transition matrix with any real confidence. "
                "Refusing to guess."
            ),
        }

    states = compute_daily_states(closes)
    current_state = states[-1]

    matrix, sample_count, row_sums = build_transition_matrix(states)
    current_idx = STATES.index(current_state)

    # Guard: today's regime never occurred as a SOURCE state among the strided samples
    # used to fit the matrix, so its row is all zeros. Reporting persistence=0.0 from
    # that row would misleadingly read as "this regime never repeats" instead of the
    # true "we have zero samples of transitions from this regime", and matrix_power on
    # an all-zero row produces a forecast that doesn't sum to 1. Refuse instead of
    # guessing, consistent with the overall MIN_HISTORY_DAYS refusal above.
    if row_sums[current_idx] == 0:
        return {
            "symbol": symbol,
            "current_regime": current_state,
            "regime_basis": f"trailing {LOOKBACK_DAYS}-trading-day cumulative return (this is a LAGGING label)",
            "error": (
                f"Today's regime ({current_state}) never occurred as a starting state in the "
                f"non-overlapping samples used to fit the transition matrix ({sample_count} total "
                f"transitions sampled). Cannot estimate persistence or forecast future regimes "
                f"from a state with zero observed transitions -- refusing to guess rather than "
                f"report a fabricated 0% persistence or an invalid forecast."
            ),
            "transition_matrix_sample_size": sample_count,
        }

    persistence = matrix[current_idx][current_idx]

    return {
        "symbol": symbol,
        "current_regime": current_state,
        "regime_basis": f"trailing {LOOKBACK_DAYS}-trading-day cumulative return (this is a LAGGING label)",
        "persistence": round(float(persistence), 3),
        "forecast_tomorrow": {k: round(v, 3) for k, v in forecast(matrix, current_state, 1).items()},
        "forecast_5_days": {k: round(v, 3) for k, v in forecast(matrix, current_state, 5).items()},
        "transition_matrix_sample_size": sample_count,
        "confidence": confidence_label(sample_count),
        "note": (
            "This regime label reflects a move that has already happened over the last "
            f"{LOOKBACK_DAYS} days -- it confirms a trend, it does not predict a turn. "
            "Trend persistence ('stickiness') is not universal: it fails in mean-reverting "
            "assets/periods. Treat this as one input alongside your research, not a standalone "
            "trading instruction, and it says nothing on its own about profitability."
        ),
    }
