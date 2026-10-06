"""Technical indicators calculated from historical closing prices."""

from __future__ import annotations

import pandas as pd


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of OHLCV data with SMA, RSI, and MACD columns."""
    result = data.copy()
    close = result["Close"]

    # Rolling indicators use only the current and earlier observations.
    result["SMA 20"] = close.rolling(window=20, min_periods=20).mean()
    result["SMA 50"] = close.rolling(window=50, min_periods=50).mean()

    daily_change = close.diff()
    average_gain = daily_change.clip(lower=0).ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()
    average_loss = (-daily_change.clip(upper=0)).ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()
    relative_strength = average_gain / average_loss.replace(0, float("nan"))
    result["RSI"] = 100 - (100 / (1 + relative_strength))
    # When there have been gains but no losses, RSI is 100 rather than NaN.
    result.loc[(average_loss == 0) & (average_gain > 0), "RSI"] = 100.0

    ema_12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    result["MACD"] = ema_12 - ema_26
    result["MACD Signal"] = result["MACD"].ewm(
        span=9, min_periods=9, adjust=False
    ).mean()
    result["MACD Histogram"] = result["MACD"] - result["MACD Signal"]

    return result
