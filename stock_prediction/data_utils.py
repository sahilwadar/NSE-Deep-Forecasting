"""Download and validate historical Indian stock data with yfinance."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class StockDataError(Exception):
    """A readable error for missing, invalid, or unavailable market data."""


def download_stock_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Download daily OHLCV data. The requested end date is included."""
    if start_date >= end_date:
        raise StockDataError("The start date must be earlier than the end date.")

    # yfinance treats `end` as exclusive, so add one day to include the date the
    # user selected in the sidebar.
    inclusive_end = end_date + timedelta(days=1)

    try:
        history = yf.Ticker(ticker).history(
            start=start_date.isoformat(),
            end=inclusive_end.isoformat(),
            interval="1d",
            auto_adjust=True,
            timeout=20,
        )
    except Exception as exc:
        raise StockDataError(
            "Yahoo Finance could not be reached. Check the internet connection "
            "and try again."
        ) from exc

    if history is None or history.empty:
        raise StockDataError(
            f"No historical data was returned for {ticker}. Try another ticker "
            "or a wider date range."
        )

    # Flatten columns defensively in case yfinance returns a MultiIndex.
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = [str(column[0]) for column in history.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in history.columns]
    if missing_columns:
        raise StockDataError(
            "The downloaded data is missing required columns: "
            + ", ".join(missing_columns)
        )

    history = history[REQUIRED_COLUMNS].copy()
    history.index = pd.to_datetime(history.index)
    if history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    history.index.name = "Date"
    history = history.sort_index()
    history = history[~history.index.duplicated(keep="last")]
    history = history.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if history.empty:
        raise StockDataError("The downloaded data contains no complete OHLCV rows.")

    return history.astype(
        {
            "Open": "float64",
            "High": "float64",
            "Low": "float64",
            "Close": "float64",
            "Volume": "float64",
        }
    )
