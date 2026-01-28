from __future__ import annotations

import pandas as pd
import yfinance as yf


_INTERVAL_MAP = {
    "1d": "1d",
    "1h": "60m",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
}


def fetch_ohlcv(symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
    """
    Returns a dataframe indexed by timestamp (UTC naive), columns:
    open, high, low, close, volume
    """
    symbol = symbol.strip().upper()
    interval = _INTERVAL_MAP.get(timeframe)
    if interval is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize columns
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    keep = ["open", "high", "low", "close", "volume"]
    df = df[keep].dropna()

    # Ensure monotonic increasing index
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)

    return df
