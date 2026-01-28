from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Literal


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_signals(
    df: pd.DataFrame,
    strategy: Literal["sma_cross", "rsi_reversion"],
    params: Dict[str, Any],
) -> pd.Series:
    close = df["close"]

    if strategy == "sma_cross":
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 50))
        if fast <= 1 or slow <= 1 or fast >= slow:
            raise ValueError("sma_cross requires 1 < fast < slow")

        sma_fast = _sma(close, fast)
        sma_slow = _sma(close, slow)

        # position: long when fast > slow else flat
        pos = (sma_fast > sma_slow).astype(int)

        return pos.reindex(df.index).fillna(0).astype(int)

    if strategy == "rsi_reversion":
        n = int(params.get("rsi_n", 14))
        low_th = float(params.get("low", 30))
        high_th = float(params.get("high", 70))

        rsi = _rsi(close, n=n)

        # Simple mean-reversion: long under low_th, exit above 50; (flat otherwise)
        pos = pd.Series(0, index=df.index, dtype=int)
        in_long = False
        for i in range(len(df)):
            ts = df.index[i]
            if not in_long and rsi.iloc[i] <= low_th:
                in_long = True
            elif in_long and rsi.iloc[i] >= 50:
                in_long = False
            pos.loc[ts] = 1 if in_long else 0

        return pos.astype(int)

    raise ValueError(f"Unknown strategy: {strategy}")
