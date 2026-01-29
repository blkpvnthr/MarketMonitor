from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    mu = r.mean() * periods_per_year
    sig = r.std(ddof=1) * np.sqrt(periods_per_year)
    if sig == 0 or np.isnan(sig):
        return 0.0
    return float(mu / sig)


def summarize(
    equity: pd.Series,
    returns: pd.Series,
    trades: list,
    periods_per_year: float,
) -> Dict[str, Any]:
    eq0 = float(equity.iloc[0])
    eq1 = float(equity.iloc[-1])
    total_return = (eq1 / eq0) - 1.0 if eq0 != 0 else 0.0

    mdd = max_drawdown(equity)
    sharpe = sharpe_ratio(returns, periods_per_year)

    # trade stats
    wins = 0
    losses = 0
    pnl_list = []
    for t in trades:
        if t.get("type") == "CLOSE":
            pnl = float(t.get("pnl", 0.0))
            pnl_list.append(pnl)
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

    n_closed = wins + losses
    win_rate = (wins / n_closed) if n_closed else 0.0
    avg_pnl = float(np.mean(pnl_list)) if pnl_list else 0.0

    return {
        "starting_equity": eq0,
        "ending_equity": eq1,
        "total_return": float(total_return),
        "max_drawdown": float(mdd),
        "sharpe": float(sharpe),
        "closed_trades": int(n_closed),
        "win_rate": float(win_rate),
        "avg_trade_pnl": float(avg_pnl),
    }