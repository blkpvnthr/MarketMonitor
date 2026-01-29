from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Literal

from engine.metrics import summarize


def _periods_per_year_from_timeframe(timeframe: str) -> float:
    # rough defaults good enough for Sharpe scaling
    if timeframe == "1d":
        return 252.0
    if timeframe == "1h":
        return 252.0 * 6.5
    if timeframe == "30m":
        return 252.0 * 13.0
    if timeframe == "15m":
        return 252.0 * 26.0
    if timeframe == "5m":
        return 252.0 * 78.0
    return 252.0


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_cash: float,
    commission_per_trade: float,
    slippage_bps: float,
    allow_short: bool,
    position_size: Literal["all_in", "fixed_qty"] = "all_in",
    fixed_qty: float = 1.0,
) -> Dict[str, Any]:
    """
    Simple 1-asset backtester.
    - signals: target position per bar (+1 long, 0 flat, -1 short)
    - trades occur on bar close (you can change to next open if you want)
    """
    df = df.copy()
    signals = signals.reindex(df.index).fillna(0).astype(int)

    if not allow_short:
        signals = signals.clip(lower=0)

    close = df["close"].astype(float)

    slippage = (slippage_bps / 10_000.0)

    cash = float(initial_cash)
    position_qty = 0.0
    position_side = 0  # +1 long, -1 short, 0 flat
    entry_price = None
    entry_time = None
    entry_value = None

    equity = []
    returns = []
    ts_list = []

    trades = []

    prev_equity = initial_cash

    for i, ts in enumerate(df.index):
        px = float(close.iloc[i])
        target_side = int(signals.iloc[i])

        # Determine desired qty
        if position_size == "fixed_qty":
            target_qty = float(fixed_qty) if target_side != 0 else 0.0
        else:
            # all_in uses full equity into the position (no leverage)
            # qty = cash / price when entering; stays constant until exit
            target_qty = None  # computed at entry time only

        # If target changes, execute
        if target_side != position_side:
            # 1) Close existing position if any
            if position_side != 0 and position_qty != 0:
                # Apply slippage against you:
                # selling long -> slightly worse price, buying to cover short -> slightly worse
                exit_px = px * (1 - slippage) if position_side == +1 else px * (1 + slippage)

                # Realize PnL
                if position_side == +1:
                    pnl = (exit_px - float(entry_price)) * position_qty
                    cash += exit_px * position_qty
                else:
                    pnl = (float(entry_price) - exit_px) * position_qty
                    cash -= exit_px * position_qty  # buy to cover cost

                cash -= commission_per_trade

                trades.append({
                    "type": "SELL" if position_side == +1 else "BUY_TO_COVER",
                    "time": ts.isoformat(),
                    "price": float(exit_px),
                    "qty": float(position_qty),
                })
                trades.append({
                    "type": "CLOSE",
                    "time": ts.isoformat(),
                    "price": float(exit_px),
                    "qty": float(position_qty),
                    "pnl": float(pnl - commission_per_trade),
                    "entry_time": entry_time,
                    "entry_price": float(entry_price),
                })

                position_qty = 0.0
                position_side = 0
                entry_price = None
                entry_time = None
                entry_value = None

            # 2) Open new position if target_side != 0
            if target_side != 0:
                # Determine entry qty
                entry_px = px * (1 + slippage) if target_side == +1 else px * (1 - slippage)

                if position_size == "fixed_qty":
                    qty = float(fixed_qty)
                else:
                    # all_in: invest all available cash (no leverage)
                    if target_side == +1:
                        qty = cash / entry_px
                    else:
                        # shorting: for simplicity, require cash collateral equal to notional (no leverage)
                        qty = cash / entry_px
                qty = max(0.0, float(qty))

                if qty > 0:
                    if target_side == +1:
                        cash -= entry_px * qty
                    else:
                        # receive proceeds from short sale
                        cash += entry_px * qty

                    cash -= commission_per_trade

                    position_qty = qty
                    position_side = target_side
                    entry_price = float(entry_px)
                    entry_time = ts.isoformat()
                    entry_value = float(entry_px * qty)

                    trades.append({
                        "type": "BUY" if target_side == +1 else "SELL_SHORT",
                        "time": ts.isoformat(),
                        "price": float(entry_px),
                        "qty": float(qty),
                    })

        # Mark-to-market equity
        pos_value = 0.0
        if position_side == +1:
            pos_value = position_qty * px
        elif position_side == -1:
            # short: liability at current price
            pos_value = -position_qty * px

        eq = cash + pos_value
        r = (eq / prev_equity) - 1.0 if prev_equity != 0 else 0.0

        ts_list.append(ts.isoformat())
        equity.append(float(eq))
        returns.append(float(r))

        prev_equity = eq

    equity_s = pd.Series(equity, index=df.index, dtype=float)
    returns_s = pd.Series(returns, index=df.index, dtype=float)

    periods_per_year = _periods_per_year_from_timeframe("1d")  # conservative default
    metrics = summarize(equity_s, returns_s, trades, periods_per_year)

    ohlcv_payload = {
        "t": [t.isoformat() for t in df.index],
        "open": df["open"].astype(float).tolist(),
        "high": df["high"].astype(float).tolist(),
        "low": df["low"].astype(float).tolist(),
        "close": df["close"].astype(float).tolist(),
        "volume": df["volume"].astype(float).tolist(),
    }

    equity_payload = {
        "t": ts_list,
        "equity": equity,
        "returns": returns,
    }

    return {
        "ohlcv": ohlcv_payload,
        "equity": equity_payload,
        "trades": trades,
        "metrics": metrics,
    }