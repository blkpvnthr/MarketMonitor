from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any

from engine.data import fetch_ohlcv
from engine.strategies import build_signals
from engine.backtester import run_backtest


app = FastAPI(title="Backtesting Engine", version="1.0.0")

# Allow your HTML/JS UI to call this API from another port/domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BacktestRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    start: str = Field(..., description="YYYY-MM-DD", examples=["2022-01-01"])
    end: str = Field(..., description="YYYY-MM-DD", examples=["2024-12-31"])
    timeframe: Literal["1d", "1h", "30m", "15m", "5m"] = "1d"

    strategy: Literal["sma_cross", "rsi_reversion"] = "sma_cross"
    params: Dict[str, Any] = Field(default_factory=dict)

    initial_cash: float = 10_000.0
    commission_per_trade: float = 0.0
    slippage_bps: float = 0.0  # basis points, e.g., 5 = 0.05%
    allow_short: bool = False
    position_size: Literal["all_in", "fixed_qty"] = "all_in"
    fixed_qty: float = 1.0


class BacktestResponse(BaseModel):
    symbol: str
    timeframe: str
    start: str
    end: str
    ohlcv: Dict[str, list]         # timestamp, open, high, low, close, volume
    equity: Dict[str, list]        # timestamp, equity, returns
    trades: list                   # list of trades (buy/sell) w/ timestamps & prices
    metrics: Dict[str, Any]        # performance summary


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest):
    try:
        df = fetch_ohlcv(
            symbol=req.symbol,
            start=req.start,
            end=req.end,
            timeframe=req.timeframe,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data fetch failed: {e}")

    if df.empty or len(df) < 50:
        raise HTTPException(status_code=400, detail="Not enough data for backtest.")

    try:
        signals = build_signals(df, strategy=req.strategy, params=req.params)
        result = run_backtest(
            df=df,
            signals=signals,
            initial_cash=req.initial_cash,
            commission_per_trade=req.commission_per_trade,
            slippage_bps=req.slippage_bps,
            allow_short=req.allow_short,
            position_size=req.position_size,
            fixed_qty=req.fixed_qty,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Backtest failed: {e}")

    return {
        "symbol": req.symbol.upper(),
        "timeframe": req.timeframe,
        "start": req.start,
        "end": req.end,
        "ohlcv": result["ohlcv"],
        "equity": result["equity"],
        "trades": result["trades"],
        "metrics": result["metrics"],
    }
