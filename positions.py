import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


from dotenv import load_dotenv # pyright: ignore[reportMissingImports]

import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest
from alpaca.data.timeframe import TimeFrame



load_dotenv()

APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_PAPER = os.getenv("APCA_PAPER", "true").lower() == "true"

if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
    raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment or .env")


def _n(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _iso_from_unix_seconds(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).astimezone().isoformat()


@dataclass
class AccountSnapshot:
    equity: float
    last_equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    initial_margin: float
    maintenance_margin: float

    # Intraday approx from Alpaca's account fields:
    day_pl: float
    day_plpc: float  # fraction (0.01 = 1%)

    # Optional extra:
    long_market_value: float
    short_market_value: float
    timestamp_iso: str


class AlpacaAccountClient:
    """
    Thin wrapper around alpaca-py TradingClient that returns JSON-friendly dicts.
    """

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.trading = TradingClient(api_key=api_key, secret_key=api_secret, paper=paper)

    def get_account_snapshot(self) -> AccountSnapshot:
        acct = self.trading.get_account()

        equity = _n(getattr(acct, "equity", None))
        last_equity = _n(getattr(acct, "last_equity", None))
        cash = _n(getattr(acct, "cash", None))
        buying_power = _n(getattr(acct, "buying_power", None))
        portfolio_value = _n(getattr(acct, "portfolio_value", None))
        initial_margin = _n(getattr(acct, "initial_margin", None))
        maintenance_margin = _n(getattr(acct, "maintenance_margin", None))
        long_mv = _n(getattr(acct, "long_market_value", None))
        short_mv = _n(getattr(acct, "short_market_value", None))

        day_pl = equity - last_equity if (equity == equity and last_equity == last_equity) else float("nan")
        day_plpc = (day_pl / last_equity) if (last_equity and last_equity == last_equity) else float("nan")

        return AccountSnapshot(
            equity=equity,
            last_equity=last_equity,
            cash=cash,
            buying_power=buying_power,
            portfolio_value=portfolio_value,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            day_pl=day_pl,
            day_plpc=day_plpc,
            long_market_value=long_mv,
            short_market_value=short_mv,
            timestamp_iso=datetime.now(tz=timezone.utc).astimezone().isoformat(),
        )

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Returns normalized positions list:
          symbol, qty, side, avg_entry, market_price, unrealized_pl, unrealized_plpc
        """
        positions = self.trading.get_all_positions()
        out: List[Dict[str, Any]] = []

        for p in positions:
            qty = _n(getattr(p, "qty", None))
            side = "short" if qty < 0 else "long"

            out.append(
                {
                    "symbol": getattr(p, "symbol", ""),
                    "qty": qty,
                    "side": side,
                    "avg_entry": _n(getattr(p, "avg_entry_price", None)),
                    "market_price": _n(getattr(p, "current_price", None)),
                    "unrealized_pl": _n(getattr(p, "unrealized_pl", None)),
                    # alpaca gives fractional (e.g., 0.0123 = 1.23%)
                    "unrealized_plpc": _n(getattr(p, "unrealized_plpc", None)),
                    "market_value": _n(getattr(p, "market_value", None)),
                    "cost_basis": _n(getattr(p, "cost_basis", None)),
                }
            )

        out.sort(key=lambda r: abs(_n(r.get("unrealized_pl"))), reverse=True)
        return out

    def _normalize_portfolio_tf(self, tf: str) -> str:
        """
        Alpaca portfolio history expects timeframe as a STRING.
        Common accepted values include: "1Min", "5Min", "15Min", "30Min", "1Hour", "1Day".
        """
        tf = str(tf).strip()

        aliases = {
            "1m": "1Min",
            "1min": "1Min",
            "5m": "5Min",
            "5min": "5Min",
            "15m": "15Min",
            "15min": "15Min",
            "30m": "30Min",
            "30min": "30Min",
            "1h": "1Hour",
            "1hr": "1Hour",
            "1hour": "1Hour",
            "1d": "1Day",
            "day": "1Day",
            "1day": "1Day",

            # keep your old label working
            "1H": "1Hour",
            "1D": "1Day",
        }

        # case-insensitive alias lookup, otherwise return as-is
        return aliases.get(tf.lower(), aliases.get(tf, tf))


    def get_equity_curve(
        self,
        period: str = "1D",
        timeframe: str = "1Min",
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """
        Uses Alpaca Portfolio History:
        period: "1D","1W","1M","3M","6M","1A","ALL"
        timeframe: "1Min","5Min","15Min","30Min","1Hour","1Day" (depends on period)

        Returns:
        equity_curve:  [{t, equity}]
        returns_curve: [{t, ret}] where ret is cumulative return from first point (fraction)
        max_drawdown:  fraction (negative)
        """
        tf = self._normalize_portfolio_tf(timeframe)

        req = GetPortfolioHistoryRequest(
            period=period,
            timeframe=tf,               # ✅ MUST be string
            extended_hours=extended_hours,
        )

        hist = self.trading.get_portfolio_history(req)

        equity = list(getattr(hist, "equity", []) or [])
        ts = list(getattr(hist, "timestamp", []) or [])

        curve: List[Dict[str, Any]] = []
        returns: List[Dict[str, Any]] = []
        max_drawdown: Optional[float] = None

        if equity and ts and len(equity) == len(ts):
            first = float(equity[0]) if equity[0] is not None else None
            peak = -1e99
            worst = 0.0

            for t, eq in zip(ts, equity):
                eqf = float(eq)
                iso = _iso_from_unix_seconds(int(t))

                curve.append({"t": iso, "equity": eqf})

                if first and first != 0:
                    returns.append({"t": iso, "ret": (eqf / first) - 1.0})
                else:
                    returns.append({"t": iso, "ret": float("nan")})

                peak = max(peak, eqf)
                dd = (eqf / peak - 1.0) if peak else 0.0
                worst = min(worst, dd)

            max_drawdown = float(worst)

        return {
            "equity_curve": curve,
            "returns_curve": returns,
            "max_drawdown": max_drawdown,
            "meta": {
                "period": period,
                "timeframe": tf,  # ✅ return the normalized one
                "extended_hours": extended_hours,
                "points": len(curve),
            },
        }



    def get_performance_bundle(
        self,
        period: str = "1D",
        timeframe: str = "1Min",
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """
        Single call that returns:
          account snapshot + positions + equity/returns curves + drawdown.
        """
        snap = self.get_account_snapshot()
        positions = self.get_positions()
        curve = self.get_equity_curve(period=period, timeframe=timeframe, extended_hours=extended_hours)

        return {
            "account": {
                "equity": snap.equity,
                "last_equity": snap.last_equity,
                "cash": snap.cash,
                "buying_power": snap.buying_power,
                "portfolio_value": snap.portfolio_value,
                "initial_margin": snap.initial_margin,
                "maintenance_margin": snap.maintenance_margin,
                "long_market_value": snap.long_market_value,
                "short_market_value": snap.short_market_value,
                "day_pl": snap.day_pl,
                "day_plpc": snap.day_plpc,
                "timestamp": snap.timestamp_iso,
            },
            "positions": positions,
            **curve,
        }


if __name__ == "__main__":
    client = AlpacaAccountClient(
        api_key=APCA_API_KEY_ID,
        api_secret=APCA_API_SECRET_KEY,
        paper=APCA_PAPER,
    )

    bundle = client.get_performance_bundle(period="1D", timeframe="1Min")
    # Print a small summary (avoid dumping huge curves)
    acct = bundle["account"]
    print(f"Equity: {acct['equity']:.2f} | Cash: {acct['cash']:.2f} | BP: {acct['buying_power']:.2f}")
    print(f"Day P/L: {acct['day_pl']:.2f} ({acct['day_plpc']*100:.2f}%)")
    print(f"Positions: {len(bundle['positions'])} | Curve points: {bundle['meta']['points']}")
