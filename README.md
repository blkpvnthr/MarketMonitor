# MarketMonitor (Alpaca) — Closed-Bar Feature + Eligibility Engine

A real-time market monitor that streams **1-minute bars + live quotes** from Alpaca, aggregates them into **CLOSED** 5m/15m/30m/1h bars, computes technical features **only on closed bars**, and runs an **eligibility / session-state engine** (VWAP + candle direction + chop detection) to determine which symbols are “trade-worthy” as the session develops.

It also labels a **market regime** (TREND/RANGE × HIGH/LOW volatility) from **1h closed bars** for a selected benchmark symbol.

---

## What this script does

### 1) Streams Alpaca data
- **1m bars** (used to build higher timeframes)
- **quotes** (context-only: bid/ask snapshot, spread, mid)

### 2) Aggregates bars (IMPORTANT: closed bars only)
A `BarAggregator` converts incoming **1m bars** into:
- 5m bars
- 15m bars
- 30m bars
- 1h bars

✅ It **emits only CLOSED bars** (no partial candles).

### 3) Computes features (only from CLOSED bars)
For each timeframe and symbol, `RollingSeries` computes:

- `ret_1` (last close-to-close return)
- `ema_20`, `ema_50`
- `rsi_14`
- `atr_14`, `atrp_14` (ATR % of price)
- `adx_14`
- `vwap` (intraday rolling VWAP)
- `vwap_dist_pct` (distance from VWAP)
- `vol_z_20` (20-period volume z-score)

### 4) Eligibility / Session-state engine (VWAP checkpoint rules)
The `EligibilityEngine` maintains a `SymbolSession` per symbol and updates state on:
- **5m close**: early filters (extreme VWAP dislocation, flip logic, chop penalty)
- **15m close**: assigns VWAP-based directional bias (LONG_ONLY / SHORT_ONLY / BOTH)
- **30m close**: confirmation / reconsider logic (trend vs reversion allowances)
- **1h close**: sizing multiplier adjustments based on VWAP context

States:
- `ACTIVE`
- `SUSPENDED`
- `RECONSIDER`
- `CONFIRMED`

### 5) Market regime labeling (1h CLOSED only)
For the chosen `REGIME_SYMBOL`, on each **1h close** it writes a regime row using:
- `ADX >= 22` → `TREND`, else `RANGE`
- `ATRP >= 0.60%` → `HIGH_VOL`, else `LOW_VOL`

---

## Output / Data layout

All outputs are **CSV append-only** under `OUT_DIR` (default: `./data_store`).

Writes a session_state.csv every time the 5m/15m/30m gates updates a symbol's state.

`session_state.csv` Contains:
- state, score, bias, flip_count
- vwap distances per TF
- allow_trend / allow_mean_reversion
- sizing_mult + computed `risk_mult`
- last_reason
- quote snapshot fields (bid/ask/mid/spread)

---

## Environment variables

Create a .env in the project root (or export these in your shell):

Required

APCA_API_KEY_ID — Alpaca key

APCA_API_SECRET_KEY — Alpaca secret

Common

SYMBOLS — comma-separated list
Default is a large prefilled list in code.

OUT_DIR — output folder (default: ./data_store)

ALPACA_DATA_FEED — IEX (default) or SIP

ALPACA_WS_URL — defaults to wss://stream.data.alpaca.markets/v2/iex (kept for compatibility; stream uses feed=)

Eligibility / VWAP tuning

VWAP_EXTREME_DIST_PCT — default 1.25
If abs(vwap_dist_pct) ≥ this on 5m #1, symbol is suspended.

VWAP_NEAR_PCT — default 0.10
Defines “near VWAP” zone used for chop/mean-reversion logic.

Regime benchmark symbol selection

REGIME_SYMBOL = If set to a symbol in SYMBOLS, it will be used.

Otherwise the monitor will auto-select the symbol with the highest CONFIRMED score after 30m logic.

---

## Quote logging

LOG_QUOTES — true|false (default false)
When true, writes OUT_DIR/quotes/quotes_YYYYMMDD.csv

Container safety caps (present for runner integration)

These are defined but not used directly in the streaming loop yet:

RUN_TIMEOUT_SEC (default 30)

RUN_CPUS (default 1.0)

RUN_MEMORY (default 768m)

RUN_PIDS (default 256)

RUNNER_IMAGE (default trading-runner:latest)

WORK_ROOT (default workspaces)

---

## Running
To run the monitor:
```bash
python market_monitor.py
```
### Minimal .env example:

APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
ALPACA_DATA_FEED=IEX
SYMBOLS=SPY,QQQ,AAPL,NVDA
OUT_DIR=./data_store
LOG_QUOTES=false
VWAP_EXTREME_DIST_PCT=1.25
VWAP_NEAR_PCT=0.10
REGIME_SYMBOL=""       #(Optional)

### Notes / Design decisions

No partial candles: features + eligibility update only on CLOSED 5m/15m/30m/1h bars.

VWAP is intraday reset: VWAP accumulators reset on NY trading day change.

Quotes are context-only: they do not affect features; they are logged with session state for traceability.

---

## Troubleshooting

### Missing keys

If you see: Missing APCA_API_KEY_ID... or Missing APCA_API_SECRET_KEY...

> Ensure .env exists and python-dotenv is installed or export the vars in your shell session

### No output files

Check OUT_DIR path and permissions.

Verify the process is receiving bars (feed and subscription symbols correct).

### Data feed issues

ALPACA_DATA_FEED=IEX is easiest for most accounts.

---

## License

> MIT License. Feel free to fork and adapt for your own trading experiments and research.