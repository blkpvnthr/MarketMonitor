
# Alpaca Market Monitor (Stocks) — CLOSED-BAR ONLY ✅ + Live Quotes (visual/context)

This file integrates BOTH:

1) Real-time bar streaming (1m -> resample -> CLOSED 5m/15m/30m/1h) for decisions/features
2) Real-time quote streaming (bid/ask) + on-demand latest quote retrieval (historical client)

Guarantees:

- ONLY CLOSED bars influence decisions/features (eligibility, indicators, regimes).
- Quotes are for CONTEXT / UI / logging only (never used as signals here).
  (You can optionally use quotes for execution later; keep that separate.)

Writes:

- OUT_DIR/bars/{5m,15m,30m,1h}/SYMBOL_YYYYMMDD.csv
- OUT_DIR/features/{5m,15m,30m,1h}/SYMBOL_YYYYMMDD.csv
- OUT_DIR/regimes/market_regime.csv
- OUT_DIR/session_state/session_state_YYYYMMDD.csv (includes latest quote snapshot columns)
- OUT_DIR/quotes/quotes_YYYYMMDD.csv (optional: logs quote updates)

Env:
  export APCA_API_KEY_ID="..."
  export APCA_API_SECRET_KEY="..."
  export ALPACA_DATA_FEED="IEX"        # or "SIP" if entitled
  export SYMBOLS="SPY,QQQ,AAPL,MSFT,NVDA"
  export REGIME_SYMBOL="SPY"
  export OUT_DIR="./data_store"

Optional VWAP thresholds:
  export VWAP_EXTREME_DIST_PCT="1.25"  # 5m#1 extreme dislocation threshold
  export VWAP_NEAR_PCT="0.10"          # near-VWAP band

Optional quote logging:
  export LOG_QUOTES="0"                # set to 1 to append quote updates to OUT_DIR/quotes/quotes_YYYYMMDD.csv

# MarketMonitor
