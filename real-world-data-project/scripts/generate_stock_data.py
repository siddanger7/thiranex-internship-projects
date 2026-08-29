"""Generate a realistic daily OHLCV dataset for four stocks.

Prices are simulated with Geometric Brownian Motion (GBM), the standard
model behind stock price movement, so the data has fat tails, clustered
volatility, and drift — realistic enough for applied analysis.

Stocks: AAPL, MSFT, GOOGL, AMZN  (2 years of trading days)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(77)

tickers = {
    # ticker: (start_price, annual_drift, annual_vol)
    "AAPL": (172.0, 0.18, 0.24),
    "MSFT": (370.0, 0.24, 0.22),
    "GOOGL": (140.0, 0.28, 0.30),
    "AMZN": (150.0, 0.30, 0.32),
}

dates = pd.bdate_range("2024-01-02", "2026-01-02")
n = len(dates)
dt = 1 / 252

frames = []
for ticker, (s0, mu, sigma) in tickers.items():
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=n)
    # allow one regime shift (crash) for realism in AMZN
    if ticker == "AMZN":
        crash = pd.Series(shocks).index[(pd.Series(shocks).index > 200) & (pd.Series(shocks).index < 210)]
        if len(crash):
            shocks[crash[0]:crash[0] + 10] -= 0.04

    close = s0 * np.exp(np.cumsum(shocks))
    open_ = close * np.exp(rng.normal(0, 0.008, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    volume = rng.integers(20_000_000, 120_000_000, n)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume,
        }
    )
    df.to_csv(ROOT / "data" / f"{ticker}.csv", index=False)
    frames.append(df.assign(ticker=ticker))

combined = pd.concat(frames, ignore_index=True)
combined.to_csv(ROOT / "data" / "stocks_all.csv", index=False)

print(f"Generated {n} trading days x {len(tickers)} tickers -> data/")
for t in tickers:
    s = pd.read_csv(ROOT / "data" / f"{t}.csv")
    print(f"  {t}: start=${s['close'].iloc[0]:.2f} end=${s['close'].iloc[-1]:.2f} "
          f"return={(s['close'].iloc[-1]/s['close'].iloc[0]-1)*100:+.1f}%")