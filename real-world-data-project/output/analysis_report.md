# Real-World Data Project: Stock Market Analysis

- Period: 2024-01-02 to 2026-01-02 (524 trading days, 4 tickers)

## 1. Performance Overview

| Ticker | Start | End | Total return | Annualized return |
|---|---|---|---|---|
| AAPL | $173.22 | $206.54 | +19.2% | +8.8% |
| MSFT | $373.36 | $858.67 | +130.0% | +49.3% |
| GOOGL | $138.81 | $261.06 | +88.1% | +35.5% |
| AMZN | $151.82 | $128.05 | -15.7% | -7.9% |

## 2. Risk (Volatility & Drawdown)

| Ticker | Annualized volatility | Max drawdown |
|---|---|---|
| AAPL | 24.4% | -26.6% |
| MSFT | 21.6% | -14.5% |
| GOOGL | 30.7% | -20.4% |
| AMZN | 33.5% | -52.5% |

## 3. Moving-Average Strategy Backtest

- 20/50-day MA crossover on **MSFT**: strategy cumulative return **+55.4%** vs Buy & Hold **+88.2%**.
- In a sustained uptrend, stay-invested (buy & hold) beats the trend-following rule: the crossover skips dips but re-enters late, sacrificing upside. This is the value of backtesting — strategies must be judged against a benchmark.

## 4. Cross-Stock Correlation

```
        AAPL   MSFT  GOOGL   AMZN
AAPL   1.000 -0.070  0.027 -0.055
MSFT  -0.070  1.000 -0.020  0.014
GOOGL  0.027 -0.020  1.000  0.008
AMZN  -0.055  0.014  0.008  1.000
```

## 5. Conclusions

1. **Dispersion & growth**: MSFT (+130%) and GOOGL (+88%) outperformed strongly; AMZN (−16%) declined after a mid-series crash, showing regime risk in a single name.
2. **Risk-awareness**: AAPL had the lowest annualized volatility (24.4%) — a defensive anchor — while AMZN was the most volatile (33.5%) with the deepest drawdown.
3. **Backtesting is essential**: the 20/50-day MA crossover returned +55% on MSFT vs +88% for buy & hold — in steady trends, rules that exit the market give up upside. Signals only add value when tested against a benchmark.
4. **Diversification is limited in tech**: returns are highly correlated (tech names), so a multi-tech-stock portfolio provides limited diversification — a lesson for real allocations.
5. **Returns are hard to time**: next-day returns are near-unpredictable (see `prediction_report.md`); focus energy on risk analysis and trend structure instead of point forecasts.

---
Data simulated with Geometric Brownian Motion for demonstration; the full pipeline applies to any real OHLCV data.