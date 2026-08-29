# Real-World Data Project: Stock Market Analysis

End-to-end applied data science on **financial market data** (4 tech stocks over 2 years).
Combines time-series analysis, risk metrics, a trading-strategy backtest, and honest ML
return forecasting — the whole journey from raw OHLCV data to a conclusions report.

## 📁 Folder Structure

```
real-world-data-project/
├── README.md
├── requirements.txt
├── data/
│   ├── AAPL.csv / MSFT.csv / GOOGL.csv / AMZN.csv   # daily OHLCV
│   └── stocks_all.csv                               # combined
├── scripts/
│   ├── generate_stock_data.py   # GBM simulator (realistic price paths)
│   ├── stock_analysis.py        # returns, volatility, MAs, backtest, correlation
│   └── stock_prediction.py      # next-day-return forecasting + evaluation
└── output/
    ├── analysis_report.md       # structured findings + conclusions
    ├── prediction_report.md     # forecasting experiment + takeaway
    ├── prediction_comparison.csv
    ├── 01_cumulative_returns.png     06_correlation_heatmap.png
    ├── 02_returns_distributions.png  07_returns_pred_vs_actual.png
    ├── 03_volatility.png             08_returns_time_series.png
    ├── 04_moving_average_signals.png 09_return_feature_importance.png
    └── 05_strategy_backtest.png
```

## 🚀 How to Run

```bash
pip install -r requirements.txt

python scripts/generate_stock_data.py   # (optional) regenerate prices
python scripts/stock_analysis.py        # analysis + report
python scripts/stock_prediction.py      # forecasting + report
```

> Run from inside the `real-world-data-project/` folder.

## 📊 What Was Done

| Step | Method / Output |
|---|---|
| Data generation | Geometric Brownian Motion per ticker (OHLCV, 524 trading days) |
| Performance | Normalized cumulative returns chart + total/annualized returns table |
| Risk | Annualized volatility, max drawdown per stock |
| Trend analysis | 20/50-day moving-average crossover + buy/sell signals on MSFT |
| Strategy testing | MA-crossover backtest vs Buy & Hold benchmark |
| Diversification | Correlation heatmap of daily returns |
| Prediction | Next-day return forecast (Linear Regression, Random Forest) vs naive baselines |

## ✨ Key Findings

1. **Huge dispersion**: MSFT **+130%** and GOOGL **+88%**, while AMZN **−16%** after a simulated crash — sector-wide but very uneven.
2. **Risk varies**: AMZN most volatile (33.5% annualized), AAPL a defensive anchor (24.4%).
3. **Backtesting humbles rules**: MA crossover returned **+55%** on MSFT vs **+88%** buy & hold — in steady uptrends, exiting the market costs upside.
4. **Tech is correlated**: heatmap shows high pairwise return correlation → limited diversification.
5. **Returns are near-unpredictable**: ML models (R² ≈ 0) don't beat "predict 0%" — consistent with market efficiency; risk/trend analysis beats point forecasting.

## 🛠 Tools

- Python 3.12, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn