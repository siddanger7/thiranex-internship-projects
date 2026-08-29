# Return Prediction - Colophon

Task: predict the **next-day return (%)** of MSFT from today's features.

```
                  Model   RMSE    MAE      R2  Dir Acc
     Naive (predict 0%) 1.3638 1.1011 -0.0000   0.0000
      Linear Regression 1.3739 1.1172 -0.0149   0.4851
          Random Forest 1.4072 1.1429 -0.0646   0.4653
Naive (yesterday's ret) 2.0089 1.6400 -1.1697   0.4554
```

## Takeaway

Daily returns are close to **unpredictable** (R² near 0), and machine-learning models barely beat naive baselines — consistent with the efficient-market view. The value is in the workflow: feature engineering, walk-forward splits, baseline comparison, and honest error metrics. Trend/volatility analysis (see `analysis_report.md`) is more actionable than point-forecasting next-day returns.
