"""Predict next-day stock returns (finance prediction task).

Question: can today's market features predict tomorrow's return?

- Engineered features: lagged returns (1-5d), volume change, momentum ratio
- Models: Linear Regression, Random Forest
- Naive baselines: "predict 0%", "predict yesterday's return"
- Walk-forward split: train on first 80%, test on last 20%
- Evaluation: RMSE, MAE, Direction Accuracy (does the sign match?)
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
os.makedirs(OUT, exist_ok=True)

TICKER = "MSFT"
df = pd.read_csv(ROOT / "data" / f"{TICKER}.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df["ret"] = df["close"].pct_change() * 100  # percentage returns
df["vol_change"] = df["volume"].pct_change() * 100

feat = pd.DataFrame(index=df.index)
for l in range(1, 6):
    feat[f"ret_lag{l}"] = df["ret"].shift(l)
feat["vol_change"] = df["vol_change"].shift(1)
ma20 = df["close"].rolling(20).mean()
feat["mom_20_ratio"] = (df["close"] / ma20 - 1) * 100  # momentum vs 20d MA
feat["range_pct"] = ((df["high"] - df["low"]) / df["close"]) * 100

target = df["ret"].shift(-1)  # next-day return
X = feat.iloc[:-1]
y = target.iloc[:-1].astype(float)
X = X.replace([np.inf, -np.inf], np.nan)
ok = X.notna().all(axis=1)
X, y = X[ok], y[ok]

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42),
}

baselines = {
    "Naive (predict 0%)": np.zeros_like(y_test),
    "Naive (yesterday's ret)": np.where(np.isnan(feat["ret_lag1"].iloc[:-1][ok].values[split:]), 0, feat["ret_lag1"].iloc[:-1][ok].values[split:]),
}

results, lines, roc = [], [], []
fig, ax = plt.subplots(figsize=(7, 7))
lim = max(abs(y_test.min()), abs(y_test.max()))
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="Perfect fit")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    direction = float(np.mean(np.sign(y_pred) == np.sign(y_test)))
    results.append({"Model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4), "Dir Acc": round(direction, 4)})
    lines.append(f"{name}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} dir_acc={direction:.3f}")
    ax.scatter(y_test, y_pred, alpha=0.4, s=16, label=name)
    if name == "Random Forest":
        rf = model

for name, pred in baselines.items():
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    direction = float(np.mean(np.sign(pred) == np.sign(y_test)))
    results.append({"Model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4), "Dir Acc": round(direction, 4)})
    lines.append(f"{name}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} dir_acc={direction:.3f}")

ax.set_xlabel("Actual next-day return (%)")
ax.set_ylabel("Predicted next-day return (%)")
ax.set_title(f"{TICKER} - Predicted vs Actual Next-Day Returns")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "07_returns_pred_vs_actual.png", dpi=150)
plt.close()

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(OUT / "prediction_comparison.csv", index=False)

# time series of best model's test predictions
fig, ax = plt.subplots(figsize=(11, 5))
test_dates = df["date"].iloc[:-1][ok].values[split:]
lr = models["Linear Regression"].predict(X_test)
ax.plot(test_dates, y_test.values, label="Actual", color="black", lw=1.2)
ax.plot(test_dates, lr, label="Linear Regression", color="steelblue", alpha=0.9)
ax.plot(test_dates, rf.predict(X_test), label="Random Forest", color="green", alpha=0.8)
ax.set_title(f"{TICKER} - Actual vs Predicted Next-Day Returns (test window)")
ax.set_xlabel("Date")
ax.set_ylabel("Return (%)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "08_returns_time_series.png", dpi=150)
plt.close()

# feature importance
fig, ax = plt.subplots(figsize=(8, 5))
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=imp.values, y=imp.index, hue=imp.index, palette="rocket", legend=False, ax=ax)
ax.set_title("Feature Importance - Random Forest Return Model")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(OUT / "09_return_feature_importance.png", dpi=150)
plt.close()

with open(OUT / "prediction_report.md", "w", encoding="utf-8") as f:
    f.write("# Return Prediction - Colophon\n\n")
    f.write(f"Task: predict the **next-day return (%)** of {TICKER} from today's features.\n\n")
    f.write("```\n")
    f.write(results_df.to_string(index=False))
    f.write("\n```\n\n")
    f.write("## Takeaway\n\n")
    f.write("Daily returns are close to **unpredictable** (R² near 0), and machine-learning "
            "models barely beat naive baselines — consistent with the efficient-market view. "
            "The value is in the workflow: feature engineering, walk-forward splits, baseline "
            "comparison, and honest error metrics. Trend/volatility analysis (see `analysis_report.md`) "
            "is more actionable than point-forecasting next-day returns.\n")

print(f"Results:\n{results_df.to_string(index=False)}")
print(f"\nSaved prediction charts + report to {OUT}")