"""Regression: predict monthly charges (numeric target).

Models compared:
- Linear Regression
- Random Forest Regressor

Evaluation:
- R2, Root-Mean-Squared Error (RMSE), Mean Absolute Error (MAE)
- predicted-vs-actual scatter plots
- residual histograms
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
DATA = ROOT / "data" / "preprocessed"
OUT = ROOT / "output"
os.makedirs(OUT, exist_ok=True)

X_train = pd.read_csv(DATA / "X_train_sc.csv")
X_test = pd.read_csv(DATA / "X_test_sc.csv")
y_train = pd.read_csv(DATA / "y_reg_train.csv").squeeze()
y_test = pd.read_csv(DATA / "y_reg_test.csv").squeeze()

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42),
}

results, lines = [], []
fig, ax = plt.subplots(figsize=(7, 7))
vals = np.linspace(y_test.min(), y_test.max(), 100)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = mean_absolute_error(y_test, y_pred)

    results.append({"Model": name, "R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
    lines.append(f"{name}: R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f}")

    # predicted vs actual
    ax.scatter(y_test, y_pred, alpha=0.35, s=18, label=f"{name}")
    ax.plot(vals, vals, "k--", lw=1)

    # residuals
    fig_r, ax_r = plt.subplots(figsize=(7, 4.5))
    sns.histplot(y_test - y_pred, bins=40, kde=True, color="steelblue", ax=ax_r)
    ax_r.axvline(0, color="red", linestyle="--")
    ax_r.set_title(f"Residuals - {name}\n(R2={r2:.3f}, RMSE={rmse:.2f})")
    ax_r.set_xlabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(OUT / f"residuals_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()

ax.set_xlabel("Actual monthly charges")
ax.set_ylabel("Predicted monthly charges")
ax.set_title("Predicted vs Actual - Regression Models")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "pred_vs_actual.png", dpi=150)
plt.close()

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
results_df.to_csv(OUT / "regression_comparison.csv", index=False)

with open(OUT / "regression_report.txt", "w", encoding="utf-8") as f:
    f.write("REGRESSION RESULTS - PREDICT MONTHLY CHARGES\n")
    f.write("=" * 60 + "\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nBest model by R2: {results_df.iloc[0]['Model']}")

print("\n".join(lines))
print("\nBest model by R2:", results_df.iloc[0]["Model"])
print("Saved pred_vs_actual.png, residuals, and regression_comparison.csv")