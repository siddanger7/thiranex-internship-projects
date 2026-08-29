"""Generate visualisations + written summary of insights.

Outputs (saved to /output):
- 01_revenue_by_region.png        bar chart
- 02_sales_by_category.png        bar chart
- 03_price_distribution.png       histogram + boxplot before/after cleaning
- 04_monthly_trend.png            line chart of revenue over time
- 05_top_products.png             horizontal bar of top products by revenue
- 06_correlation_heatmap.png      numeric correlation heatmap
- insights.md                    markdown summary of key findings
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean_sales_data.csv"
RAW = ROOT / "data" / "raw_sales_data.csv"
OUT = ROOT / "output"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CLEAN, parse_dates=["order_date"])
df["revenue"] = df["price"] * df["quantity"]
df["month"] = df["order_date"].dt.to_period("M")

# sanity check
assert df["price"].isna().sum() == 0 and df["quantity"].isna().sum() == 0

fig, ax = plt.subplots(figsize=(8, 5))
rev_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
sns.barplot(x=rev_region.values, y=rev_region.index, ax=ax, hue=rev_region.index, palette="Blues_d", legend=False)
ax.set_title("Total Revenue by Region")
ax.set_xlabel("Revenue ($)")
ax.bar_label(ax.containers[0], fmt="%.0f")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_revenue_by_region.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
rev_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
sns.barplot(x=rev_cat.index, y=rev_cat.values, ax=ax, hue=rev_cat.index, palette="rocket", legend=False)
ax.set_title("Revenue by Category")
ax.set_ylabel("Revenue ($)")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_sales_by_category.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
raw = pd.read_csv(RAW, dtype={"price": str})
raw["price"] = raw["price"].str.replace("$", "", regex=False).astype(float)
sns.histplot(raw["price"].dropna(), bins=50, ax=axes[0], color="salmon")
axes[0].set_title("Price Distribution - RAW (with outliers)")
sns.histplot(df["price"], bins=50, ax=axes[1], color="steelblue")
axes[1].set_title("Price Distribution - CLEANED")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_price_distribution.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
monthly = df.groupby("month")["revenue"].sum()
sns.lineplot(x=monthly.index.astype(str), y=monthly.values, marker="o", ax=ax, color="teal")
ax.set_title("Monthly Revenue Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_monthly_trend.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
top = df.groupby("product")["revenue"].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top.values, y=top.index, ax=ax, hue=top.index, palette="Blues_d", legend=False)
ax.set_title("Top 10 Products by Revenue")
ax.set_xlabel("Revenue ($)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_top_products.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df[["price", "quantity", "revenue"]].corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap - Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_correlation_heatmap.png"), dpi=150)
plt.close()

# ---- insights.md ----
lines = [
    "# Sales Analytics - Key Insights",
    "",
    f"- **Dataset**: {len(df):,} cleaned transactions, 5 regions x 5 categories.",
    f"- **Total revenue**: ${df['revenue'].sum():,.2f}",
    f"- **Best region**: {rev_region.index[0]} (${rev_region.iloc[0]:,.2f})",
    f"- **Best category**: {rev_cat.index[0]} (${rev_cat.iloc[0]:,.2f})",
    f"- **Best product**: {top.index[0]} (${top.iloc[0]:,.2f})",
    f"- **Average order value**: ${df['revenue'].mean():,.2f}",
    f"- **Busiest periods**: monthly peak revenue trend shown in `04_monthly_trend.png`.",
    "- Price outliers (IQR method) were removed; see price distribution chart before/after.",
    "- `price` and `quantity` correlate strongly with `revenue` (by construction).",
]
with open(os.path.join(OUT, "insights.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nCharts saved to {OUT}")