"""Data Cleaning pipeline for the raw e-commerce sales dataset.

Steps performed:
1. Load raw data + initial inspection
2. Use consistent dtypes (read 'price' as string to catch $ formatting, parse dates)
3. Drop duplicate rows
4. Standardise categorical strings (strip whitespace, fix case)
5. Correct wrong data types ($-prefixed prices -> float, dates -> datetime)
6. Handle missing values (impute with median / most frequent)
7. Remove / fix outliers (negative quantities, IQR-based price caps)
8. Export a clean dataset + a cleaning report
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_sales_data.csv"
CLEAN = ROOT / "data" / "clean_sales_data.csv"
REPORT = ROOT / "output" / "cleaning_report.txt"

df = pd.read_csv(RAW, dtype={"price": str})


def to_float_price(s):
    """Read a price cell that may be '1234.5', '$45.99', or '' and return float / NaN."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lstrip("$").replace(",", "")
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


report = []
report.append("=" * 60)
report.append("DATA CLEANING REPORT")
report.append("=" * 60)
report.append(f"Raw data shape: {df.shape}")
report.append(f"Duplicate rows before cleaning: {df.duplicated().sum()}")
report.append(f"Missing values before cleaning:\n{df.isna().sum().to_string()}")

# 1. Drop duplicates
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
report.append(f"\nDropped {before - len(df)} duplicate rows. New shape: {df.shape}")

# 2. Standardise categorical strings
df["region"] = df["region"].str.strip().str.title()
df["category"] = df["category"].str.strip().str.title()
report.append(f"\nUnique regions: {df['region'].unique().tolist()}")
report.append(f"Unique categories: {df['category'].unique().tolist()}")

# 3. Fix wrong data types
df["price"] = df["price"].map(to_float_price)
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
report.append(f"\nAfter type conversion:\n{df[['price', 'quantity', 'order_date']].dtypes.to_string()}")

# 4. Handle missing values
report.append(f"\nMissing values before imputation:\n{df.isna().sum().to_string()}")

# quantity: negative values are invalid -> set to NaN before imputing
df.loc[df["quantity"] < 0, "quantity"] = np.nan
report.append("Negative quantities set to NaN (invalid).")

# impute numerics with median
df["price"] = df["price"].fillna(df["price"].median())
df["quantity"] = df["quantity"].fillna(df["quantity"].median().round())

# impute categoricals with most frequent
df["region"] = df["region"].fillna(df["region"].mode()[0])
df["order_date"] = df["order_date"].fillna(df["order_date"].mode()[0])

# imputation can recreate identical rows -> dedupe again
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
report.append(f"Re-deduplicated after imputation (removed {before - len(df)} rows).")

report.append(f"Missing values after cleaning:\n{df.isna().sum().to_string()} (all 0 expected)")

# 5. Handle outliers with IQR on price
q1, q3 = df["price"].quantile(0.25), df["price"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = int(((df["price"] < lower) | (df["price"] > upper)).sum())
df = df[(df["price"] >= lower) & (df["price"] <= upper)].reset_index(drop=True)
report.append(f"\nPrice outlier bounds (IQR): lower={lower:.2f}, upper={upper:.2f}")
report.append(f"Removed {n_out} outliers. New shape: {df.shape}")

report.append(f"\nFinal shape: {df.shape}")
report.append(f"Final duplicates: {df.duplicated().sum()}")
report.append(f"Final missing values: {df.isna().sum().sum()}")
report.append(f"\nSummary stats:\n{df.describe().to_string()}")

df.to_csv(CLEAN, index=False)
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print("\n".join(report))
print(f"\nSaved clean data to {CLEAN}")
print(f"Saved cleaning report to {REPORT}")