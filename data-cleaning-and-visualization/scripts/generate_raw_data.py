"""Generate a messy, raw e-commerce sales dataset.

Intentionally injects realistic data-quality problems:
- Missing values (NaN) in several columns
- Duplicate rows
- Outliers (unrealistic prices / quantities)
- Inconsistent strings / typos in categorical columns
- Wrong data types (numbers stored as strings)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)

n = 1200
regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Home", "Books", "Sports"]
products = {
    "Electronics": ["Headphones", "Smartphone", "Laptop", "Camera"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sneakers"],
    "Home": ["Lamp", "Chair", "Cookware", "Curtains"],
    "Books": ["Novel", "Textbook", "Comics"],
    "Sports": ["Yoga Mat", "Dumbbells", "Skipping Rope", "Bicycle"],
}

rows = []
for i in range(n):
    cat = rng.choice(categories)
    prod = rng.choice(products[cat])
    region = rng.choice(regions)
    qty = int(rng.integers(1, 10))
    # ~10x overpriced outliers for some electronics
    if cat == "Electronics" and rng.random() < 0.04:
        price = round(rng.uniform(4000, 12000), 2)
    else:
        base = {"Headphones": 80, "Smartphone": 700, "Laptop": 1200, "Camera": 600,
                "T-Shirt": 15, "Jeans": 40, "Jacket": 90, "Sneakers": 60,
                "Lamp": 25, "Chair": 75, "Cookware": 55, "Curtains": 30,
                "Novel": 12, "Textbook": 45, "Comics": 8,
                "Yoga Mat": 20, "Dumbbells": 35, "Skipping Rope": 10, "Bicycle": 400}
        price = round(rng.normal(base[prod], base[prod] * 0.25), 2)
        price = max(1.0, price)

    days_ago = int(rng.integers(0, 365))
    date = (pd.Timestamp("2025-12-31") - pd.Timedelta(days=days_ago)).date().isoformat()
    rows.append([region, cat, prod, price, qty, date])

df = pd.DataFrame(rows, columns=["region", "category", "product", "price", "quantity", "order_date"])

# --- Inject data-quality problems ---
# Missing values
df.loc[rng.choice(df.index, size=120, replace=False), "price"] = np.nan
df.loc[rng.choice(df.index, size=60, replace=False), "quantity"] = np.nan
df.loc[rng.choice(df.index, size=80, replace=False), "order_date"] = np.nan
df.loc[rng.choice(df.index, size=40, replace=False), "region"] = np.nan

# Duplicate rows
dup = df.sample(n=40, replace=True, random_state=7)
df = pd.concat([df, dup], ignore_index=True)

# Inconsistent / dirty categorical strings
df.loc[df.sample(25).index, "region"] = "north "
df.loc[df.sample(20).index, "region"] = "WEST"
df.loc[df.sample(15).index, "category"] = "Electronics "
df.loc[df.sample(10).index, "category"] = "clothing"

# Wrong data type: price stored as string for some rows
price_as_str = df.loc[df.sample(50).index, "price"].map(
    lambda p: f"${p:.2f}" if not pd.isna(p) else np.nan
)
df["price"] = df["price"].astype(object)
df.loc[price_as_str.index, "price"] = price_as_str

# Negative quantity outlier
df.loc[df.sample(8).index, "quantity"] = -1

df = df.sample(frac=1, random_state=3).reset_index(drop=True)

out = ROOT / "data" / "raw_sales_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows to {out}")
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())