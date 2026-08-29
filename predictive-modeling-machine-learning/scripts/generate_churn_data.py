"""Generate a synthetic customer-churn dataset.

The dataset describes telecom customers and whether they churned.
A realistic-but-known relationship with the target is baked in so the
models can learn meaningful patterns, while noise keeps the task honest.

Columns:
- tenure_months, monthly_charges, total_charges (numeric)
- age, num_contracts, support_calls (numeric)
- plan_type (categorical), payment_method (categorical), region (categorical)
- churned (target: 0 = stayed, 1 = churned)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(2026)

n = 2500
age = rng.integers(18, 70, n).astype(float)
tenure = rng.integers(0, 72, n).astype(float)
support_calls = rng.integers(0, 15, n).astype(float)
num_contracts = rng.integers(1, 6, n).astype(float)

plan_type = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.5, 0.35, 0.15])
payment_method = rng.choice(["Credit Card", "Bank Transfer", "Cash", "Prepaid"], n, p=[0.3, 0.3, 0.2, 0.2])
region = rng.choice(["North", "South", "East", "West"], n)

# monthly charges are tied to plan / region / age / contracts -> learnable signal
plan_charge = {"Basic": 30.0, "Standard": 65.0, "Premium": 100.0}
region_effect = {"North": 1.0, "South": 1.15, "East": 0.9, "West": 1.05}
monthly_charges = (
    np.array([plan_charge[p] for p in plan_type])
    * np.array([region_effect[r] for r in region])
    + 0.15 * (age - 44)
    + 1.5 * num_contracts
    + rng.normal(0, 8.0, n)
)
monthly_charges = monthly_charges.round(2)
total_charges = (monthly_charges * tenure).round(2)

df = pd.DataFrame(
    {
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_calls": support_calls,
        "num_contracts": num_contracts,
        "plan_type": plan_type,
        "payment_method": payment_method,
        "region": region,
    }
)

df["plan_premium"] = (df["plan_type"] == "Premium").astype(int)
df["plan_basic"] = (df["plan_type"] == "Basic").astype(int)
df["prepaid"] = (df["payment_method"] == "Prepaid").astype(int)

logit = (
    -3.0
    + 0.25 * df["support_calls"]
    - 0.06 * df["tenure_months"]
    + 0.02 * df["monthly_charges"]
    + 0.6 * df["plan_basic"]
    - 0.8 * df["plan_premium"]
    + 0.7 * df["prepaid"]
    + rng.normal(0, 1.2, n)
)
prob = 1 / (1 + np.exp(-logit))
df["churned"] = (rng.random(n) < prob).astype(int)

df = df.drop(columns=["plan_premium", "plan_basic", "prepaid"])
churn_rate = df["churned"].mean()
print(f"Generated {len(df)} customers. Churn rate: {churn_rate:.1%}")

# small amount of realistic missingness
df.loc[rng.choice(df.index, size=120, replace=False), "monthly_charges"] = np.nan
df.loc[rng.choice(df.index, size=90, replace=False), "support_calls"] = np.nan
df.loc[df["monthly_charges"].isna(), "total_charges"] = np.nan
print("Missing values injected:\n", df.isna().sum())

out = ROOT / "data" / "churn_data.csv"
df.to_csv(out, index=False)
print(f"Saved to {out}")