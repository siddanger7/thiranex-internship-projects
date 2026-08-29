"""Preprocess the churn dataset for modeling.

Steps:
1. Load raw data, inspect missing values & class balance
2. Impute missing numerics with the median
3. Drop the derived 'total_charges' column (perfectly collinear sub of monthly*tenure)
4. One-hot encode categoricals
5. Split into train/test (stratified, 80/20)
6. Standardize numeric features (fit on train only)
7. Save prep data to disk
"""

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "churn_data.csv"
OUT = ROOT / "data" / "preprocessed"

df = pd.read_csv(SRC)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
print("Missing values:\n", df.isna().sum())
print("Class balance:\n", df["churned"].value_counts(normalize=True).round(4).to_string())

# impute numerics with median
for col in ["monthly_charges", "support_calls", "total_charges"]:
    df[col] = df[col].fillna(df[col].median())

# 'total_charges' is exactly monthly_charges * tenure_months -> collinear, drop it
df = df.drop(columns=["total_charges"])

y = df["churned"]
y_reg = df["monthly_charges"]
X = df.drop(columns=["churned", "monthly_charges"])

# one-hot encode categoricals
cat_cols = X.select_dtypes(include="object").columns.tolist()
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False).astype(int)
X_enc = X_enc.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42, stratify=y
)

# standardize
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_enc.columns)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_enc.columns)

OUT.mkdir(parents=True, exist_ok=True)
for name, obj in [
    ("X_train_sc.csv", X_train_sc),
    ("X_test_sc.csv", X_test_sc),
    ("y_train.csv", y_train),
    ("y_test.csv", y_test),
]:
    obj.to_csv(OUT / name, index=False)

# regression target (monthly charges) aligned to the same splits
reg_targets = pd.DataFrame({"monthly_charges": y_reg})
reg_train, reg_test = train_test_split(
    reg_targets, test_size=0.2, random_state=42, stratify=df["churned"]
)
reg_train["monthly_charges"].to_csv(OUT / "y_reg_train.csv", index=False)
reg_test["monthly_charges"].to_csv(OUT / "y_reg_test.csv", index=False)

print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
print(f"Saved preprocessed data to {OUT}")