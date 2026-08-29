"""Classification: predict customer churn.

Models compared:
- Logistic Regression
- Decision Tree  (max_depth tuned)
- Random Forest  (n_estimators, max_depth tuned)

Evaluation:
- accuracy / precision / recall / F1
- confusion matrices (heatmaps)
- ROC curves + AUC
- feature importances (tree-based)
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "preprocessed"
OUT = ROOT / "output"
os.makedirs(OUT, exist_ok=True)

X_train = pd.read_csv(DATA / "X_train_sc.csv")
X_test = pd.read_csv(DATA / "X_test_sc.csv")
y_train = pd.read_csv(DATA / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA / "y_test.csv").squeeze()

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=42),
}

results, lines, roc_handles = [], [], []
plt.figure(figsize=(8, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append(
        {"Model": name, "Accuracy": round(acc, 4), "Precision": round(prec, 4),
         "Recall": round(rec, 4), "F1": round(f1, 4), "ROC AUC": round(auc, 4)}
    )
    lines.append(f"{name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} auc={auc:.4f}")

    # confusion matrix
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"], ax=ax)
    ax.set_title(f"Confusion Matrix - {name}\nAcc={acc:.3f}, AUC={auc:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT / f"cm_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_handles.append(plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")[0])

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Churn Prediction")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "roc_curves.png", dpi=150)
plt.close()

# feature importance (Random Forest)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=imp.values, y=imp.index, hue=imp.index, palette="rocket", legend=False, ax=ax)
ax.set_title("Top Feature Importances - Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()

results_df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False)
results_df.to_csv(OUT / "model_comparison.csv", index=False)

with open(OUT / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write("CLASSIFICATION RESULTS - CHURN PREDICTION\n")
    f.write("=" * 55 + "\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nBest model by ROC AUC: {results_df.iloc[0]['Model']}")
    f.write(f"\nTop 5 features:\n{imp.head(5).to_string()}")

print("\n".join(lines))
print("\nBest model by ROC AUC:", results_df.iloc[0]["Model"])
print("Saved confusion matrices, ROC curve, feature importance, and model_comparison.csv")