# Predictive Modeling Using Machine Learning

Predict customer churn and monthly charges with **scikit-learn**.
This project demonstrates the full supervised-learning workflow:

1. Generate a realistic telecom-customer dataset (with injected missing values)
2. Preprocess: impute, encode, scale, train/test split
3. Train + compare **Logistic Regression**, **Decision Tree**, **Random Forest**
4. Evaluate with accuracy / precision / recall / F1 and **confusion matrices + ROC curves**
5. Regression task: **Linear Regression** vs **Random Forest Regressor** (R², RMSE, residuals)

## 📁 Folder Structure

```
predictive-modeling-machine-learning/
├── README.md
├── requirements.txt
├── data/
│   ├── churn_data.csv              # raw customer data (with missing values)
│   └── preprocessed/               # imputed, encoded, scaled + train/test splits
├── scripts/
│   ├── generate_churn_data.py      # builds the synthetic dataset
│   ├── data_preprocessing.py       # impute / one-hot encode / standardize / split
│   ├── classification_models.py    # LR, Decision Tree, Random Forest
│   └── regression_models.py        # Linear Regression + Random Forest Regressor
└── output/
    ├── classification_report.txt
    ├── model_comparison.csv        # accuracy, precision, recall, F1, AUC
    ├── cm_*.png                    # confusion matrix per model
    ├── roc_curves.png              # ROC curves (all models overlaid)
    ├── feature_importance.png
    ├── regression_report.txt
    ├── regression_comparison.csv   # R2, RMSE, MAE
    ├── pred_vs_actual.png
    └── residuals_*.png
```

## 🚀 How to Run

```bash
pip install -r requirements.txt

python scripts/generate_churn_data.py    # 1. (optional) recreate dataset
python scripts/data_preprocessing.py     # 2. clean + split
python scripts/classification_models.py  # 3. classification + evaluation
python scripts/regression_models.py      # 4. regression + evaluation
```

> Run from inside the `predictive-modeling-machine-learning/` folder.

## 📊 Classification Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.810 | 0.698 | 0.367 | 0.481 | **0.813** |
| Random Forest | 0.800 | 0.685 | 0.308 | 0.425 | 0.796 |
| Decision Tree | 0.784 | 0.579 | 0.367 | 0.449 | 0.757 |

- **Logistic Regression** wins on ROC AUC; Random Forest wins the accuracy/residual balance.
- Low recall reflects the class imbalance (~24% churn) — flagged for follow-up (resampling, class weights).
- `roc_curves.png` overlays all three models; `cm_*.png` shows the misclassification heatmaps.

## 📈 Regression Results (predict monthly charges)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | **0.873** | 10.53 | 7.80 |
| Random Forest Regressor | 0.866 | 10.82 | 8.17 |

## 🛠 Tools

- Python 3.12, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn