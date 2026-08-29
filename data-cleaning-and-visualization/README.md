# Data Cleaning & Visualization Project

A complete data processing pipeline applied to a raw, messy e-commerce sales dataset.
The goal is to demonstrate the full data-analysis workflow:

1. **Clean** a raw dataset (missing values, duplicates, wrong dtypes, inconsistent strings, outliers)
2. **Analyze** it with Pandas
3. **Visualize** the findings with Matplotlib & Seaborn
4. **Tell a story** with a written insights report

## 📁 Folder Structure

```
data-cleaning-and-visualization/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_sales_data.csv        # messy raw dataset (deliberately flawed)
│   └── clean_sales_data.csv      # output of the cleaning pipeline
├── scripts/
│   ├── generate_raw_data.py      # creates the raw dataset (with injected problems)
│   ├── data_cleaning.py          # cleaning pipeline + cleaning report
│   └── visualization.py          # charts + insights report
└── output/
    ├── cleaning_report.txt       # step-by-step cleaning log
    ├── insights.md               # written summary of findings
    ├── 01_revenue_by_region.png
    ├── 02_sales_by_category.png
    ├── 03_price_distribution.png
    ├── 04_monthly_trend.png
    ├── 05_top_products.png
    └── 06_correlation_heatmap.png
```

## 🚀 How to Run

```bash
pip install -r requirements.txt

# 1. (optional) regenerate the raw dataset
python scripts/generate_raw_data.py

# 2. run the cleaning pipeline
python scripts/data_cleaning.py

# 3. generate charts + insights
python scripts/visualization.py
```

> Run from inside the `data-cleaning-and-visualization/` folder.

## 🧹 Cleaning Techniques Used

| Problem | Technique |
|---|---|
| Duplicates | `df.drop_duplicates()` |
| Missing numerical values | Impute with column median |
| Missing categorical values | Impute with most frequent value |
| Negative / invalid quantities | Replace with NaN, then impute |
| Outliers | IQR method (1.5 × IQR fence) on price |
| Wrong data types | Typed parsing (money strings, dates, numerics) |
| Inconsistent strings | `strip()` + `title()` casing |

## 📊 Visualizations

| Chart | Reveals |
|---|---|
| Revenue by Region | Which regions drive the most sales |
| Revenue by Category | Best-performing product categories |
| Price Distribution (raw vs clean) | Impact of outlier removal |
| Monthly Revenue Trend | Seasonality / growth over time |
| Top 10 Products | Best sellers |
| Correlation Heatmap | Relationship between numeric variables |

## 🔑 Key Insights (summary)

Top findings are written to `output/insights.md` and summarized in the images below.

## 🛠 Tools

- Python 3.12, Pandas, NumPy, Matplotlib, Seaborn