# Exploratory Data Analysis (EDA) Project

A structured EDA on a **student-performance dataset** (2,000 students, 13 attributes).
The goal is to uncover patterns and the factors that most influence exam results,
and to present them as a structured report with supporting visuals.

## 📁 Folder Structure

```
exploratory-data-analysis/
├── README.md
├── requirements.txt
├── data/
│   └── student_performance.csv      # generated dataset
├── scripts/
│   ├── generate_student_data.py     # builds the dataset (realistic built-in patterns)
│   └── eda.py                       # runs the full EDA pipeline
└── output/
    ├── eda_report.md                # structured insights report (main deliverable)
    ├── statistical_summary.txt
    ├── 01_score_distributions.png
    ├── 02_study_attendance_distributions.png
    ├── 03_score_by_gender.png
    ├── 04_score_by_parental_education.png
    ├── 05_score_by_internet.png
    ├── 06_attendance_vs_score.png
    ├── 07_study_time_vs_score.png
    ├── 08_correlation_heatmap.png
    └── 09_key_factors.png
```

## 🚀 How to Run

```bash
pip install -r requirements.txt

python scripts/generate_student_data.py   # (optional) recreate dataset
python scripts/eda.py                     # full EDA -> output/
```

> Run from inside the `exploratory-data-analysis/` folder.

## 🔍 What the EDA Covers

1. **Overview & data quality** — shape, dtypes, missing values, duplicates
2. **Statistical summaries** — `describe()`, value counts, means/medians
3. **Univariate analysis** — KDE histograms of scores, study time, attendance
4. **Group comparisons** — boxplots of scores by gender, parental education, internet access
5. **Relationships** — scatter + regression of attendance / study time vs average score
6. **Correlations** — heatmap + ranked "key influencing factors" chart
7. **Structured report** — `output/eda_report.md` with a ranked list of drivers and takeaways

## ✨ Key Findings (summary)

- **Study time** is the strongest factor: correlation **0.80** with average score;
  students studying ≥15 hrs/wk score ~**20 pts higher** than those studying <5 hrs.
- **Attendance** is the 2nd strongest factor: correlation **0.48**.
- **Failures** are penalizing: correlation **−0.20**; school support offsets part of the penalty (+2.2 pts).
- Subject-level gender differences: males score higher math, females higher reading/writing.
- No missing values, no duplicates → clean dataset, free for modeling.

## 🛠 Tools

- Python 3.12, Pandas, NumPy, Matplotlib, Seaborn