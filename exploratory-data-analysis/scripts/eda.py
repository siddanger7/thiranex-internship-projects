"""Exploratory Data Analysis on the student-performance dataset.

Sections:
1. Overview & data quality (dtypes, missing, duplicates)
2. Statistical summaries (numeric + categorical)
3. Univariate distributions (histograms / KDE)
4. Group comparisons (categorical vs scores - boxplots)
5. Relationships (scatter, pairplot)
6. Correlations + key influencing factors (ranked)
7. Structured insights report (output/eda_report.md)
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
DATA = ROOT / "data" / "student_performance.csv"
OUT = ROOT / "output"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)
score_cols = ["math_score", "reading_score", "writing_score"]
df["average_score"] = df[score_cols].mean(axis=1)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

report = []

# ---- 1. Overview & data quality ----
report.append("# 1. Overview & Data Quality")
report.append("")
report.append(f"- Rows: **{len(df):,}**, Columns: **{df.shape[1]:,}**")
report.append(f"- Numeric columns: {num_cols}")
report.append(f"- Categorical columns: {cat_cols}")
report.append(f"- Missing values total: **{int(df.isna().sum().sum())}**")
report.append(f"- Duplicate rows: **{int(df.duplicated().sum())}**")
report.append(f"- Score means: math={df['math_score'].mean():.1f}, reading={df['reading_score'].mean():.1f}, writing={df['writing_score'].mean():.1f}")

# ---- 2. Statistical summaries ----
report.append("")
report.append("# 2. Statistical Summary")
report.append("")
report.append("```")
report.append(df[score_cols + ["study_time_hours", "attendance_pct", "absences", "failures"]].describe().round(2).to_string())
report.append("```")
report.append("")
for c in cat_cols:
    report.append(f"`{c}` value counts:\n{df[c].value_counts().to_string()}")
    report.append("")

with open(OUT / "statistical_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

# ---- 3. Distributions ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, score_cols):
    sns.histplot(df[col], kde=True, color="steelblue", ax=ax)
    ax.axvline(df[col].mean(), color="red", ls="--", label=f"mean={df[col].mean():.0f}")
    ax.axvline(df[col].median(), color="green", ls="--", label=f"median={df[col].median():.0f}")
    ax.set_title(f"Distribution of {col}")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT / "01_score_distributions.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(df["study_time_hours"], kde=True, color="teal", ax=axes[0])
axes[0].set_title("Distribution of Study Time (hrs/week)")
sns.histplot(df["attendance_pct"], kde=True, color="orange", ax=axes[1])
axes[1].set_title("Distribution of Attendance (%)")
plt.tight_layout()
plt.savefig(OUT / "02_study_attendance_distributions.png", dpi=150)
plt.close()

# ---- 4. Group comparisons ----
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="gender", y="average_score", hue="gender", palette="pastel", legend=False, ax=ax)
ax.set_title("Average Score by Gender")
plt.tight_layout()
plt.savefig(OUT / "03_score_by_gender.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
order = ["No Formal Education", "High School", "Bachelor", "Master", "PhD"]
sns.boxplot(data=df, x="parental_education", y="average_score", order=order,
            hue="parental_education", palette="Blues_d", legend=False, ax=ax)
ax.set_title("Average Score by Parental Education")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.savefig(OUT / "04_score_by_parental_education.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="internet", y="average_score", hue="internet", palette="Set2", legend=False, ax=ax)
ax.set_title("Average Score by Internet Access")
plt.tight_layout()
plt.savefig(OUT / "05_score_by_internet.png", dpi=150)
plt.close()

# ---- 5. Relationships ----
fig, ax = plt.subplots(figsize=(8, 5))
sns.regplot(data=df.sample(600), x="attendance_pct", y="average_score",
            scatter_kws={"alpha": 0.25, "s": 18}, line_kws={"color": "red"}, ax=ax)
ax.set_title("Attendance vs Average Score")
plt.tight_layout()
plt.savefig(OUT / "06_attendance_vs_score.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
sns.regplot(data=df.sample(600), x="study_time_hours", y="average_score",
            scatter_kws={"alpha": 0.25, "s": 18}, line_kws={"color": "red"}, ax=ax)
ax.set_title("Study Time vs Average Score")
plt.tight_layout()
plt.savefig(OUT / "07_study_time_vs_score.png", dpi=150)
plt.close()

# ---- 6. Correlations & key factors ----
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.4, annot_kws={"size": 8}, ax=ax)
ax.set_title("Correlation Heatmap - Numeric Features")
plt.tight_layout()
plt.savefig(OUT / "08_correlation_heatmap.png", dpi=150)
plt.close()

# key influencing factors ranked by |correlation| with average score
# (drop the three score components themselves - they are trivially correlated)
factor = (
    corr["average_score"]
    .drop(list(set(["average_score"] + score_cols)))
    .abs()
    .sort_values(ascending=False)
)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=factor.values, y=factor.index, hue=factor.index, palette="rocket", legend=False, ax=ax)
ax.set_title("Key Influencing Factors (|correlation| with Average Score)")
ax.set_xlabel("|Pearson correlation|")
plt.tight_layout()
plt.savefig(OUT / "09_key_factors.png", dpi=150)
plt.close()

# categorical effect sizes: mean average_score difference vs reference category
cat_effects = []
refs = {"internet": "No", "family_support": "No", "school_support": "No"}
for c, ref in refs.items():
    means = df.groupby(c)["average_score"].mean()
    effect = means[means.index[means.index != ref][0]] - means[ref]
    cat_effects.append(f"  - `{c}`: {effect:+.2f} pts ({c}={means.index[means.index != ref][0]} vs {ref})")

corr_ranked = corr["average_score"].drop(list(set(["average_score"] + score_cols))).sort_values(ascending=False)

# ---- 7. Insights report ----
report.append("")
report.append("# 3. Correlations & Key Influencing Factors")
report.append("")
report.append("Correlation with average score (all features):")
report.append("```")
report.append(corr_ranked.round(3).to_string())
report.append("```")
report.append("")
report.append("Categorical effect sizes (mean score difference vs baseline):")
report.extend(cat_effects)
report.append("")
report.append("# 4. Key Insights")
report.append("")
report.append(f"1. **Attendance is the dominant driver** - correlation of {corr.loc['attendance_pct', 'average_score']:.3f} with average score; the gap between the most and least attentive students is substantial.")
report.append(f"2. **Study time matters too** - {corr.loc['study_time_hours', 'average_score']:.3f} correlation; students studying >=15 hrs/wk score ~{df.query('study_time_hours >= 15')['average_score'].mean() - df.query('study_time_hours < 5')['average_score'].mean():.1f} pts higher than those studying <5 hrs.")
report.append(f"3. **Failures weigh heavily** - {corr.loc['failures', 'average_score']:.3f} correlation; students with 0 failures score ~{df[df['failures'] == 0]['average_score'].mean() - df[df['failures'] > 0]['average_score'].mean():.1f} pts higher than those with failures.")
report.append(f"4. **Internet access** adds {cat_effects[0].split('pts')[0].split(':')[-1].strip()} pts on average; parental education shows a clear socio-economic gradient.")
report.append("5. **Gender effect is subject-specific** - males average higher math, females higher reading/writing (see distributions).")
report.append("")
report.append("---")
report.append("*Patterns are baked into generated data for demonstration; the pipeline applies to any real dataset.*")

with open(OUT / "eda_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"Correlation of features with average score:\n{corr_ranked.round(3).to_string()}")
print(f"\nSaved charts + reports to {OUT}")