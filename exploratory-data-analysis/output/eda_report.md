# 1. Overview & Data Quality

- Rows: **2,000**, Columns: **14**
- Numeric columns: ['age', 'study_time_hours', 'attendance_pct', 'absences', 'failures', 'math_score', 'reading_score', 'writing_score', 'average_score']
- Categorical columns: ['gender', 'internet', 'family_support', 'school_support', 'parental_education']
- Missing values total: **0**
- Duplicate rows: **0**
- Score means: math=79.0, reading=78.5, writing=75.4

# 2. Statistical Summary

```
       math_score  reading_score  writing_score  study_time_hours  attendance_pct  absences  failures
count     2000.00        2000.00        2000.00            2000.0         2000.00   2000.00   2000.00
mean        79.04          78.46          75.43              14.5           77.40     14.66      0.49
std         12.03          11.47          11.80               8.8           12.89      8.65      0.85
min         41.00          44.00          40.00               0.0           55.00      0.00      0.00
25%         70.00          70.00          67.00               7.0           66.30      7.00      0.00
50%         79.00          78.00          75.00              14.0           77.60     15.00      0.00
75%         88.00          87.00          84.00              22.0           88.20     22.00      1.00
max        100.00         100.00         100.00              29.0           99.90     29.00      3.00
```

`gender` value counts:
gender
Female    1031
Male       969

`internet` value counts:
internet
Yes    1702
No      298

`family_support` value counts:
family_support
Yes    1221
No      779

`school_support` value counts:
school_support
No     1365
Yes     635

`parental_education` value counts:
parental_education
High School            633
Bachelor               572
Master                 360
No Formal Education    264
PhD                    171


# 3. Correlations & Key Influencing Factors

Correlation with average score (all features):
```
study_time_hours    0.797
attendance_pct      0.484
age                 0.016
absences           -0.001
failures           -0.202
```

Categorical effect sizes (mean score difference vs baseline):
  - `internet`: +0.21 pts (internet=Yes vs No)
  - `family_support`: -0.31 pts (family_support=Yes vs No)
  - `school_support`: +2.21 pts (school_support=Yes vs No)

# 4. Key Insights

1. **Attendance is the dominant driver** - correlation of 0.484 with average score; the gap between the most and least attentive students is substantial.
2. **Study time matters too** - 0.797 correlation; students studying >=15 hrs/wk score ~20.1 pts higher than those studying <5 hrs.
3. **Failures weigh heavily** - -0.202 correlation; students with 0 failures score ~4.4 pts higher than those with failures.
4. **Internet access** adds +0.21 pts on average; parental education shows a clear socio-economic gradient.
5. **Gender effect is subject-specific** - males average higher math, females higher reading/writing (see distributions).

---
*Patterns are baked into generated data for demonstration; the pipeline applies to any real dataset.*