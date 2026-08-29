"""Generate a synthetic student-performance dataset.

Mirrors the structure of well-known UCI student performance data but adds
realistic baked-in relationships so EDA can uncover meaningful patterns:

- attendance % and study time strongly drive exam scores
- failures hurt scores
- gender has a small subject-level effect (females higher reading/writing)
- internet access gives a small boost
- school support reduces the failure penalty
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(11)

n = 2000

gender = rng.choice(["Female", "Male"], n, p=[0.5, 0.5])
age = rng.integers(15, 20, n).astype(int)
internet = rng.choice(["Yes", "No"], n, p=[0.85, 0.15])
family_support = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
parental_education = rng.choice(
    ["No Formal Education", "High School", "Bachelor", "Master", "PhD"], n,
    p=[0.12, 0.32, 0.30, 0.18, 0.08],
)
school_support = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
study_time = rng.integers(0, 30, n).astype(float)          # hours / week
attendance = rng.uniform(55, 100, n)                        # %
absences = rng.integers(0, 30, n).astype(float)
failures = rng.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.10, 0.05]).astype(float)

internet_boost = np.where(internet == "Yes", 1.5, 0.0)
support_boost = np.where(school_support == "Yes", 2.0, 0.0)
fail_penalty = np.where(school_support == "Yes", 2.0, 3.0) * failures

base_math = 30 + 0.42 * attendance + 1.05 * study_time - fail_penalty
base_read = 32 + 0.40 * attendance + 0.95 * study_time - fail_penalty
base_write = 28 + 0.40 * attendance + 1.00 * study_time - fail_penalty

math_score = base_math + internet_boost + support_boost + np.where(gender == "Male", 2.0, 0.0)
read_score = base_read + internet_boost + support_boost + np.where(gender == "Female", 2.5, 0.0)
write_score = base_write + internet_boost + support_boost + np.where(gender == "Female", 2.5, 0.0)

math_score += rng.normal(0, 5.5, n)
read_score += rng.normal(0, 5.5, n)
write_score += rng.normal(0, 5.5, n)

def clip(x):
    return np.clip(np.round(x), 0, 100).astype(int)

df = pd.DataFrame(
    {
        "gender": gender,
        "age": age,
        "internet": internet,
        "family_support": family_support,
        "school_support": school_support,
        "parental_education": parental_education,
        "study_time_hours": study_time.round(1),
        "attendance_pct": attendance.round(1),
        "absences": absences.astype(int),
        "failures": failures.astype(int),
        "math_score": clip(math_score),
        "reading_score": clip(read_score),
        "writing_score": clip(write_score),
    }
)

df = df.sample(frac=1, random_state=5).reset_index(drop=True)

out = ROOT / "data" / "student_performance.csv"
df.to_csv(out, index=False)
print(f"Generated {len(df)} students, shape {df.shape}")
print("Score means:\n", df[["math_score", "reading_score", "writing_score"]].mean().round(2).to_string())