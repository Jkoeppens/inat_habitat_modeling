# --------------------------------------------------
# M7b – GLM: Fall Parasol from Spring–Summer Weather
# --------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

print("🚀 START M7b – GLM fall parasol")

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = Path(
    "/Volumes/Data/iNaturalist/weather/derived/"
    "m7_features_parasol_fall_from_spring_summer.parquet"
)

OUT_DIR = Path(
    "/Volumes/Data/iNaturalist/weather/models/m7b_glm_fall"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Config
# --------------------------------------------------
MIN_TOTAL = 10   # minimum fall observations per grid×year

TEMP_COLS = [f"temp_m{m:02d}" for m in range(1, 9)]
PREC_COLS = [f"precip_m{m:02d}" for m in range(1, 9)]
X_COLS = TEMP_COLS + PREC_COLS

# --------------------------------------------------
# Load data
# --------------------------------------------------
print("📥 Loading data")

df = pd.read_parquet(DATA_PATH)

df = (
    df
    .query("y_n_total >= @MIN_TOTAL")
    .assign(offset=lambda d: np.log(d.y_n_total))
)

print(f"✔ Rows after filter: {len(df):,}")
print(f"✔ Grids: {df.grid_id.nunique()}")
print(f"✔ Years: {sorted(df.year.unique())}")

# --------------------------------------------------
# Handle missing values
# --------------------------------------------------
df[X_COLS] = df[X_COLS].fillna(df[X_COLS].median())

# Optional but recommended: standardize predictors
X_std = (df[X_COLS] - df[X_COLS].mean()) / df[X_COLS].std()

# --------------------------------------------------
# Fit GLM
# --------------------------------------------------
print("📐 Fitting Negative Binomial GLM")

X = sm.add_constant(X_std)
y = df["y_n_parasol"]
offset = df["offset"]

model = sm.GLM(
    y,
    X,
    family=sm.families.NegativeBinomial(),
    offset=offset
)

res = model.fit()

print(res.summary())

# --------------------------------------------------
# Save outputs
# --------------------------------------------------
coef = (
    res.params
    .rename("beta")
    .to_frame()
)

coef["RR"] = np.exp(coef.beta)  # rate ratios
coef["variable"] = coef.index

coef.to_parquet(OUT_DIR / "coefficients.parquet")

summary = {
    "n_rows": int(len(df)),
    "n_grids": int(df.grid_id.nunique()),
    "year_min": int(df.year.min()),
    "year_max": int(df.year.max()),
    "llf": float(res.llf),
    "aic": float(res.aic),
}

pd.Series(summary).to_frame("value").to_parquet(
    OUT_DIR / "summary.parquet"
)

print("✅ M7b complete")
print("📁 Outputs written to:", OUT_DIR)