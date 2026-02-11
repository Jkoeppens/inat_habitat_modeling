# pipe/weather/m6_xgb_parasol_monthly.py

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = Path(
    "/Volumes/Data/iNaturalist/weather/derived/"
    "merged_grid_weather_counts_monthly.parquet"
)

OUT_DIR = Path(
    "/Volumes/Data/iNaturalist/weather/models/m6_xgb_parasol_monthly"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TOTAL = 10

print("🚀 START M6 – XGBoost Parasol (monthly)")
print("📥 Loading data")

# --------------------------------------------------
# Load + basic filter
# --------------------------------------------------
df = pd.read_parquet(DATA_PATH)

df = df.query("n_total >= @MIN_TOTAL").copy()

print(f"✔ Rows: {len(df):,}")
print(f"✔ Years: {sorted(df.year.unique())}")

# --------------------------------------------------
# Target + weights (Option A)
# --------------------------------------------------
df["y"] = df.n_parasol / df.n_total
weights = df.n_total.values

# --------------------------------------------------
# Features
# --------------------------------------------------
X_num = df[["temp", "precip"]]

# month as categorical (very important!)
enc = OneHotEncoder(sparse_output=False, drop="first")
X_month = enc.fit_transform(df[["month"]])

X = np.hstack([X_num.values, X_month])

feature_names = (
    ["temp", "precip"]
    + enc.get_feature_names_out(["month"]).tolist()
)

# --------------------------------------------------
# Model config
# --------------------------------------------------
model = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

# --------------------------------------------------
# LOYO evaluation
# --------------------------------------------------
print("🔁 LOYO evaluation")

rows = []

for test_year in sorted(df.year.unique()):
    train_idx = df.year != test_year
    test_idx  = df.year == test_year

    model.fit(
        X[train_idx],
        df.loc[train_idx, "y"],
        sample_weight=weights[train_idx],
    )

    pred = model.predict(X[test_idx])
    obs  = df.loc[test_idx, "y"]

    r2 = r2_score(obs, pred)

    rows.append({
        "test_year": test_year,
        "r2_rate": r2,
        "n_obs": test_idx.sum(),
    })

loyo = pd.DataFrame(rows)
loyo.to_parquet(OUT_DIR / "loyo.parquet")

print(loyo)
print(f"✔ Mean R²: {loyo.r2_rate.mean():.3f}")

# --------------------------------------------------
# Final model (all data)
# --------------------------------------------------
print("📐 Fitting final model")

model.fit(X, df["y"], sample_weight=weights)

# --------------------------------------------------
# Feature importance
# --------------------------------------------------
imp = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

imp.to_parquet(OUT_DIR / "feature_importance.parquet")

print("=== TOP FEATURES ===")
print(imp.head(15))

print("✅ M6 complete")
print("📁 Outputs in:", OUT_DIR)