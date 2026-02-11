#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M8 – XGBoost Regression
Ziel:
  Herbst-Parasol-Rate (Sep–Dez)
  aus Frühling/Sommer-Wetter (Jan–Aug)

Output:
  - xgb_model.json
  - feature_importance.parquet
  - feature_importance.json
  - loyo.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import r2_score

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path("/Volumes/Data/iNaturalist/weather")

DATA_PATH = BASE / "derived/m7_features_parasol_fall_from_spring_summer.parquet"
OUT_DIR   = BASE / "models/m8_xgb_parasol_fall"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Config
# --------------------------------------------------
MIN_TOTAL = 10
RANDOM_STATE = 42

# --------------------------------------------------
# Load data
# --------------------------------------------------
print("🚀 START M8 – XGBoost Parasol (fall)")

df = pd.read_parquet(DATA_PATH)

df = (
    df
    .query("y_n_total >= @MIN_TOTAL")
    .assign(
        y_rate=lambda d: d.y_n_parasol / d.y_n_total
    )
)

print(f"✔ Rows: {len(df):,}")
print(f"✔ Years: {sorted(df.year.unique())}")

# --------------------------------------------------
# Features / target
# --------------------------------------------------
FEATURES = [
    c for c in df.columns
    if c.startswith("temp_m") or c.startswith("precip_m")
]

X = df[FEATURES]
y = df["y_rate"]

# --------------------------------------------------
# Model
# --------------------------------------------------
model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=4,
)

# --------------------------------------------------
# LOYO evaluation
# --------------------------------------------------
print("🔁 LOYO evaluation")

rows = []

for year in sorted(df.year.unique()):
    train = df[df.year != year]
    test  = df[df.year == year]

    model.fit(train[FEATURES], train["y_rate"])

    pred = model.predict(test[FEATURES])
    r2   = r2_score(test["y_rate"], pred)

    rows.append({
        "test_year": int(year),
        "r2_rate": r2,
        "n_obs": len(test),
    })

loyo = pd.DataFrame(rows)
loyo.to_parquet(OUT_DIR / "loyo.parquet")

print(loyo)
print(f"✔ Mean R²: {loyo.r2_rate.mean():.3f}")

# --------------------------------------------------
# Fit final model
# --------------------------------------------------
print("📐 Fitting final model")

model.fit(X, y)

# --------------------------------------------------
# Save model
# --------------------------------------------------
MODEL_PATH = OUT_DIR / "xgb_model.json"
model.save_model(MODEL_PATH)
print(f"✔ Model written: {MODEL_PATH}")

# --------------------------------------------------
# Feature importance
# --------------------------------------------------
booster = model.get_booster()
scores = booster.get_score(importance_type="gain")

fi = (
    pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in scores.items()
    ])
    .sort_values("importance", ascending=False)
)

fi.to_parquet(OUT_DIR / "feature_importance.parquet")
fi.to_json(
    OUT_DIR / "feature_importance.json",
    orient="records",
    indent=2
)

print("=== TOP FEATURES ===")
print(fi.head(12))

print("✅ M8 complete")
print("📁 Outputs in:", OUT_DIR)