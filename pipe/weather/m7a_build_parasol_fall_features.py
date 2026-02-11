# pipe/weather/m7a_build_parasol_fall_features.py

import pandas as pd
import numpy as np
from pathlib import Path

print("🚀 M7a – building fall Parasol feature table")

BASE = Path("/Volumes/Data/iNaturalist/weather")

IN_PATH  = BASE / "derived/merged_grid_weather_counts_monthly.parquet"
OUT_PATH = BASE / "derived/m7_features_parasol_fall_from_spring_summer.parquet"

# --------------------------------------------------
# Load
# --------------------------------------------------
df = pd.read_parquet(IN_PATH)

print(f"📥 Loaded rows: {len(df):,}")
print(f"Years: {sorted(df.year.unique())}")

# --------------------------------------------------
# 1️⃣ Target: Sep–Dec aggregation
# --------------------------------------------------
fall = (
    df
    .query("month >= 9")
    .groupby(["grid_id", "year"], as_index=False)
    .agg(
        y_n_parasol=("n_parasol", "sum"),
        y_n_total=("n_total", "sum"),
    )
)

print(f"🍂 Fall rows: {len(fall):,}")

# --------------------------------------------------
# 2️⃣ Features: Jan–Aug weather (monthly, wide)
# --------------------------------------------------
spring_summer = (
    df
    .query("month <= 8")
    .pivot_table(
        index=["grid_id", "year"],
        columns="month",
        values=["temp", "precip"]
    )
)

# flatten column names: temp_m01, precip_m08, …
spring_summer.columns = [
    f"{var}_m{int(m):02d}" for var, m in spring_summer.columns
]

spring_summer = spring_summer.reset_index()

print(f"🌱 Spring–summer feature rows: {len(spring_summer):,}")

# --------------------------------------------------
# 3️⃣ Merge target + features
# --------------------------------------------------
final = (
    fall
    .merge(spring_summer, on=["grid_id", "year"], how="inner")
)

# Optional safety filter: remove empty seasons
final = final.query("y_n_total > 0")

print(f"✅ Final rows: {len(final):,}")

# --------------------------------------------------
# 4️⃣ Write parquet
# --------------------------------------------------
final.to_parquet(OUT_PATH)

print("📦 Written:", OUT_PATH)
print("🏁 M7a complete")