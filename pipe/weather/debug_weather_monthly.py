"""
DEBUG – Monthly weather grid sanity checks

Checks:
1. Shape & columns
2. Rows per year
3. Missing values
4. Physical plausibility (temp / precip)
5. Seasonal signal (monthly means)
6. Spatial variance
7. Join with iNat grid counts
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path("/Volumes/Data/iNaturalist/weather")

WEATHER_PATH = BASE / "derived/weather_grid_year_monthly.parquet"
COUNTS_PATH  = BASE / "inat/processed/grid_counts.parquet"
GRID_PATH    = BASE / "grid/grid_20km_DE.gpkg"

# --------------------------------------------------
# Load
# --------------------------------------------------
print("📥 Loading weather data")
weather = pd.read_parquet(WEATHER_PATH)

print("📥 Loading counts")
counts = pd.read_parquet(COUNTS_PATH)

# --------------------------------------------------
# 1) Structure
# --------------------------------------------------
print("\n=== STRUCTURE ===")
print("Shape:", weather.shape)
print("Columns:")
print(weather.columns.tolist())

# expected columns
temp_cols   = [f"temp_m{m:02d}" for m in range(1, 13)]
precip_cols = [f"precip_m{m:02d}" for m in range(1, 13)]

missing_cols = [c for c in temp_cols + precip_cols if c not in weather.columns]
if missing_cols:
    print("❌ Missing expected columns:", missing_cols)
else:
    print("✔ All monthly columns present")

# --------------------------------------------------
# 2) Rows per year
# --------------------------------------------------
print("\n=== ROWS PER YEAR ===")
rows_per_year = weather.groupby("year").size()
print(rows_per_year)

# --------------------------------------------------
# 3) Missing values
# --------------------------------------------------
print("\n=== MISSING VALUES (fraction) ===")
na_frac = weather.isna().mean().sort_values(ascending=False)
print(na_frac.head(15))

# --------------------------------------------------
# 4) Physical plausibility
# --------------------------------------------------
print("\n=== TEMPERATURE STATS (°C) ===")
print(weather[temp_cols].describe())

print("\n=== PRECIPITATION STATS (mm) ===")
print(weather[precip_cols].describe())

# extreme checks
print("\n=== EXTREME VALUES CHECK ===")
print("Temp < -30°C:", (weather[temp_cols] < -30).sum().sum())
print("Temp > 40°C:",  (weather[temp_cols] > 40).sum().sum())
print("Precip < 0:",   (weather[precip_cols] < 0).sum().sum())

# --------------------------------------------------
# 5) Seasonal signal (mean over all grids/years)
# --------------------------------------------------
print("\n=== SEASONAL SIGNAL ===")
monthly_temp_mean = weather[temp_cols].mean()
monthly_prec_mean = weather[precip_cols].mean()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

monthly_temp_mean.plot(marker="o", ax=ax[0])
ax[0].set_title("Mean monthly temperature (°C)")
ax[0].set_xlabel("Month")
ax[0].set_ylabel("°C")

monthly_prec_mean.plot(marker="o", ax=ax[1])
ax[1].set_title("Mean monthly precipitation (mm)")
ax[1].set_xlabel("Month")
ax[1].set_ylabel("mm")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6) Spatial variance (example month)
# --------------------------------------------------
print("\n=== SPATIAL VARIANCE ===")
for m in [1, 7]:
    col = f"temp_m{m:02d}"
    print(f"{col} std per year:")
    print(weather.groupby("year")[col].std())

# --------------------------------------------------
# 7) Join sanity with iNat counts
# --------------------------------------------------
print("\n=== JOIN WITH COUNTS ===")
df = counts.merge(weather, on=["grid_id", "year"], how="inner")
print("Joined shape:", df.shape)

print("Rows per year after join:")
print(df.groupby("year").size())

print("\n=== BASIC TARGET STATS ===")
print(df.describe())

print("\n✅ DEBUG COMPLETE")