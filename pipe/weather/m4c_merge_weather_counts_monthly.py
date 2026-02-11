# pipe/weather/m4c_merge_weather_counts_monthly.py

import pandas as pd
from pathlib import Path

print("🔗 M4c – merge monthly weather + counts")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path("/Volumes/Data/iNaturalist/weather")

WEATHER_PATH = BASE / "derived/weather_grid_year_monthly.parquet"
COUNTS_PATH  = BASE / "inat/processed/grid_counts.parquet"
OUT_PATH     = BASE / "derived/merged_grid_weather_counts_monthly.parquet"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load
# --------------------------------------------------
print("📥 Loading data")

weather = pd.read_parquet(WEATHER_PATH)
counts  = pd.read_parquet(COUNTS_PATH)

print(f"   weather rows: {len(weather):,}")
print(f"   counts rows : {len(counts):,}")

# --------------------------------------------------
# Sanity checks
# --------------------------------------------------
required_weather_cols = {"grid_id", "year"}
required_counts_cols  = {"grid_id", "year", "month", "n_parasol", "n_total"}

assert required_weather_cols.issubset(weather.columns), "❌ weather schema mismatch"
assert required_counts_cols.issubset(counts.columns),  "❌ counts schema mismatch"

# --------------------------------------------------
# Weather: wide → long (monthly)
# --------------------------------------------------
print("🔄 Reshaping weather to monthly (long)")

temp_cols   = [c for c in weather.columns if c.startswith("temp_m")]
precip_cols = [c for c in weather.columns if c.startswith("precip_m")]

assert len(temp_cols) == 12, "❌ expected 12 temp columns"
assert len(precip_cols) == 12, "❌ expected 12 precip columns"

weather_long = (
    weather
    .melt(
        id_vars=["grid_id", "year"],
        value_vars=temp_cols + precip_cols,
        var_name="var",
        value_name="value"
    )
    .assign(
        month=lambda d: d["var"].str.extract(r"m(\d{2})").astype(int),
        kind=lambda d: d["var"].str.split("_").str[0]
    )
    .pivot_table(
        index=["grid_id", "year", "month"],
        columns="kind",
        values="value"
    )
    .reset_index()
)

print(f"   weather_long rows: {len(weather_long):,}")

# --------------------------------------------------
# Merge
# --------------------------------------------------
print("🔗 Merging")

df = (
    counts
    .merge(
        weather_long,
        on=["grid_id", "year", "month"],
        how="inner",
        validate="many_to_one"  # counts many × weather one
    )
)

print(f"✔ merged rows: {len(df):,}")

# --------------------------------------------------
# Final sanity
# --------------------------------------------------
print("\n=== MERGED STRUCTURE ===")
print(df.dtypes)

print("\nRows per year:")
print(df.groupby("year").size())

print("\nMissing values (fraction):")
print(df[["temp", "precip"]].isna().mean())

# --------------------------------------------------
# Save
# --------------------------------------------------
df.to_parquet(OUT_PATH)

print("\n✅ M4c complete")
print("📁 Written to:", OUT_PATH)