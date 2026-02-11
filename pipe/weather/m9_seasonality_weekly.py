# pipe/weather/m9_seasonality_weekly.py

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path("/Volumes/Data/iNaturalist/weather")
DATA_PATH = BASE / "derived/merged_grid_weather_counts_monthly.parquet"
OUT_DIR = BASE / "models/m9_seasonality_weekly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 START M9 – weekly seasonality")

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_parquet(DATA_PATH)

# If only month exists → approximate week
if "week" not in df.columns:
    print("⚠️ No week column found – approximating from month")
    df["week"] = (df["month"] * 4).clip(1, 52)

df = (
    df
    .groupby(["year", "week"], as_index=False)
    .agg(
        n_parasol=("n_parasol", "sum"),
        n_total=("n_total", "sum")
    )
    .query("n_total > 0")
)

print(f"✔ Rows: {len(df):,}")
print(f"Weeks: {sorted(df.week.unique())}")

# --------------------------------------------------
# Build cyclic design matrix
# --------------------------------------------------
# Use Fourier basis (clean + robust)
def fourier_terms(w, K=3):
    terms = {}
    for k in range(1, K + 1):
        terms[f"sin{k}"] = np.sin(2 * np.pi * k * w / 52)
        terms[f"cos{k}"] = np.cos(2 * np.pi * k * w / 52)
    return pd.DataFrame(terms)

X = fourier_terms(df.week, K=3)
X = sm.add_constant(X)

y = df.n_parasol
offset = np.log(df.n_total)

# --------------------------------------------------
# Fit GLM
# --------------------------------------------------
model = sm.GLM(
    y,
    X,
    family=sm.families.Poisson(),
    offset=offset
)

res = model.fit()
print(res.summary())

# --------------------------------------------------
# Predict smooth curve
# --------------------------------------------------
weeks = np.arange(1, 53)
Xp = sm.add_constant(fourier_terms(weeks, K=3))
log_mu = res.predict(Xp)
mu = np.exp(log_mu)

# Normalize (peak = 1)
mu_norm = mu / mu.max()

curve = pd.DataFrame({
    "week": weeks,
    "relative_rate": mu_norm
})

curve.to_parquet(OUT_DIR / "seasonality_weekly.parquet")

# --------------------------------------------------
# JSON export
# --------------------------------------------------
json_out = {
    "species": "Macrolepiota procera",
    "unit": "week",
    "peak_week": int(curve.loc[curve.relative_rate.idxmax(), "week"]),
    "curve": [
        {"week": int(w), "value": float(v)}
        for w, v in zip(curve.week, curve.relative_rate)
    ]
}

(Path(OUT_DIR) / "seasonality_weekly.json").write_text(
    json.dumps(json_out, indent=2),
    encoding="utf-8"
)

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(curve.week, curve.relative_rate, lw=2)
plt.xlabel("Kalenderwoche")
plt.ylabel("Relative Auftretenswahrscheinlichkeit")
plt.title("Saisonkurve – Macrolepiota procera")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "seasonality_weekly.png", dpi=150)
plt.close()

print("✅ M9 complete")
print("📁 Outputs in:", OUT_DIR)