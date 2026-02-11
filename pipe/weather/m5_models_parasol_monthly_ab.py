# pipe/weather/m5_models_parasol_monthly_ab.py

"""
M5 – Monthly Parasol models (A vs B) on merged grid×year×month table

Input:
    /Volumes/Data/iNaturalist/weather/derived/merged_grid_weather_counts_monthly.parquet

Models:
    A (baseline): n_parasol ~ C(month) + offset(log(n_total))
    B (weather)  : n_parasol ~ C(month) + temp + precip + offset(log(n_total))

Validation:
    LOYO (leave-one-year-out) for both models

Outputs:
    /Volumes/Data/iNaturalist/weather/models/m5_parasol_monthly_ab/
        - summary.parquet
        - loyo.parquet
        - coefficients_A.parquet
        - coefficients_B.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ----------------------------
# Paths
# ----------------------------
BASE = Path("/Volumes/Data/iNaturalist/weather")
DATA_PATH = BASE / "derived" / "merged_grid_weather_counts_monthly.parquet"
OUT_DIR = BASE / "models" / "m5_parasol_monthly_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Config
# ----------------------------
MIN_TOTAL = 10          # drop very low-effort cells
USE_CONTROLS_AS_OFFSET = False  # keep False unless you explicitly want fungi-controls logic

# ----------------------------
# Load
# ----------------------------
print("🚀 START M5 – Parasol monthly models A vs B")
print("📥 Loading merged data")
df = pd.read_parquet(DATA_PATH)

required = {"grid_id", "year", "month", "n_parasol", "n_total", "temp", "precip"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in merged dataset: {missing}")

print(f"   rows: {len(df):,}")
print(f"   years: {sorted(df.year.unique().tolist())}")
print(f"   months: {sorted(df.month.unique().tolist())}")

# ----------------------------
# Clean / filter
# ----------------------------
df = df.copy()

# Ensure types
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)

# Guardrails: avoid log(0)
df = df[df["n_total"].fillna(0).astype(int) >= MIN_TOTAL].copy()

# Offset (exposure)
# NOTE: "n_total" is the exposure. If you ever want controls, swap here.
offset_base = df["n_total"].astype(float)

# Replace impossible values
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
df["precip"] = pd.to_numeric(df["precip"], errors="coerce")

# Handle missing weather: median imputation (simple, deterministic)
df["temp"] = df["temp"].fillna(df["temp"].median())
df["precip"] = df["precip"].fillna(df["precip"].median())

df["offset"] = np.log(np.maximum(offset_base, 1.0))

print(f"✔ Rows after MIN_TOTAL filter: {len(df):,}")
print("   n_parasol total:", int(df["n_parasol"].sum()))
print("   n_total total  :", int(df["n_total"].sum()))

# ----------------------------
# Model formulas
# ----------------------------
# Baseline: month-only (seasonality)
FORMULA_A = "n_parasol ~ C(month)"
# Weather: month + temp + precip
FORMULA_B = "n_parasol ~ C(month) + temp + precip"

# ----------------------------
# Fit helper
# ----------------------------
def fit_nb_glm(formula: str, data: pd.DataFrame):
    # Using statsmodels GLM NegativeBinomial (alpha fixed at 1 by default)
    # Good enough for comparative A vs B + LOYO.
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.NegativeBinomial(),
        offset=data["offset"]
    )
    return model.fit()

def predict_mean(res, data: pd.DataFrame):
    # returns predicted mean count (mu) on original scale
    return res.predict(data, offset=data["offset"])

def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and (np.std(a) > 0) and (np.std(b) > 0):
        return float(np.corrcoef(a, b)[0, 1])
    return np.nan

# ----------------------------
# Fit full models
# ----------------------------
print("\n📐 Fitting full Model A (month-only)")
resA = fit_nb_glm(FORMULA_A, df)
print("   done. llf:", float(resA.llf), "aic:", float(resA.aic))

print("\n📐 Fitting full Model B (month + weather)")
resB = fit_nb_glm(FORMULA_B, df)
print("   done. llf:", float(resB.llf), "aic:", float(resB.aic))

# ΔLL as likelihood-ratio style signal (bigger is better)
# (Not perfect since NB alpha fixed, but still a useful diagnostic.)
delta_ll = float(resB.llf - resA.llf)

# ----------------------------
# LOYO
# ----------------------------
print("\n🔁 LOYO (leave-one-year-out) for A and B")

loyo_rows = []
years = sorted(df["year"].unique())

for y in years:
    train = df[df["year"] != y]
    test = df[df["year"] == y]

    rA = fit_nb_glm(FORMULA_A, train)
    rB = fit_nb_glm(FORMULA_B, train)

    predA = predict_mean(rA, test)
    predB = predict_mean(rB, test)

    # Observed rate (interpretable)
    obs_rate = test["n_parasol"] / np.maximum(test["n_total"], 1)

    # Predicted rate (convert predicted counts to rate by dividing by exposure)
    pred_rate_A = predA / np.maximum(test["n_total"], 1)
    pred_rate_B = predB / np.maximum(test["n_total"], 1)

    loyo_rows.append({
        "test_year": y,
        "n_test": int(len(test)),
        "llf_A_train": float(rA.llf),
        "llf_B_train": float(rB.llf),
        "delta_ll_train_B_minus_A": float(rB.llf - rA.llf),
        "corr_pred_rate_A_vs_obs_rate": safe_corr(pred_rate_A, obs_rate),
        "corr_pred_rate_B_vs_obs_rate": safe_corr(pred_rate_B, obs_rate),
        "mean_obs_rate": float(obs_rate.mean()),
        "mean_pred_rate_A": float(np.mean(pred_rate_A)),
        "mean_pred_rate_B": float(np.mean(pred_rate_B)),
    })

loyo = pd.DataFrame(loyo_rows)
loyo.to_parquet(OUT_DIR / "loyo.parquet", index=False)
print("✔ LOYO written:", OUT_DIR / "loyo.parquet")

# ----------------------------
# Save coefficients
# ----------------------------
coefsA = (
    pd.DataFrame({
        "term": resA.params.index,
        "beta": resA.params.values,
        "se": resA.bse.values,
        "z": resA.tvalues.values,
        "p": resA.pvalues.values,
        "RR": np.exp(resA.params.values),
    })
)
coefsB = (
    pd.DataFrame({
        "term": resB.params.index,
        "beta": resB.params.values,
        "se": resB.bse.values,
        "z": resB.tvalues.values,
        "p": resB.pvalues.values,
        "RR": np.exp(resB.params.values),
    })
)

coefsA.to_parquet(OUT_DIR / "coefficients_A.parquet", index=False)
coefsB.to_parquet(OUT_DIR / "coefficients_B.parquet", index=False)
print("✔ coefficients written")

# ----------------------------
# Summary
# ----------------------------
summary = pd.DataFrame([{
    "n_rows": int(len(df)),
    "n_grids": int(df["grid_id"].nunique()),
    "years": ",".join(map(str, years)),
    "min_total_filter": int(MIN_TOTAL),

    "llf_A": float(resA.llf),
    "aic_A": float(resA.aic),

    "llf_B": float(resB.llf),
    "aic_B": float(resB.aic),

    "delta_ll_B_minus_A": delta_ll,
    "delta_aic_A_minus_B": float(resA.aic - resB.aic),

    "loyo_mean_corr_A": float(loyo["corr_pred_rate_A_vs_obs_rate"].mean()),
    "loyo_mean_corr_B": float(loyo["corr_pred_rate_B_vs_obs_rate"].mean()),
    "loyo_mean_delta_ll_train": float(loyo["delta_ll_train_B_minus_A"].mean()),
}])

summary.to_parquet(OUT_DIR / "summary.parquet", index=False)
print("✔ summary written:", OUT_DIR / "summary.parquet")

# Optional: print a compact readout
print("\n=== QUICK READOUT ===")
print(summary.T)

print("\n✅ M5 complete")
print("📁 Outputs in:", OUT_DIR)