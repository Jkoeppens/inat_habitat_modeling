#!/usr/bin/env python3
"""
analyze_glm.py

Führt eine logistische Regression (GLM) auf klimatologischen Features durch,
um Ziel- vs. Kontrastarten zu unterscheiden. Pfade und Spezies werden aus
default.yaml / local.yaml gelesen (bootstrap).
"""

import sys
from pathlib import Path

# Projektpfad hinzufügen (damit bootstrap gefunden wird)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import statsmodels.api as sm
from bootstrap import init as bootstrap_init

# --------------------------------------------------
# Init config
# --------------------------------------------------
cfg = bootstrap_init(verbose=True)

target_key = cfg["defaults"]["target_species"]
contrast_key = cfg["defaults"]["contrast_species"]

target_name = target_key
contrast_name = contrast_key if contrast_key != "all" else "background"

features_dir = Path(cfg["paths"]["features_dir"])
csv_path = features_dir / target_name / f"inat_with_climatology_{target_name}_vs_{contrast_name}.csv"

# --------------------------------------------------
# Load data
# --------------------------------------------------
print(f"\n📂 Lade Feature-CSV: {csv_path}")
df = pd.read_csv(csv_path)
y = df["label"]

# --------------------------------------------------
# Feature selection (pilzrelevant, reduziert)
# --------------------------------------------------
X_COLS = [
    "m08_ndvi_mean", "m09_ndvi_mean", "m10_ndvi_mean",
    "m08_ndwi_mean", "m09_ndwi_mean", "m10_ndwi_mean",
    "m08_ndvi_std",  "m09_ndvi_std",  "m10_ndvi_std",
]

X = df[X_COLS].copy()
X = X.fillna(X.median())  # Missing → Median
X = sm.add_constant(X)    # Intercept

# --------------------------------------------------
# Fit logistic GLM
# --------------------------------------------------
print("\n📊 Fitte logistische Regression…")
model = sm.GLM(y, X, family=sm.families.Binomial())
res = model.fit()

print("\n📄 Modell-Zusammenfassung:")
print(res.summary())