#!/usr/bin/env python3
"""
Interaktive SHAP-Landschaft (mit Z-Score Normalisierung der Feature-Werte)

- X-Achse: SHAP-Wert
- Y-Achse: Feature
- Farbe: z-Score der ursprünglichen Feature-Werte
- Hintergrund: vollständig transparent
"""

import shap
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
from pathlib import Path
from bootstrap import init as bootstrap_init

# ---------------------------------------------------
# 1) Config & Pfade
# ---------------------------------------------------
cfg = bootstrap_init(verbose=False)

tkey = cfg["defaults"]["target_species"]
ckey = cfg["defaults"]["contrast_species"]

t_pretty = cfg["species"][tkey]["name"].replace(" ", "_")
c_pretty = cfg["species"][ckey]["name"].replace(" ", "_")

model_path = Path(cfg["paths"]["output_dir"]) / tkey / f"model_MONTHLY_{t_pretty}_vs_{c_pretty}.json"
feature_csv = Path(cfg["paths"]["features_dir"]) / t_pretty / f"inat_with_climatology_{t_pretty}_vs_{c_pretty}.csv"

print("Model:", model_path)
print("CSV  :", feature_csv)

# ---------------------------------------------------
# 2) Modell und Daten laden
# ---------------------------------------------------
model = xgb.XGBClassifier()
model.load_model(str(model_path))

df = pd.read_csv(feature_csv)

valid_stats = ["ndvi_mean","ndwi_mean","moran_ndvi","geary_ndvi","moran_ndwi","geary_ndwi"]
feature_cols = [c for c in df.columns if c.startswith("m") and any(s in c for s in valid_stats)]

# Sampling für Performance
n = min(4000, len(df))
X = df[feature_cols].sample(n, random_state=0).reset_index(drop=True)

# ---------------------------------------------------
# 3) SHAP Werte berechnen
# ---------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X)

# In DataFrame bringen
rows = []
for i, feat in enumerate(feature_cols):
    vals = X[feat].values
    shapv = shap_vals[:, i]

    # Z-Score Normalisierung
    v_mean = np.nanmean(vals)
    v_std  = np.nanstd(vals)
    if v_std < 1e-6:
        zvals = np.zeros_like(vals)
    else:
        zvals = (vals - v_mean) / v_std

    for v, z, s in zip(vals, zvals, shapv):
        rows.append({
            "feature": feat,
            "value": v,
            "value_z": z,
            "shap": s
        })

df_shap = pd.DataFrame(rows)

# ---------------------------------------------------
# 4) Interaktiver Plotly-Beeswarm mit z-Score Farbskala
# ---------------------------------------------------
fig = px.scatter(
    df_shap,
    x="shap",
    y="feature",
    color="value_z",                 # z-Score statt raw value
    color_continuous_scale="Turbo",  # bessere Farbdynamik
    opacity=0.5,
    render_mode="webgl",
)

fig.update_traces(marker=dict(size=4))

fig.update_layout(
    title="Interaktive SHAP-Landschaft (z-Score normalisiert)",
    xaxis_title="SHAP value",
    yaxis_title="Feature",
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

# ---------------------------------------------------
# 5) HTML speichern
# ---------------------------------------------------
output_file = "shap_landscape_interactive_zscore.html"
fig.write_html(output_file, include_plotlyjs="cdn", full_html=True)

print("HTML gespeichert:", output_file)

fig.show()