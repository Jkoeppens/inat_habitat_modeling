#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trainiert einen Global Surrogate Tree für ein XGBoost-Modell
und speichert den Baum als JSON.

Aufruf z.B.:

python analyse/global_surrogate_train.py \
  --model "/Volumes/Data/iNaturalist/outputs/macrolepiota_procera/model_MONTHLY_Macrolepiota_procera_vs_Parus_major.json" \
  --data  "/Volumes/Data/iNaturalist/features/Macrolepiota_procera/inat_with_climatology_Macrolepiota_procera_vs_Parus_major.csv" \
  --out-json surrogate_tree.json \
  --depth 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# -------------------------------
# Farbpaletten & Ranges
# -------------------------------

NDVI_PALETTE = ["#f2f2f2", "#a3c586", "#2f6b3a"]
NDWI_PALETTE = ["#f7fbff", "#6baed6", "#08519c"]
MORAN_PALETTE = ["#fee8c8", "#fdbb84", "#e34a33"]
GEARY_PALETTE = ["#f7f4f9", "#998ec3", "#542788"]

RANGES = {
    "ndvi_mean": (0.0, 1.0),
    "ndwi_mean": (-1.0, 1.0),
    "moran": (-0.2, 4.0),
    "geary": (0.0, 1.5),
}

# -------------------------------
# Feature-Semantik
# -------------------------------

def infer_semantics(feature_name: str):
    """
    Nimmt z.B.
      m12_ndvi_mean
      m08_geary_ndwi
      m10_moran_ndvi
    und liefert Label, Palette, Range.
    """
    if not (feature_name.startswith("m") and "_" in feature_name):
        return {
            "label": feature_name,
            "palette": ["#dddddd", "#aaaaaa", "#666666"],
            "range": (0.0, 1.0),
        }

    try:
        month = int(feature_name[1:3])
        rest = feature_name[4:]
    except Exception:
        return {
            "label": feature_name,
            "palette": ["#dddddd", "#aaaaaa", "#666666"],
            "range": (0.0, 1.0),
        }

    if month in (7, 8, 9):
        season = "Sommer"
    elif month in (10, 11, 12):
        season = "Herbst"
    else:
        season = "Saison"

    label = feature_name
    palette = ["#dddddd", "#aaaaaa", "#666666"]
    rng = (0.0, 1.0)

    if rest.startswith("ndvi_mean"):
        label = f"Vegetationsdichte ({season}, NDVI)"
        palette = NDVI_PALETTE
        rng = RANGES["ndvi_mean"]
    elif rest.startswith("ndwi_mean"):
        label = f"Feuchtigkeit ({season}, NDWI)"
        palette = NDWI_PALETTE
        rng = RANGES["ndwi_mean"]
    elif rest.startswith("moran_ndvi"):
        label = f"Vegetations-Cluster ({season}, Moran)"
        palette = MORAN_PALETTE
        rng = RANGES["moran"]
    elif rest.startswith("moran_ndwi"):
        label = f"Feuchtigkeits-Cluster ({season}, Moran)"
        palette = MORAN_PALETTE
        rng = RANGES["moran"]
    elif rest.startswith("geary_ndvi"):
        label = f"Vegetations-Heterogenität ({season}, Geary)"
        palette = GEARY_PALETTE
        rng = RANGES["geary"]
    elif rest.startswith("geary_ndwi"):
        label = f"Feuchtigkeits-Heterogenität ({season}, Geary)"
        palette = GEARY_PALETTE
        rng = RANGES["geary"]

    return {
        "label": label,
        "palette": palette,
        "range": rng,
    }

# -------------------------------
# 1) Surrogate trainieren
# -------------------------------

def train_surrogate(model_path: str, data_path: str, max_depth: int = 4):
    print("→ Lade XGBoost-Modell…")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    booster = model.get_booster()
    feature_names = booster.feature_names
    if feature_names is None:
        raise ValueError("❌ Modell enthält keine Feature-Namen im Booster!")

    print(f"→ Modell hat {len(feature_names)} Features")

    print("→ Lade Daten…")
    df = pd.read_csv(data_path)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"❌ CSV enthält nicht alle Modell-Features. Fehlend: {missing}")

    X = df[feature_names].copy()

    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = X[col].astype(float)
            except Exception:
                raise ValueError(f"❌ Spalte {col} ist object und lässt sich nicht in float casten.")

    print("→ Berechne Modell-Predictions (P(1))…")
    y_pred = model.predict_proba(X)[:, 1]

    print("→ Trainiere Surrogate DecisionTreeRegressor…")
    surrogate = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=50,
        random_state=42
    )
    surrogate.fit(X, y_pred)

    return surrogate, feature_names

# -------------------------------
# 2) Sklearn-Tree → JSON-Baum
# -------------------------------

def tree_to_json(tree, feature_names):
    """
    Konvertiert sklearn.tree_ in rekursives JSON:
    - Splits: {feature, threshold, yes, no, label, palette, range}
    - Leafs:  {leaf, suit}
    suit = mittlere Vorhersage im Leaf (≈ P(geeignet))
    """

    def recurse(node_id: int):
        # Leaf?
        if tree.feature[node_id] == -2:
            val = float(tree.value[node_id][0][0])
            return {
                "leaf": val,
                "suit": val
            }

        feat_idx = int(tree.feature[node_id])
        feat_name = feature_names[feat_idx]
        thr = float(tree.threshold[node_id])

        semantics = infer_semantics(feat_name)

        left_id = int(tree.children_left[node_id])
        right_id = int(tree.children_right[node_id])

        left = recurse(left_id)
        right = recurse(right_id)

        node = {
            "feature": feat_name,
            "threshold": thr,
            "yes": right,  # "Ja" = rechts (>= thr)
            "no": left,    # "Nein" = links  (< thr)
        }
        node.update(semantics)
        return node

    return recurse(0)

# -------------------------------
# MAIN
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Pfad zum XGBoost-JSON-Modell")
    parser.add_argument("--data", required=True, help="CSV mit Features")
    parser.add_argument("--out-json", default="surrogate_tree.json", help="JSON-Ausgabedatei")
    parser.add_argument("--depth", type=int, default=4, help="max_depth des Surrogate Trees")
    args = parser.parse_args()

    surrogate, featnames = train_surrogate(args.model, args.data, args.depth)
    tree_dict = tree_to_json(surrogate.tree_, featnames)

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(tree_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Surrogate-Tree-JSON gespeichert unter: {out_path}")

    print("=== DONE ===")

if __name__ == "__main__":
    main()