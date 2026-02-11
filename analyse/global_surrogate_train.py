#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# Optional: Konfiguration laden
try:
    from bootstrap import init as bootstrap_init
except ImportError:
    bootstrap_init = None

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


def infer_semantics(feature_name: str):
    if not (feature_name.startswith("m") and "_" in feature_name):
        return {"label": feature_name, "palette": ["#dddddd"] * 3, "range": (0.0, 1.0)}
    try:
        month = int(feature_name[1:3])
        rest = feature_name[4:]
    except Exception:
        return {"label": feature_name, "palette": ["#dddddd"] * 3, "range": (0.0, 1.0)}

    if month in (7, 8, 9): season = "Sommer"
    elif month in (10, 11, 12): season = "Herbst"
    else: season = "Saison"

    label = feature_name
    palette = ["#dddddd"] * 3
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
    elif rest.startswith("geary_ndvi"):
        label = f"Vegetations-Heterogenität ({season}, Geary)"
        palette = GEARY_PALETTE
        rng = RANGES["geary"]
    elif rest.startswith("moran_ndwi"):
        label = f"Feuchtigkeits-Cluster ({season}, Moran)"
        palette = MORAN_PALETTE
        rng = RANGES["moran"]
    elif rest.startswith("geary_ndwi"):
        label = f"Feuchtigkeits-Heterogenität ({season}, Geary)"
        palette = GEARY_PALETTE
        rng = RANGES["geary"]

    return {"label": label, "palette": palette, "range": rng}


def train_surrogate(model_path: str, data_path: str, max_depth: int = 4):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    booster = model.get_booster()
    feature_names = booster.feature_names
    if feature_names is None:
        raise ValueError("❌ Modell enthält keine Feature-Namen im Booster!")

    df = pd.read_csv(data_path)
    X = df[feature_names].copy()
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    y_pred = model.predict_proba(X)[:, 1]
    surrogate = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=50, random_state=42
    )
    surrogate.fit(X, y_pred)
    return surrogate, feature_names


def tree_to_json(tree, feature_names):
    def recurse(node_id: int):
        if tree.feature[node_id] == -2:
            val = float(tree.value[node_id][0][0])
            return {"leaf": val, "suit": val}
        feat_idx = int(tree.feature[node_id])
        feat_name = feature_names[feat_idx]
        thr = float(tree.threshold[node_id])
        semantics = infer_semantics(feat_name)
        return {
            "feature": feat_name,
            "threshold": thr,
            "yes": recurse(tree.children_right[node_id]),
            "no": recurse(tree.children_left[node_id]),
            **semantics
        }
    return recurse(0)


def resolve_paths_from_cfg():
    if not bootstrap_init:
        raise RuntimeError("bootstrap.py konnte nicht importiert werden.")

    cfg = bootstrap_init(verbose=True)
    target_key = cfg["defaults"]["target_species"]
    contrast_key = cfg["defaults"].get("contrast_species", "background")

    target_name = cfg["species"][target_key]["name"].replace(" ", "_").lower()
    contrast_name = (
        cfg["species"].get(contrast_key, {"name": "background"})["name"].replace(" ", "_").lower()
    )

    model_path = Path(cfg["paths"]["output_dir"]) / target_name / f"model_MONTHLY_{target_name}_vs_{contrast_name}.json"
    data_path = Path(cfg["paths"]["features_dir"]) / target_name / f"inat_with_climatology_{target_name}_vs_{contrast_name}.csv"
    out_json = Path("surrogate_tree_" + target_name + "_vs_" + contrast_name + ".json")

    return model_path, data_path, out_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Pfad zum XGBoost-Modell")
    parser.add_argument("--data", help="Pfad zur Feature-CSV")
    parser.add_argument("--out-json", help="Ausgabe-JSON-Pfad")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--from-cfg", action="store_true", help="Pfadinfos aus default.yaml laden")
    args = parser.parse_args()

    if args.from_cfg:
        model_path, data_path, out_json = resolve_paths_from_cfg()
    else:
        if not (args.model and args.data and args.out_json):
            raise ValueError("❌ Bitte entweder --from-cfg oder alle --model/--data/--out-json angeben.")
        model_path = Path(args.model)
        data_path = Path(args.data)
        out_json = Path(args.out_json)

    surrogate, featnames = train_surrogate(model_path, data_path, args.depth)
    tree_dict = tree_to_json(surrogate.tree_, featnames)

    out_json.write_text(json.dumps(tree_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Surrogate-Tree gespeichert unter: {out_json}")

if __name__ == "__main__":
    main()