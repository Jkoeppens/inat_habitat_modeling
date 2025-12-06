#!/usr/bin/env python3
"""
train_pilz_model_monthly.py – reduzierte Modellvariante
Nur MEAN + AUTOCORR, keine STD, keine COUNT.

Ein "leichtes" Modell speziell für monatliche Predictions.
Erzeugt neue Output-Files, um Kollisionen zu vermeiden.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix
)
from pathlib import Path
from bootstrap import init as bootstrap_init


# ----------------------------------------------------------
# Optimaler Threshold
# ----------------------------------------------------------
def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thr, best_j = 0.5, -999
    for t in thresholds:
        pred = (y_prob > t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thr = t
    return best_thr


# ----------------------------------------------------------
# Hauptfunktion
# ----------------------------------------------------------
def train_pilz_model_monthly(cfg=None):

    # ------------------------------
    # 0. Config
    # ------------------------------
    if cfg is None:
        cfg = bootstrap_init(verbose=False)

    region = cfg["defaults"]["region"]
    base = Path(cfg["paths"]["base_data_dir"])

    target = cfg["defaults"]["target_species"]
    contrast = cfg["defaults"]["contrast_species"]

    tname = cfg["species"][target]["name"].replace(" ", "_")
    cname = (
        cfg["species"][contrast]["name"].replace(" ", "_")
        if contrast in cfg["species"]
        else "background"
    )

    print(f"🎯 MONTHLY-MODELL: {tname} vs {cname}")


    # ------------------------------
    # 1. Feature-Ordner finden
    # ------------------------------
    features_root = Path(cfg["paths"]["features_dir"])
    folder_key  = features_root / target
    folder_name = features_root / tname

    candidate_folders = [folder_key, folder_name]

    features_dir = None
    for f in candidate_folders:
        if f.exists():
            features_dir = f
            break

    if features_dir is None:
        raise FileNotFoundError("❌ Kein Feature-Ordner gefunden.")

    fname = f"inat_with_climatology_{tname}_vs_{cname}.csv"
    input_csv = features_dir / fname
    if not input_csv.exists():
        raise FileNotFoundError(
            f"❌ Feature-Datei fehlt: {input_csv}"
        )

    print(f"📄 Lade: {input_csv}")


    # ------------------------------
    # 2. OUTPUT-PFADE (NEU!)
    # ------------------------------
    out_root = Path(cfg["paths"]["output_dir"]) / target
    out_root.mkdir(parents=True, exist_ok=True)

    model_out = out_root / f"model_MONTHLY_{tname}_vs_{cname}.json"
    fig_out   = out_root / f"feature_importance_MONTHLY_{tname}_vs_{cname}.png"

    print(f"💾 Modell-Output: {model_out}")
    print(f"💾 Importance-Plot: {fig_out}")


    # ------------------------------
    # 3. Daten laden
    # ------------------------------
    df = pd.read_csv(input_csv)

    y = df["label"].astype(int)

    # ------------------------------
    # 4. FEATURE-WHITELIST
    # ------------------------------
    whitelist_keywords = [
        "ndvi_mean",
        "ndwi_mean",
        "moran",
        "geary",
    ]

    feature_cols = [
        c for c in df.columns
        if (
            c.startswith("m")
            and "coverage" not in c
            and any(kw in c.lower() for kw in whitelist_keywords)
        )
    ]

    print(f"🔍 WHITELIST Features:")
    for c in feature_cols:
        print("   •", c)

    X = df[feature_cols]


    # ------------------------------
    # 5. Sample Weights (wie vorher)
    # ------------------------------
    coverage_cols = [c for c in df.columns if "coverage" in c]
    sample_weights = df[coverage_cols].mean(axis=1).values if coverage_cols else np.ones(len(df))


    # ------------------------------
    # 6. Split
    # ------------------------------
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights,
        test_size=0.25,
        random_state=42,
        stratify=y
    )


    # ------------------------------
    # 7. Modell (kleiner)
    # ------------------------------
    print("🚀 Trainiere MONTHLY-Modell …")

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale = neg / pos

    model = xgb.XGBClassifier(
        n_estimators=250,      # kleiner
        max_depth=4,          # flacher
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="logloss"
    )

    model.fit(X_train, y_train, sample_weight=w_train)


    # ------------------------------
    # 8. Evaluation
    # ------------------------------
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"🔥 ROC-AUC = {auc:.3f}")

    thr = find_best_threshold(y_test, y_prob)
    print(f"🎯 Optimaler Threshold = {thr:.3f}")

    y_pred = (y_prob > thr).astype(int)
    print(confusion_matrix(y_test, y_pred))


    # ------------------------------
    # 9. Feature Importance
    # ------------------------------
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:20]

    plt.figure(figsize=(8, 10))
    plt.barh([feature_cols[i] for i in idx], importance[idx])
    plt.gca().invert_yaxis()
    plt.title(f"Top Features – MONTHLY {tname} vs {cname}")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=180)


    # ------------------------------
    # 10. Speichern
    # ------------------------------
    model.save_model(model_out)

    # NEU: vollständigen Booster-Dump sichern (für Rule Extraction)
    booster = model.get_booster()

    json_dump = model_out.with_suffix(".dump.json")
    text_dump = model_out.with_suffix(".dump.txt")

    booster.dump_model(str(json_dump), dump_format="json", with_stats=True)
    booster.dump_model(str(text_dump), dump_format="text", with_stats=True)

    print(f"💾 Booster-Dump JSON: {json_dump}")
    print(f"💾 Booster-Dump TEXT: {text_dump}")

    print(f"💾 Modell gespeichert: {model_out}")

    return model, feature_cols


if __name__ == "__main__":
    train_pilz_model_monthly()