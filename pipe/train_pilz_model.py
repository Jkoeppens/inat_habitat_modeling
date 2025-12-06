#!/usr/bin/env python3
"""
train_pilz_model.py – generisches Trainingsskript
funktioniert für jede Spezies-Kombination aus cfg.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_score,
    recall_score, f1_score
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
def train_pilz_model(cfg=None):

    # ------------------------------------------------------
    # 0. Laden der cfg
    # ------------------------------------------------------
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

    print(f"🎯 Training für Spezies-Paar: {tname} vs {cname}")


    # ------------------------------------------------------
    # 1. Richtige Input-Datei bestimmen (robust)
    # ------------------------------------------------------
    features_root = Path(cfg["paths"]["features_dir"])

    # Mögliche Ordnernamen:
    #   - species-key: macrolepiota_procera
    #   - species-name: Macrolepiota_procera
    folder_key = features_root / target
    folder_name = features_root / tname

    candidate_folders = [folder_key, folder_name]

    print("🔎 Suche Feature-Ordner…")
    for f in candidate_folders:
        print("   → prüfe:", f)

    # Wähle existierenden Ordner
    features_dir = None
    for folder in candidate_folders:
        if folder.exists():
            features_dir = folder
            break

    if features_dir is None:
        raise FileNotFoundError(
            "❌ Kein Feature-Ordner gefunden!\n"
            f"Versucht:\n  {folder_key}\n  {folder_name}"
        )

    print(f"📁 Feature-Ordner gefunden: {features_dir}")

    # Dateinamen (nur einmal definiert)
    fname = f"inat_with_climatology_{tname}_vs_{cname}.csv"
    expected_file = features_dir / fname

    if not expected_file.exists():
        raise FileNotFoundError(
            "❌ Feature-Datei konnte nicht gefunden werden!\n"
            f"Gesucht:\n  {expected_file}"
        )

    input_csv = expected_file
    print(f"📄 Feature-Datei: {input_csv}")

    # ------------------------------------------------------
    # 2. Modell-Output-Pfade
    # ------------------------------------------------------
    model_out = (
        Path(cfg["paths"]["output_dir"]) 
        / target
        / f"model_{tname}_vs_{cname}.json"
    )
    fig_out = (
        Path(cfg["paths"]["output_dir"])
        / target
        / f"feature_importance_{tname}_vs_{cname}.png"
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)


    # ------------------------------------------------------
    # 3. Daten laden
    # ------------------------------------------------------
    df = pd.read_csv(input_csv)

    y = df["label"].astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())

    print(f"Pilz (1): {pos}  |  Kontrast (0): {neg}")
    scale = neg / pos
    print(f"⚖ scale_pos_weight={scale:.3f}")


    # ------------------------------------------------------
    # 4. Feature Columns
    # ------------------------------------------------------

    feature_cols = [
    c for c in df.columns
    if c.startswith("m") and "coverage" not in c
    ]
    X = df[feature_cols]
    print(f"🔢 {len(feature_cols)} Features (coverage entfernt)")

    print(f"🔢 {len(feature_cols)} Features")


    # ------------------------------------------------------
    # 5. Sample Weights
    # ------------------------------------------------------
    coverage_cols = [c for c in df.columns if "coverage" in c]

    if coverage_cols:
        sample_weights = df[coverage_cols].mean(axis=1).values
    else:
        sample_weights = np.ones(len(df))


    # ------------------------------------------------------
    # 6. Split
    # ------------------------------------------------------
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights,
        test_size=0.25,
        random_state=42,
        stratify=y
    )


    # ------------------------------------------------------
    # 7. Modell
    # ------------------------------------------------------
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.9,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="logloss"
    )

    print("🚀 Trainiere Modell…")
    model.fit(X_train, y_train, sample_weight=w_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"🔥 ROC-AUC = {auc:.3f}")

    thr = find_best_threshold(y_test, y_prob)
    print(f"🎯 Bester Threshold = {thr:.3f}")

    y_pred = (y_prob > thr).astype(int)
    print(confusion_matrix(y_test, y_pred))


    # ------------------------------------------------------
    # 8. Feature Importance Plot
    # ------------------------------------------------------
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:20]

    plt.figure(figsize=(8, 10))
    plt.barh([feature_cols[i] for i in idx], importance[idx])
    plt.gca().invert_yaxis()
    plt.title(f"Top 20 Features – {tname} vs {cname}")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=180)

    print(f"📊 Feature Importance gespeichert: {fig_out}")

    # ------------------------------------------------------
    # 9. Speichern
    # ------------------------------------------------------
    model.save_model(model_out)
    print(f"💾 Modell gespeichert: {model_out}")

    print("✅ Training abgeschlossen.")
    return model, feature_cols