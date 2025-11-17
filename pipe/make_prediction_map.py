#!/usr/bin/env python3
"""
make_prediction_map.py

Erzeugt eine Habitat-Suitability-Karte für das aktuelle Spezies-Paar
aus cfg (defaults.target_species vs defaults.contrast_species).

Nutzt:
- XGBoost-Modell aus train_pilz_model.py
- Feature-Tabelle aus build_point_climatology_table.py
- Monatliche CLIMATOLOGY-Raster (region-aware)

Output:
- <output_dir>/<target_key>/suitability_map_<Target>_vs_<Contrast>.tif
- <output_dir>/<target_key>/suitability_map_<Target>_vs_<Contrast>.png
"""

import sys
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import xgboost as xgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------
# 0. Projektroot & bootstrap importieren
# ----------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # .../inat_habitat_modeling

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import init as bootstrap_init  # noqa: E402


# ----------------------------------------------------------
# Mapping der Bandnamen → Bandindex
# Muss zu build_point_climatology_table.infer_band_names passen!
# ----------------------------------------------------------
STAT_TO_BAND = {
    "ndvi_median": 1,
    "ndvi_mean": 2,
    "ndvi_std": 3,
    "ndvi_coverage": 4,
    "ndwi_median": 5,
    "ndwi_mean": 6,
    "ndwi_std": 7,
    "ndwi_coverage": 8,
    "moran_ndvi": 9,
    "geary_ndvi": 10,
    "moran_ndwi": 11,
    "geary_ndwi": 12,
}


def parse_feature_name(colname: str):
    """
    Erwartetes Format: 'm07_ndvi_mean' → (7, 'ndvi_mean')
    """
    assert colname.startswith("m"), f"Ungültiger Feature-Name: {colname}"
    month = int(colname[1:3])
    stat = colname[4:]
    return month, stat


def load_climatology_rasters(cfg: dict):
    """
    Lädt alle CLIMATOLOGY-TIFFs für die aktuelle Region.

    Sucht nacheinander:
    - <processed_dir>/<region>/CLIMATOLOGY_<region>_MONTH_XX.tif
    - <processed_dir>/<region>/CLIMATOLOGY_MONTH_XX.tif
    - <processed_dir>/CLIMATOLOGY_<region>_MONTH_XX.tif
    - <processed_dir>/CLIMATOLOGY_MONTH_XX.tif
    """
    region_key = cfg["defaults"]["region"]

    base_data_dir = Path(cfg["paths"]["base_data_dir"])
    processed_root = Path(cfg["paths"]["processed_dir"])

    candidates_root = [
        processed_root / region_key,
        processed_root,
    ]

    rasters = {}
    print("\n🌍 Suche CLIMATOLOGY-Raster…")

    for m in range(1, 13):
        found = None
        for root in candidates_root:
            p1 = root / f"CLIMATOLOGY_{region_key}_MONTH_{m:02d}.tif"
            p2 = root / f"CLIMATOLOGY_MONTH_{m:02d}.tif"

            if p1.exists():
                found = p1
                print(f"   ✔ Monat {m:02d}: {p1}")
                break
            if p2.exists():
                found = p2
                print(f"   ✔ Monat {m:02d}: {p2}")
                break

        if found is not None:
            rasters[m] = rasterio.open(found)
        else:
            print(f"   ⚠ Monat {m:02d}: kein TIFF gefunden")

    if not rasters:
        raise FileNotFoundError("❌ Keine CLIMATOLOGY-TIFFs gefunden!")

    return rasters


def find_model_and_feature_csv(cfg: dict):
    """
    Sucht:
    - Modell: model_<Target>_vs_<Contrast>.json
      unter <output_dir>/<target_key>/...
    - Feature-CSV: inat_with_climatology_<Target>_vs_<Contrast>.csv
      unter <features_dir>/<Target>/...
    """

    tkey = cfg["defaults"]["target_species"]         # z.B. macrolepiota_procera
    ckey = cfg["defaults"]["contrast_species"]

    t_pretty = cfg["species"][tkey]["name"].replace(" ", "_")  # Macrolepiota_procera
    c_pretty = cfg["species"][ckey]["name"].replace(" ", "_")  # Parus_major

    out_root = Path(cfg["paths"]["output_dir"])
    feat_root = Path(cfg["paths"]["features_dir"])

    # --- Modell ---
    model_candidates = [
        out_root / tkey / f"model_{t_pretty}_vs_{c_pretty}.json",     # outputs/macrolepiota_procera/...
        out_root / t_pretty / f"model_{t_pretty}_vs_{c_pretty}.json", # (Fallback)
    ]

    model_path = None
    for p in model_candidates:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(
            "❌ Kein Modell gefunden!\nProbiert wurden:\n" +
            "\n".join(f"  - {p}" for p in model_candidates)
        )

    # --- Feature-CSV ---
    features_dir_species_candidates = [
        feat_root / t_pretty,
        feat_root / tkey,
    ]

    feature_path = None
    for d in features_dir_species_candidates:
        candidate = d / f"inat_with_climatology_{t_pretty}_vs_{c_pretty}.csv"
        if candidate.exists():
            feature_path = candidate
            break

    if feature_path is None:
        raise FileNotFoundError(
            "❌ Keine Feature-CSV gefunden!\nProbiert wurden:\n" +
            "\n".join(
                f"  - {d / f'inat_with_climatology_{t_pretty}_vs_{c_pretty}.csv'}"
                for d in features_dir_species_candidates
            )
        )

    return model_path, feature_path, tkey, t_pretty, c_pretty


def build_prediction_map(cfg: dict):
    """
    Hauptfunktion: lädt Modell, Features, Klimaraster
    und schreibt eine Suitability-Karte als GeoTIFF + PNG.
    """

    region_key = cfg["defaults"]["region"]
    model_path, feature_csv, tkey, t_pretty, c_pretty = find_model_and_feature_csv(cfg)

    print("===============================================")
    print("🏞  Habitat-Suitability-Karten Generator")
    print("===============================================")
    print(f"🌱 Target:    {t_pretty}")
    print(f"🐦 Kontrast:  {c_pretty}")
    print(f"🌍 Region:    {region_key}")
    print("-----------------------------------------------")
    print(f"📄 Modell:    {model_path}")
    print(f"📄 Features:  {feature_csv}")

    # ------------------------------------------------------
    # 1. Modell laden
    # ------------------------------------------------------
    print("\n📂 Lade XGBoost-Modell…")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    # ------------------------------------------------------
    # 2. Feature-Liste aus CSV ableiten
    # ------------------------------------------------------
    print("📄 Lese Feature-Header…")
    df_head = pd.read_csv(feature_csv, nrows=1)
    feature_cols = [c for c in df_head.columns if re.match(r"m\d{2}_.+", c)]
    print(f"🔢 Anzahl Features: {len(feature_cols)}")
    print(f"   Beispiel: {feature_cols[:6]}")

    # ------------------------------------------------------
    # 3. Klimaraster laden
    # ------------------------------------------------------
    rasters = load_climatology_rasters(cfg)
    sample_raster = next(iter(rasters.values()))
    H, W = sample_raster.height, sample_raster.width
    print(f"\n🗺️ Rastergröße: {W} × {H}")

    profile = sample_raster.profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        compress="deflate",
        predictor=3,
    )

    out_root = Path(cfg["paths"]["output_dir"]) / tkey
    out_root.mkdir(parents=True, exist_ok=True)

    out_tif = out_root / f"suitability_map_{t_pretty}_vs_{c_pretty}.tif"
    out_png = out_root / f"suitability_map_{t_pretty}_vs_{c_pretty}.png"

    # ------------------------------------------------------
    # 4. Kachelweises Predicten
    # ------------------------------------------------------
    print(f"\n💾 Schreibe Suitability-Raster: {out_tif}")

    tile = 512

    with rasterio.open(out_tif, "w", **profile) as dst:
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                h = min(tile, H - y0)
                w = min(tile, W - x0)

                window = Window(x0, y0, w, h)

                # Feature-Matrix für diese Kachel
                X_tile = np.zeros((h * w, len(feature_cols)), dtype=np.float32)

                for idx, col in enumerate(feature_cols):
                    month, stat = parse_feature_name(col)

                    if stat not in STAT_TO_BAND:
                        raise KeyError(f"Unbekannte Statistik '{stat}' in Feature '{col}'")

                    band = STAT_TO_BAND[stat]

                    raster = rasters.get(month)
                    if raster is None:
                        raise ValueError(f"Kein CLIMATOLOGY-Raster für Monat {month}")

                    arr = raster.read(band, window=window)
                    X_tile[:, idx] = arr.reshape(-1)

                # Vorhersage
                preds = model.predict_proba(X_tile)[:, 1]
                pred_tile = preds.reshape(h, w).astype("float32")

                dst.write(pred_tile, 1, window=window)
                print(f"  → Block x={x0}:{x0+w}  y={y0}:{y0+h} fertig.")

    print("🎉 GeoTIFF fertig.")

    # ------------------------------------------------------
    # 5. PNG-Vorschau
    # ------------------------------------------------------
    print("🎨 Erzeuge PNG-Vorschau…")
    with rasterio.open(out_tif) as src:
        img = src.read(1)

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="viridis")
    plt.colorbar(label="Suitability (0–1)")
    plt.title(f"Habitat-Suitability: {t_pretty} vs {c_pretty}")
    plt.axis("off")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📁 PNG gespeichert: {out_png}")
    print("✅ Alles fertig.")
    return out_tif, out_png


def main():
    cfg = bootstrap_init(verbose=False)
    return build_prediction_map(cfg)


if __name__ == "__main__":
    main()