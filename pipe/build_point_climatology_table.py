#!/usr/bin/env python3
"""
build_point_climatology_table.py

Verknüpft punktbasierte iNaturalist-Beobachtungen mit
monatlichen Klimatologie-Rastern (NDVI/NDWI + Autokorrelation).

- Liest Merged-iNat-CSV (Target vs. Kontrast) → aus cfg
- Konvertiert WGS84 → UTM (cfg['region']['utm_crs'])
- Sampelt für jeden Monat die entsprechenden CLIMATOLOGY-Raster
- Schreibt ein Feature-CSV pro Species-Paar:

  <features_dir_species>/inat_with_climatology_<target>_vs_<contrast>.csv

Erwartete Rasterstruktur (12 Bänder, wie in build_month_climatology_tiled):

  Band  1: NDVI median
  Band  2: NDVI mean
  Band  3: NDVI std
  Band  4: NDVI coverage
  Band  5: NDWI median
  Band  6: NDWI mean
  Band  7: NDWI std
  Band  8: NDWI coverage
  Band  9: mean Moran NDVI
  Band 10: mean Geary NDVI
  Band 11: mean Moran NDWI
  Band 12: mean Geary NDWI
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol


# ==========================================================
# 1. CSV sicher laden
# ==========================================================
def load_input_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    print(f"\n📄 Lade Input-Datei: {path}")
    try:
        df = pd.read_csv(path)
        if "month" in df.columns:
            df = df.drop(columns=["month"])
            print("🧹 'month' aus Feature-Tabelle entfernt.")
        if df.shape[1] == 1:
            raise ValueError("Nur 1 Spalte → vermutlich kaputtes CSV.")
        print("✔ CSV normal geladen.")
        return df
    except Exception:
        print("⚠ CSV wirkt defekt – versuche Whitespace-Parsing…")

        df = pd.read_csv(path, sep=r"\s+", engine="python")
        if df.shape[1] < 3:
            raise RuntimeError("❌ Datei konnte nicht repariert werden!")
        print(f"✔ Repariert → {df.shape[1]} Spalten erkannt.")

        # CSV in „normales“ Kommaformat zurückschreiben
        df.to_csv(path, index=False)
        print("💾 Datei neu gespeichert (korrektes CSV).")
        return df


# ==========================================================
# 2. Koordinaten nach UTM
# ==========================================================
def convert_to_utm(df: pd.DataFrame, utm_crs: str) -> pd.DataFrame:
    print(f"\n🗺️ Konvertiere Koordinaten nach UTM ({utm_crs})…")

    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    if "longitude" not in df.columns or "latitude" not in df.columns:
        raise ValueError("❌ Erwartet Spalten 'longitude' und 'latitude' im Input-CSV.")

    x, y = transformer.transform(df["longitude"].values, df["latitude"].values)
    df["x_utm"] = x
    df["y_utm"] = y

    print(f"   x_utm range: {df['x_utm'].min():.2f} → {df['x_utm'].max():.2f}")
    print(f"   y_utm range: {df['y_utm'].min():.2f} → {df['y_utm'].max():.2f}")

    print("\n📌 Beispielpunkte:")
    print(df[["latitude", "longitude", "x_utm", "y_utm"]].head(10))

    return df


# ==========================================================
# 3. Climatology-Raster laden (region-aware)
# ==========================================================


def load_rasters(processed_root: Path, region_key: str):
    processed_root = Path(processed_root)
    rasters = {}

    print("\n🌍 Lade Climatology-Raster…")

    for m in range(1, 13):
        # 1. Preferred file: with region prefix
        tif_region = processed_root / f"CLIMATOLOGY_{region_key}_MONTH_{m:02d}.tif"
        # 2. Fallback file: generic name
        tif_generic = processed_root / f"CLIMATOLOGY_MONTH_{m:02d}.tif"
        # 3. Fallback folder: processed/region/
        tif_region_subdir = processed_root / region_key / f"CLIMATOLOGY_{region_key}_MONTH_{m:02d}.tif"

        if tif_region.exists():
            rasters[m] = tif_region
            print(f"   ✔ Monat {m:02d}: {tif_region.name} (root)")
        elif tif_generic.exists():
            rasters[m] = tif_generic
            print(f"   ✔ Monat {m:02d}: {tif_generic.name} (generic)")
        elif tif_region_subdir.exists():
            rasters[m] = tif_region_subdir
            print(f"   ✔ Monat {m:02d}: {tif_region_subdir.name} (subdir)")
        else:
            print(f"   ⚠ Monat {m:02d}: kein TIFF gefunden")

    if not rasters:
        raise FileNotFoundError(
            f"❌ Keine CLIMATOLOGY TIFFs für {region_key} gefunden."
        )

    return rasters
# ==========================================================
# 4. Dynamische Bandnamen je nach TIFF
# ==========================================================
def infer_band_names(n_bands: int):
    """
    Ordnet die Bänder der Klimatologie einem Namen zu.
    Wichtig: Muss zur tatsächlichen Schreibweise in build_month_climatology_tiled passen.
    """
    if n_bands == 12:
        # exakt wie in deiner aktuellen Klimatologie:
        return [
            "ndvi_median",    # 1
            "ndvi_mean",      # 2
            "ndvi_std",       # 3
            "ndvi_coverage",  # 4
            "ndwi_median",    # 5
            "ndwi_mean",      # 6
            "ndwi_std",       # 7
            "ndwi_coverage",  # 8
            "moran_ndvi",     # 9
            "geary_ndvi",     # 10
            "moran_ndwi",     # 11
            "geary_ndwi",     # 12
        ]
    elif n_bands == 8:
        # Falls du eine Variante ohne Autokorrelation hast
        return [
            "ndvi_median", "ndvi_mean", "ndvi_std", "ndvi_coverage",
            "ndwi_median", "ndwi_mean", "ndwi_std", "ndwi_coverage",
        ]
    else:
        # Fallback – lieber etwas haben als crashen
        return [f"band{i+1}" for i in range(n_bands)]


# ==========================================================
# 5. Sampling eines Monats
# ==========================================================
def sample_month(df: pd.DataFrame, tif_path: Path, month: int):
    print(f"\n🔎 Sampling Monat {month:02d}: {tif_path.name}")

    with rasterio.open(tif_path) as src:
        n_bands = src.count
        height, width = src.height, src.width
        transform = src.transform
        nodata = src.nodata

        band_names = infer_band_names(n_bands)
        assert len(band_names) == n_bands, "Bandnamen passen nicht zur Bandanzahl!"

        # Debug: erste 10 Punkte
        print("\n🎯 Debug: Sampling erster 10 Punkte:")
        for i in range(min(10, len(df))):
            x, y = df.loc[df.index[i], "x_utm"], df.loc[df.index[i], "y_utm"]
            row, col = rowcol(transform, x, y)

            inside = (0 <= row < height) and (0 <= col < width)
            if not inside:
                print(f"  Punkt {i}: außerhalb (row={row}, col={col}) → NaN")
            else:
                vals = src.read(
                    indexes=list(range(1, n_bands + 1)),
                    window=((row, row + 1), (col, col + 1))
                ).reshape(n_bands)
                if nodata is not None:
                    vals = np.where(vals == nodata, np.nan, vals)
                print(f"  Punkt {i} row={row}, col={col} → {vals}")

        # Vollständiges Sampling
        sampled = np.zeros((len(df), n_bands), dtype="float32")

        for idx, row_series in enumerate(df.itertuples()):
            x = row_series.x_utm
            y = row_series.y_utm
            row, col = rowcol(transform, x, y)

            if 0 <= row < height and 0 <= col < width:
                vals = src.read(
                    indexes=list(range(1, n_bands + 1)),
                    window=((row, row + 1), (col, col + 1))
                ).reshape(n_bands)
                if nodata is not None:
                    vals = np.where(vals == nodata, np.nan, vals)
                sampled[idx] = vals
            else:
                sampled[idx] = np.nan

    # In Dict umbenennen
    out = {}
    for b in range(n_bands):
        key = f"m{month:02d}_{band_names[b]}"
        out[key] = sampled[:, b]

    return out


# ==========================================================
# 6. Gesamttabelle bauen (generischer Kern)
# ==========================================================
def build_feature_table(input_csv: Path,
                        processed_root: Path,
                        region_key: str,
                        utm_crs: str,
                        output_csv: Path):
    input_csv = Path(input_csv)
    processed_root = Path(processed_root)
    output_csv = Path(output_csv)

    # 1) CSV laden
    df = load_input_table(input_csv)

    # 2) Koordinaten → UTM
    df = convert_to_utm(df, utm_crs=utm_crs)

    # 3) Rasterliste
    rasters = load_rasters(processed_root, region_key)

    # 4) Alle Monate sampeln
    feature_dict = {}
    for month, tif in rasters.items():
        sampled = sample_month(df, tif, month)
        feature_dict.update(sampled)

    # 5) Features anhängen
    for col, arr in feature_dict.items():
        df[col] = arr

    print("\n🧮 Gesamtmatrix:", df.shape)
    print("📌 Beispiel-Feature-Spalten:", list(feature_dict.keys())[:8])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Fertig gespeichert: {output_csv}")

    return df


# ==========================================================
# 7. Komfortfunktion: Pfade aus cfg ableiten (NEU & robust)
# ==========================================================
def build_feature_table_for_cfg(cfg: dict):
    """
    Liest alle relevanten Pfade/Infos aus cfg und baut:

      <features_dir_species>/inat_with_climatology_<target>_vs_<contrast>.csv

    Diese Version nutzt die neue default.yaml Struktur:
      cfg["defaults"]["target_species"] → key in cfg["species"]
      cfg["defaults"]["contrast_species"] → key in cfg["species"]
    """

    # -------------------------
    # Region + CRS
    # -------------------------
    region_key = cfg["defaults"]["region"]
    utm_crs = cfg["regions"][region_key]["utm_crs"]

    # -------------------------
    # Species aus defaults holen
    # -------------------------
    tkey = cfg["defaults"]["target_species"]
    ckey = cfg["defaults"]["contrast_species"]

    target = cfg["species"][tkey]
    contrast = cfg["species"][ckey]

    tname = target["name"].replace(" ", "_")
    cname = contrast["name"].replace(" ", "_")

    # -------------------------
    # Basis-Ordner
    # -------------------------
    base_data_dir = Path(cfg["paths"]["base_data_dir"])

    processed_root = Path(cfg["paths"]["processed_dir"]) / region_key
    output_dir = Path(cfg["paths"]["output_dir"])
    features_dir = Path(cfg["paths"]["features_dir"])

    # Species-spezifische Ordner
    features_dir_species = features_dir / tname
    features_dir_species.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Input CSV suchen
    # -------------------------
    # Standard-Name des merge-Skripts
    merged_name = f"inat_merged_{tname}_vs_{cname}.csv"

    candidate_inputs = [
        output_dir / tname / merged_name,   # species-spezifischer Ordner
        output_dir / merged_name,           # allgemeiner Output
        output_dir / "inat_merged_labeled.csv",
    ]

    input_csv = None
    for p in candidate_inputs:
        if p.exists():
            input_csv = p
            break

    if input_csv is None:
        raise FileNotFoundError(
            "❌ Keine Merged-iNat-Datei gefunden.\nVersucht wurde:\n" +
            "\n".join(f"  - {p}" for p in candidate_inputs)
        )

    # -------------------------
    # Output-Datei
    # -------------------------
    output_csv = (
        features_dir_species
        / f"inat_with_climatology_{tname}_vs_{cname}.csv"
    )

    # -------------------------
    # Logging
    # -------------------------
    print("\n📌 Konfiguration für Feature-Build:")
    print(f"   Region:         {region_key}")
    print(f"   UTM-CRS:        {utm_crs}")
    print(f"   Raster-Root:    {processed_root}")
    print(f"   Input-CSV:      {input_csv}")
    print(f"   Output-CSV:     {output_csv}")

    # -------------------------
    # Tatsächlicher Build
    # -------------------------
    return build_feature_table(
        input_csv=input_csv,
        processed_root=processed_root,
        region_key=region_key,
        utm_crs=utm_crs,
        output_csv=output_csv,
    )

# ==========================================================
# 8. CLI
# ==========================================================
if __name__ == "__main__":
    # Direktlauf über cfg
    from bootstrap import init as bootstrap_init

    cfg = bootstrap_init(verbose=True)
    build_feature_table_for_cfg(cfg)