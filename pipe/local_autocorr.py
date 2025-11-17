#!/usr/bin/env python
"""
local_autocorr.py

Berechnet Moran & Geary für NDVI/NDWI aus GEE-Monatscomposites,
regionenspezifisch, liest Pfade aus default.yaml, arbeitet mit
neuem Dateinamensschema:

    {region}_GEE_MONTHLY_YYYY_MM.tif
"""

import os
import re
import yaml
import numpy as np
from pathlib import Path
import rasterio
from scipy.ndimage import uniform_filter

# =====================================================================
# 0. Konfiguration laden
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = PROJECT_ROOT / "config" / "default.yaml"

with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

region = cfg["defaults"]["region"]
base = Path(cfg["paths"]["base_data_dir"])

RAW_DIR = base / "raw" / region
PROC_DIR = base / "processed" / region

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 RAW-Root: {RAW_DIR}")
print(f"📁 OUT-Root: {PROC_DIR}")

# =====================================================================
# 1. Autocorrelation
# =====================================================================

def compute_local_moran_geary(arr, window_size=11, min_valid=5):
    arr = arr.astype(np.float32)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return np.full_like(arr, np.nan), np.full_like(arr, np.nan)

    arr_f = np.where(mask, arr, 0.0)
    arr2_f = arr_f ** 2
    area = float(window_size * window_size)

    valid_count = uniform_filter(mask.astype(np.float32),
                                 size=window_size,
                                 mode="constant", cval=0.0) * area

    sum_x = uniform_filter(arr_f, size=window_size, mode="constant", cval=0.0) * area
    sum_x2 = uniform_filter(arr2_f, size=window_size, mode="constant", cval=0.0) * area

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = sum_x / valid_count
        mean_x2 = sum_x2 / valid_count
        var_local = mean_x2 - mean_x**2
        var_local[var_local < 0] = 0

    global_mean = np.nanmean(arr)
    global_std = np.nanstd(arr)
    if global_std == 0:
        return np.full_like(arr, np.nan), np.full_like(arr, np.nan)

    z = (arr - global_mean) / global_std
    z_f = np.where(mask, z, 0.0)

    sum_z = uniform_filter(z_f, size=window_size, mode="constant", cval=0.0) * area

    center_valid = mask.astype(np.float32)
    neighbor_count = valid_count - center_valid
    neighbor_count = np.where(neighbor_count < 1, np.nan, neighbor_count)

    lag_z = (sum_z - z_f) / neighbor_count
    moran = z * lag_z

    Ex = (sum_x - arr_f) / neighbor_count
    Ex2 = (sum_x2 - arr2_f) / neighbor_count

    geary = (arr**2 - 2*arr*Ex + Ex2) / (2 * global_std**2)

    cond = (neighbor_count >= min_valid) & mask
    moran = np.where(cond, moran, np.nan)
    geary = np.where(cond, geary, np.nan)

    return moran.astype(np.float32), geary.astype(np.float32)

# =====================================================================
# 2. Dateiweise Verarbeitung
# =====================================================================

def process_file(fpath: Path):
    print(f"\n🔍 Verarbeite: {fpath.name}")

    out_path = PROC_DIR / f"{fpath.stem}_AUTOCORR.tif"

    with rasterio.open(fpath) as src:
        ndvi = src.read(1).astype(np.float32)
        ndwi = src.read(2).astype(np.float32)
        profile = src.profile.copy()

    moran_ndvi, geary_ndvi = compute_local_moran_geary(ndvi)
    moran_ndwi, geary_ndwi = compute_local_moran_geary(ndwi)

    profile.update(count=4, dtype="float32", nodata=None, compress="LZW")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(moran_ndvi, 1)
        dst.write(geary_ndvi, 2)
        dst.write(moran_ndwi, 3)
        dst.write(geary_ndwi, 4)

    print(f"   ✔ Gespeichert: {out_path}")

# =====================================================================
# 3. Main
# =====================================================================

def main():
    pattern = re.compile(rf"{region}_GEE_MONTHLY_(\d{{4}})_(\d{{2}})\.tif$")

    tifs = sorted([p for p in RAW_DIR.glob("*.tif") if pattern.search(p.name)])

    print(f"\n📦 Gefundene TIFFs ({region}): {len(tifs)}")
    for p in tifs:
        print("  •", p.name)

    if not tifs:
        print("❌ Keine passenden TIFFs gefunden.")
        return

    for f in tifs:
        process_file(f)

if __name__ == "__main__":
    main()