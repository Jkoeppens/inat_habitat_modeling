#!/usr/bin/env python
"""
Monatliche Klimatologie kachelbasiert, regionsfähig.
"""

import sys
import logging
from pathlib import Path
import re
import numpy as np
import rasterio
from rasterio.windows import Window

# ------------------------------------------------------------
# Config Loader
# ------------------------------------------------------------
from bootstrap import init as bootstrap_init

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Hauptfunktion
# ============================================================
    
def build_month_climatology_tiled(
    month: int,
    region: str = None,
    cfg: dict = None,
    tile_size: int = 2048,
):
    """
    Erzeugt eine Monats-Klimatologie über mehrere Jahre:

    Input:
      <base_data_dir>/raw/<region>/<region>_GEE_MONTHLY_YYYY_MM.tif
      <base_data_dir>/processed/<region>/<region>_GEE_MONTHLY_YYYY_MM_AUTOCORR.tif

    Output:
      <base_data_dir>/processed/<region>/CLIMATOLOGY_<region>_MONTH_XX.tif
    """

    # ----------------------------------------------------
    # 1. Konfiguration & Region
    # ----------------------------------------------------
    if cfg is None:
        cfg = bootstrap_init(verbose=False)

    if region is None:
        region = cfg["defaults"]["region"]

    # Basisverzeichnis der Daten
    base = Path(cfg["paths"]["base_data_dir"])

    RAW_DIR = base / "raw" / region
    PROC_DIR = base / "processed" / region
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("📂 RAW-DIR: %s", RAW_DIR)
    logger.info("📂 PROC-DIR: %s", PROC_DIR)

    # ----------------------------------------------------
    # 2. Dateien finden
    # ----------------------------------------------------
    patt = re.compile(rf"{region}_GEE_MONTHLY_(\d{{4}})_{month:02d}\.tif$")

    main_files = []
    years = []

    for f in RAW_DIR.glob(f"{region}_GEE_MONTHLY_*.tif"):
        m = patt.search(f.name)
        if m:
            main_files.append(f)
            years.append(int(m.group(1)))

    main_files = sorted(main_files)
    years = sorted(years)

    if not main_files:
        logger.error("❌ Keine GEE-Dateien für %s Monat %02d.", region, month)
        return

    logger.info("📦 Gefundene NDVI/NDWI Monats-TIFFs (%d):", len(main_files))
    for f in main_files:
        logger.info("   • %s", f.name)

    # ----------------------------------------------------
    # 3. Autokorrelation finden
    # ----------------------------------------------------
    ac_files = []
    for y in years:
        fp = PROC_DIR / f"{region}_GEE_MONTHLY_{y}_{month:02d}_AUTOCORR.tif"
        if fp.exists():
            ac_files.append(fp)
        else:
            logger.warning("⚠️ AUTOCORR fehlt: %s", fp.name)

    use_autocorr = len(ac_files) > 0

    if use_autocorr:
        logger.info("📦 Gefundene AUTOCORR-TIFFs (%d):", len(ac_files))
    else:
        logger.info("ℹ️ Keine AUTOCORR-Daten – wird übersprungen")

    # ----------------------------------------------------
    # 4. Basisraster öffnen
    # ----------------------------------------------------
    with rasterio.open(main_files[0]) as src0:
        height = src0.height
        width = src0.width
        profile = src0.profile.copy()
        transform = src0.transform

        logger.info("🗺️ Rastergröße: width=%d, height=%d", width, height)
        logger.info("📐 CRS: %s", src0.crs)

    # ----------------------------------------------------
    # 5. Output-Datei erstellen
    # ----------------------------------------------------
    out_name = PROC_DIR / f"CLIMATOLOGY_{region}_MONTH_{month:02d}.tif"

    profile.update(
        count=12 if use_autocorr else 8,
        dtype="float32",
        nodata=-9999.0,
        compress="LZW",
        predictor=2,
        BIGTIFF="YES",
    )

    logger.info("💾 Erzeuge Output-Datei: %s", out_name)

    main_sources = [rasterio.open(fp) for fp in main_files]
    ac_sources = [rasterio.open(fp) for fp in ac_files] if use_autocorr else []

    # ----------------------------------------------------
    # 6. Kachelbasiert verarbeiten
    # ----------------------------------------------------
    try:
        with rasterio.open(out_name, "w", **profile) as dst:
            dst.transform = transform

            n_tiles_y = (height + tile_size - 1) // tile_size
            n_tiles_x = (width + tile_size - 1) // tile_size
            total_tiles = n_tiles_x * n_tiles_y

            logger.info("🧩 Tiles: %d × %d  (total=%d)", n_tiles_x, n_tiles_y, total_tiles)

            tile_idx = 0

            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):

                    tile_idx += 1

                    row_off = ty * tile_size
                    col_off = tx * tile_size
                    win_h = min(tile_size, height - row_off)
                    win_w = min(tile_size, width - col_off)

                    window = Window(col_off, row_off, win_w, win_h)

                    logger.info("🧱 Tile %d/%d", tile_idx, total_tiles)

                    # NDVI/NDWI Stacks
                    ndvi_stack = []
                    ndwi_stack = []

                    for src in main_sources:
                        ndvi = src.read(1, window=window).astype("float32")
                        ndwi = src.read(2, window=window).astype("float32")

                        nodata = src.nodata
                        if nodata is not None:
                            ndvi = np.where(ndvi == nodata, np.nan, ndvi)
                            ndwi = np.where(ndwi == nodata, np.nan, ndwi)

                        ndvi_stack.append(ndvi)
                        ndwi_stack.append(ndwi)

                    ndvi_stack = np.stack(ndvi_stack)
                    ndwi_stack = np.stack(ndwi_stack)

                    # Aggregationen
                    with np.errstate(invalid="ignore"):
                        ndvi_median = np.nanmedian(ndvi_stack, axis=0)
                        ndvi_mean = np.nanmean(ndvi_stack, axis=0)
                        ndvi_std = np.nanstd(ndvi_stack, axis=0)
                        ndvi_cov = np.isfinite(ndvi_stack).sum(axis=0) / ndvi_stack.shape[0]

                        ndwi_median = np.nanmedian(ndwi_stack, axis=0)
                        ndwi_mean = np.nanmean(ndwi_stack, axis=0)
                        ndwi_std = np.nanstd(ndwi_stack, axis=0)
                        ndwi_cov = np.isfinite(ndwi_stack).sum(axis=0) / ndwi_stack.shape[0]

                    # Helper
                    def clean(x):
                        return np.nan_to_num(x, nan=-9999.0).astype("float32")

                    dst.write(clean(ndvi_median), 1, window=window)
                    dst.write(clean(ndvi_mean),   2, window=window)
                    dst.write(clean(ndvi_std),    3, window=window)
                    dst.write(clean(ndvi_cov),    4, window=window)

                    dst.write(clean(ndwi_median), 5, window=window)
                    dst.write(clean(ndwi_mean),   6, window=window)
                    dst.write(clean(ndwi_std),    7, window=window)
                    dst.write(clean(ndwi_cov),    8, window=window)

                    # Autokorrelation
                    if use_autocorr:
                        moran_ndvi = []
                        geary_ndvi = []
                        moran_ndwi = []
                        geary_ndwi = []

                        for src in ac_sources:
                            ac = src.read(window=window).astype("float32")
                            moran_ndvi.append(ac[0])
                            geary_ndvi.append(ac[1])
                            moran_ndwi.append(ac[2])
                            geary_ndwi.append(ac[3])

                        dst.write(clean(np.nanmean(moran_ndvi, axis=0)),  9, window=window)
                        dst.write(clean(np.nanmean(geary_ndvi, axis=0)), 10, window=window)
                        dst.write(clean(np.nanmean(moran_ndwi, axis=0)), 11, window=window)
                        dst.write(clean(np.nanmean(geary_ndwi, axis=0)), 12, window=window)

            logger.info("✅ Klimatologie fertig geschrieben: %s", out_name)

    finally:
        for s in main_sources:
            s.close()
        for s in ac_sources:
            s.close()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    month = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    build_month_climatology_tiled(month)