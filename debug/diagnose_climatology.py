#!/usr/bin/env python3
import logging
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------
# KORREKTE Bandnamen für deine 12-Band Climatology-Files
# ----------------------------------------------------------
BAND_NAMES = [
    "NDVI Median",
    "NDVI Mean",
    "NDVI Std",
    "NDVI Coverage",

    "NDWI Median",
    "NDWI Mean",
    "NDWI Std",
    "NDWI Coverage",

    "Moran NDVI",
    "Geary NDVI",
    "Moran NDWI",
    "Geary NDWI",
]

# ----------------------------------------------------------
# Diagnose einer einzelnen Datei
# ----------------------------------------------------------
def inspect_climatology(path):
    path = Path(path)
    log.info("📂 Lade Datei: %s", path)

    with rasterio.open(path) as src:
        meta = src.meta.copy()
        log.info("🧭 CRS: %s", meta["crs"])
        log.info("📐 Größe: %d × %d  | Bänder: %d",
                 meta["width"], meta["height"], meta["count"])
        log.info("🔢 Datentyp: %s | NoData: %s", meta["dtype"], meta["nodata"])

        # Dynamisch kürzen, falls mal weniger Bänder existieren
        names = BAND_NAMES[:meta["count"]]

        for i in range(1, meta["count"] + 1):
            name = names[i - 1]
            arr = src.read(i).astype(float)
            nodata = meta["nodata"]

            if nodata is not None:
                arr[arr == nodata] = np.nan

            valid = np.isfinite(arr)
            frac_valid = valid.mean() * 100

            log.info(f"\n=== {name} (Band {i}) ===")
            log.info("  ✓ gültige Pixel: %.2f%%", frac_valid)
            log.info("  ▸ min/max: %.3f / %.3f",
                     np.nanmin(arr), np.nanmax(arr))
            log.info("  ▸ mean/std: %.3f / %.3f",
                     np.nanmean(arr), np.nanstd(arr))

            # Thumbnail speichern
            out_img = path.with_suffix(f".band{i}.{name.replace(' ', '_')}.png")

            try:
                plt.figure(figsize=(4, 4))
                plt.imshow(arr, cmap="viridis")
                plt.title(f"{name}")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(out_img, dpi=150)
                plt.close()
                log.info("  🖼️ Thumbnail gespeichert: %s", out_img)
            except Exception as e:
                log.warning("  ⚠️ Thumbnail-Fehler (%s): %s", name, e)

# ----------------------------------------------------------
# Batch über alle Monatsdateien
# ----------------------------------------------------------
def inspect_all(processed_dir="data/processed"):
    processed = Path(processed_dir)
    files = sorted(processed.glob("CLIMATOLOGY_*_MONTH_*.tif"))

    if not files:
        log.error("❌ Keine Klimatologie-Dateien gefunden")
        return

    log.info("📦 Gefundene Climatology-Dateien: %d", len(files))
    for f in files:
        inspect_climatology(f)

# ----------------------------------------------------------
# Main CLI
# ----------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose für Klimatologie-TIFFs")

    parser.add_argument("--file", type=str,
                        help="Pfad zu einer einzelnen Datei")
    parser.add_argument("--all", action="store_true",
                        help="Alle Klimatologien diagnostizieren")
    parser.add_argument("--dir", type=str, default="data/processed",
                        help="Verzeichnis für Batch")

    args = parser.parse_args()

    if args.file:
        inspect_climatology(args.file)
    elif args.all:
        inspect_all(args.dir)
    else:
        parser.print_help()