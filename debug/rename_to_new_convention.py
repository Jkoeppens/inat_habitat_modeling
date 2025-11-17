#!/usr/bin/env python3
"""
rename_to_new_convention.py

Konvertiert alte Dateinamen wie:
    NDVI_NDWI_MEAN_2023_09.tif
oder GEE-Exportnamen wie:
    2023_10_test.tif
oder:
    2021_08.tif

→ in neue Pipeline-Konvention:

    <region>_GEE_MONTHLY_YYYY_MM.tif
"""

from pathlib import Path
import re

# ---------------------------------
# EINSTELLUNGEN
# ---------------------------------
REGION = "berlin"
RAW_DIR = Path("/Volumes/Data/iNaturalist/raw") / REGION

print("📂 RAW-DIR:", RAW_DIR)

# Alte Muster, die wir erkennen wollen:
PATTERNS = [
    re.compile(r".*?(\d{4})[_-](\d{2}).*?\.tif$"),     # z.B. 2023_09, 2023-09, 2023_09_test
    re.compile(r"NDVI_NDWI_MEAN_(\d{4})_(\d{2})\.tif$"),  # alte Konvention
]

def parse_date_from_name(name: str):
    """Extrahiert YYYY, MM aus verschiedenen alten Dateinamen."""
    for pat in PATTERNS:
        m = pat.search(name)
        if m:
            return m.group(1), m.group(2)
    return None, None

def main():
    tif_files = list(RAW_DIR.glob("*.tif"))
    print(f"📦 Gefundene TIFFs: {len(tif_files)}")

    for tif in tif_files:
        year, month = parse_date_from_name(tif.name)
        if year is None:
            print("⚠️ Keine Jahres/Monatsstruktur erkannt → SKIP:", tif.name)
            continue

        new_name = f"{REGION}_GEE_MONTHLY_{year}_{month}.tif"
        new_path = tif.with_name(new_name)

        if new_path.exists():
            print("⚠️ Ziel existiert schon, überspringe:", new_name)
            continue

        print(f"🔁 {tif.name}  →  {new_name}")
        tif.rename(new_path)

    print("✅ Alle möglichen Dateien umbenannt.")

if __name__ == "__main__":
    main()