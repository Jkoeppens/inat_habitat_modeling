#!/usr/bin/env python3
# ======================================================================
# export_gee_assets_to_drive.py
# Exportiert alle erzeugten Monthly Composites → Google Drive
#
# Neue Struktur:
#   projects/<project-id>/assets/<region>/<year>_<month>
#
# Bänder:
#   NDVI_MEAN, NDWI_MEAN, NDVI_COUNT, NDWI_COUNT
# ======================================================================

import ee
import sys
import time
from pathlib import Path

# ---------------------------------------------------------
# Projekt-Root auf sys.path setzen
# ---------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import bootstrap


# ---------------------------------------------------------
# Earth Engine Initialization
# ---------------------------------------------------------
def init_ee(project_id):
    try:
        ee.Initialize(project=project_id)
        print("✔ Earth Engine bereits initialisiert.")
    except Exception:
        print("🔑 Login erforderlich – bitte im Browser autorisieren…")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("✔ Earth Engine erfolgreich authentifiziert.")


# ---------------------------------------------------------
# Liste aller Monats-Assets im Regionsordner
# ---------------------------------------------------------
def list_region_assets(region_folder):
    print(f"📂 Liste Assets in: {region_folder}")

    try:
        result = ee.data.listAssets({"parent": region_folder})
    except Exception as e:
        print("❌ Fehler beim Listen der Assets:", e)
        sys.exit(1)

    raw = result.get("assets", [])

    if not raw:
        print("❌ Keine Assets gefunden!")
        sys.exit(0)

    # Assets sehen so aus:
    #   projects/.../assets/berlin/2023_10
    assets = sorted(a["name"] for a in raw)
    return assets


# ---------------------------------------------------------
# Export eines Monatsimages → Drive
# ---------------------------------------------------------
def export_to_drive(asset_id, drive_folder="iNaturalist/data"):
    """
    Exportiert EIN Monatsasset zu Google Drive.
    """
    try:
        img = ee.Image(asset_id)
    except Exception as e:
        print(f"❌ Fehler beim Laden: {asset_id}", e)
        return

    img = img.float()

    basename = asset_id.split("/")[-1]
    print(f"🚀 Export: {basename}")

    try:
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=basename,
            folder=drive_folder,
            fileNamePrefix=basename,
            scale=10,
            region=img.geometry(),
            maxPixels=1e13,
        )
        task.start()
        print(f"   ✔ Task gestartet ({basename})")
    except Exception as e:
        print(f"❌ Fehler beim Export von {basename}:", e)


# ---------------------------------------------------------
# Export aller Monatsbilder der Region
# ---------------------------------------------------------
def export_all_months(cfg):
    project_id = cfg["gee"]["project_id"]
    region = cfg["defaults"]["region"]
    drive_folder = cfg["export"].get("drive_folder", "iNaturalist/data")

    region_folder = f"projects/{project_id}/assets/{region}"

    # 1) Liste Assets
    assets = list_region_assets(region_folder)

    print(f"\n📦 Gefundene Monats-Assets ({region}): {len(assets)}")
    for a in assets:
        print(" •", a)

    print("\n==============================")
    print("🚛 STARTE GOOGLE DRIVE EXPORT")
    print("==============================\n")

    # 2) Loop über alle Assets
    for asset_id in assets:
        export_to_drive(asset_id, drive_folder=drive_folder)
        time.sleep(0.3)

    print("\n🎉 Alle Tasks angestoßen!")
    print("👉 https://code.earthengine.google.com/tasks")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    cfg = bootstrap.init(verbose=False)
    project_id = cfg["gee"]["project_id"]

    init_ee(project_id)
    export_all_months(cfg)