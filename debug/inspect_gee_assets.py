#!/usr/bin/env python3
# ======================================================================
# UNIVERSAL GEE ASSET INSPECTOR
#   - Einzelnes Asset inspizieren (--asset)
#   - Folder inspizieren (--folder)
#   - Optionaler Pattern-Filter (--pattern)
#
# Speichert Debug-Report als JSON unter:
#   debug/asset_reports/<asset_id>.json
# ======================================================================

import ee
import argparse
import json
from pathlib import Path
import sys

# ----------------------------------------------------------------------
# Earth Engine Init
# ----------------------------------------------------------------------
def init_ee():
    try:
        ee.Initialize()
        print("✔ EE initialisiert.")
    except Exception:
        print("🔑 EE Login erforderlich...")
        ee.Authenticate()
        ee.Initialize()
        print("✔ EE authentifiziert + initialisiert.")


# ----------------------------------------------------------------------
# Region-FIX: asset geometry ist unbrauchbar → WGS84 BBOX nutzen
# ----------------------------------------------------------------------
def get_default_region(cfg):
    try:
        bbox = cfg["region"]["bbox_wgs84"]
        return ee.Geometry.Rectangle(bbox)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Einzelnes Asset inspizieren
# ----------------------------------------------------------------------
def inspect_single_asset(asset_id, region=None):
    print(f"\n🔍 INSPECT: {asset_id}")

    try:
        metadata = ee.data.getAsset(asset_id)
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return

    print("📄 METADATA:")
    print(json.dumps(metadata, indent=2))

    img = ee.Image(asset_id)
    bands = img.bandNames().getInfo()

    # Hauptstruktur
    report = {
        "asset_id": asset_id,
        "bands": bands,
        "metadata": metadata,
        "projections": {},
        "stats": {},
        "thumbnail_url": None,
    }

    # Projektionen
    print("\n🧭 PROJEKTIONEN:")
    for b in bands:
        proj = img.select(b).projection().getInfo()
        report["projections"][b] = proj
        print(f" • {b} → {proj}")

    # Region-Fallback
    if region is None:
        try:
            region = img.geometry()
        except Exception:
            region = None

    # Stats (falls region bekannt)
    if region:
        print("\n📊 BERECHNE STATISTIKEN...")
        try:
            stats = img.reduceRegion(
                reducer=ee.Reducer.minMax().combine(
                    ee.Reducer.mean(), "", True
                ),
                geometry=region,
                scale=50,
                bestEffort=True,
            ).getInfo()
            report["stats"] = stats
            print(json.dumps(stats, indent=2))
        except Exception as e:
            print("⚠️ Keine Stats möglich:", e)
    else:
        print("⚠️ Keine Region verfügbar → Stats übersprungen.")

    # Thumbnail
    if region:
        print("\n🖼️ THUMBNAIL...")
        try:
            url = img.getThumbURL({
                "min": -0.2,
                "max": 0.9,
                "region": region.toGeoJSONString(),
                "dimensions": 512
            })
            report["thumbnail_url"] = url
            print("Thumbnail URL:", url)
        except Exception as e:
            print("⚠️ Thumbnail Fehler:", e)

    return report


# ----------------------------------------------------------------------
# Folder inspizieren
# ----------------------------------------------------------------------
def inspect_folder(folder, pattern=None, region=None):
    print(f"\n📁 LISTE ASSETS: {folder}")

    try:
        listing = ee.data.listAssets({"parent": folder})
    except Exception as e:
        print("❌ Fehler beim Listen:", e)
        sys.exit(1)

    assets = listing.get("assets", [])

    print(f"📦 Gefundene Assets: {len(assets)}")

    # Optional filtern
    selected = []
    for a in assets:
        name = a["name"]
        if (pattern is None) or (pattern in name):
            selected.append(name)

    print(f"📌 Selektiert: {len(selected)} Assets")
    for s in selected:
        print("  •", s)

    return {asset: inspect_single_asset(asset, region) for asset in selected}


# ----------------------------------------------------------------------
# Save report
# ----------------------------------------------------------------------
def save_report(report, out_dir="debug/asset_reports"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    name = report["asset_id"].replace("/", "_")
    path = out / f"{name}.json"

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"💾 Gespeichert: {path}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Asset Inspector")
    parser.add_argument("--asset", type=str, help="Einzelnes Asset inspizieren")
    parser.add_argument("--folder", type=str, help="Asset-Folder inspizieren")
    parser.add_argument("--pattern", type=str, help="Filter für Assetnamen")
    parser.add_argument("--use-cfg-region", action="store_true",
                        help="Region aus cfg verwenden")
    args = parser.parse_args()

    # EE start
    init_ee()

    # Optional cfg laden
    region = None
    if args.use_cfg_region:
        try:
            from bootstrap import init as bootstrap_init
            cfg = bootstrap_init(verbose=False)
            region = get_default_region(cfg)
            print("✔ Region aus cfg geladen.")
        except Exception:
            print("⚠️ cfg konnte nicht geladen werden → Region unbekannt.")

    if args.asset:
        report = inspect_single_asset(args.asset, region)
        save_report(report)

    elif args.folder:
        folder_report = inspect_folder(args.folder, args.pattern, region)
        for rep in folder_report.values():
            save_report(rep)

    else:
        print("❌ Bitte --asset oder --folder angeben.")
        sys.exit(1)