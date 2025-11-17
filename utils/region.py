# inat_habitat_modeling/utils/region.py

from pathlib import Path

try:
    from pyproj import Transformer
    HAVE_PYPROJ = True
except Exception:
    HAVE_PYPROJ = False


# ----------------------------------------------------------------------
# 🧭 Region normalisieren (bbox_utm + Synchronisierung)
# ----------------------------------------------------------------------
def normalize_region(cfg, verbose=True):
    """
    Normalisiert die *aktive* Region.
    Erwartet:
        cfg["region"] = {
            "bbox_wgs84": [...],
            "utm_crs": "...",
            ...
        }

    Ergänzt:
        cfg["region"]["bbox_utm"]
        cfg["gee"]["region_bbox"]
        cfg["gee"]["region_bbox_utm"]
        cfg["gee"]["crs"]

    Gibt cfg zurück (mutiert es).
    """

    if "region" not in cfg:
        raise ValueError("❌ cfg['region'] fehlt – wurde keine Region ausgewählt?")

    region = cfg["region"]

    # ------------------------------------------------------
    # 1. WGS84-BBOX prüfen
    # ------------------------------------------------------
    bbox = region.get("bbox_wgs84")
    if not bbox or len(bbox) != 4:
        raise ValueError(f"❌ Ungültige region.bbox_wgs84: {bbox}")

    if verbose:
        print(f"🌍 Region WGS84: {bbox}")

    # ------------------------------------------------------
    # 2. UTM-CRS bestimmen
    # ------------------------------------------------------
    utm_crs = region.get("utm_crs") or cfg.get("gee", {}).get("crs") or "EPSG:32633"
    region["utm_crs"] = utm_crs

    if verbose:
        print(f"   → UTM-CRS: {utm_crs}")

    # ------------------------------------------------------
    # 3. bbox → UTM transformieren
    # ------------------------------------------------------
    if HAVE_PYPROJ:
        try:
            trans = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
            x0, y0 = trans.transform(bbox[0], bbox[1])
            x1, y1 = trans.transform(bbox[2], bbox[3])
            region["bbox_utm"] = [x0, y0, x1, y1]

            if verbose:
                print(f"   → bbox_utm: {region['bbox_utm']}")

        except Exception as e:
            region["bbox_utm"] = None
            if verbose:
                print("⚠️ bbox_utm konnte nicht berechnet werden:", e)

    else:
        region["bbox_utm"] = None
        if verbose:
            print("⚠️ pyproj nicht installiert – bbox_utm entfällt.")

    # ------------------------------------------------------
    # 4. GEE synchronisieren
    # ------------------------------------------------------
    gee_cfg = cfg.setdefault("gee", {})
    gee_cfg["region_bbox"] = bbox
    gee_cfg["region_bbox_utm"] = region.get("bbox_utm")
    gee_cfg["crs"] = utm_crs

    if verbose:
        print("   ✔ Region → GEE synchronisiert\n")

    return cfg