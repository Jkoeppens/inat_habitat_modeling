"""
gee_monthly_composites.py

Erzeugt monatliche NDVI/NDWI-Composites aus Sentinel-2 SR (harmonized)
mit Cloud-Probability-Join und robuster Pixel-Logik.

Datenquellen:
  - COPERNICUS/S2_SR_HARMONIZED
  - COPERNICUS/S2_CLOUD_PROBABILITY

Pipeline:
  1. Filter auf Region (cfg.region.bbox_wgs84) und Zeitraum (year, month)
  2. Inner Join S2_SR <-> Cloud Probability via system:index
  3. Maskierung:
       - SCL in {4, 5, 6, 7}
       - CLOUD_PROB <= cloud_prob_thresh
  4. Berechnung von NDVI und NDWI
  5. Pixel müssen min_obs gültige Beobachtungen haben
  6. Ausgabe-Bänder:

       NDVI_MEAN
       NDWI_MEAN
       NDVI_COUNT
       NDWI_COUNT

Asset-Namenskonvention:
  projects/<project_id>/assets/monthly/<region_key>/<year>_<month><suffix>

Beispiel:
  projects/inaturalist-474012/assets/monthly/berlin/2023_10_min0

Aufruf-Beispiel im Notebook:

    import bootstrap
    import gee_monthly_composites as gmc

    cfg = bootstrap.init(verbose=True)

    task, asset_id = gmc.create_monthly_composite(
        config=cfg,
        year=2023,
        month=10,
        min_obs=0,
        suffix="min0"
    )
    print(asset_id)
"""

from __future__ import annotations

import ee
from typing import Tuple

from utils.gee_init import initialize_gee


# --------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------

def _get_region_bbox(cfg) -> list:
    """
    Holt die aktive Region-BBOX (WGS84) aus cfg.
    Erwartet: cfg["region"]["bbox_wgs84"] = [min_lon, min_lat, max_lon, max_lat]
    """
    region_cfg = cfg.get("region", {})
    bbox = region_cfg.get("bbox_wgs84")

    if not bbox or len(bbox) != 4:
        raise ValueError(
            "❌ cfg['region']['bbox_wgs84'] fehlt oder ist ungültig.\n"
            f"  Gefunden: {bbox}"
        )
    return bbox


def _get_region_key(cfg) -> str:
    """
    Liefert den Regions-Key (z.B. 'berlin') für Asset-Namen.
    Priorität:
      1. cfg['defaults']['region']
      2. 'region'
    """
    defaults = cfg.get("defaults", {})
    key = defaults.get("region")
    if not key:
        # Fallback – sollte eigentlich nie gebraucht werden
        key = "region"
    return str(key)


# --------------------------------------------------------------------
# 1) Wolkenmaskierung & Indizes
# --------------------------------------------------------------------

def _mask_s2_scl(img: ee.Image) -> ee.Image:
    """
    Maskiert Sentinel-2 basierend auf SCL.

    Behaltene Klassen:
      4 = Vegetation
      5 = Nicht-Vegetation
      6 = Wasser
      7 = Unklassifiziert
    """
    scl = img.select("SCL")
    good = (
        scl.eq(4)
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(7))
    )
    return img.updateMask(good)


def _add_indices(img: ee.Image) -> ee.Image:
    """Berechnet NDVI & NDWI und hängt sie als neue Bänder an."""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return img.addBands([ndvi, ndwi])


# --------------------------------------------------------------------
# 2) Sentinel-2 SR + Cloud Probability joinen
# --------------------------------------------------------------------

def _build_s2_with_cloudprob(
    region: ee.Geometry,
    start: ee.Date,
    end: ee.Date,
    max_cloud_pct: int = 80,
    verbose: bool = True
) -> ee.ImageCollection:
    """
    Baut eine ImageCollection mit:
      - Sentinel-2 SR Bänder + SCL
      - zusätzlichem Band CLOUD_PROB (0–100)
    über einen inner Join auf system:index.
    """

    s2_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
    )

    s2_cloud = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(region)
        .filterDate(start, end)
    )

    if verbose:
        print(f"   🛰️ S2_SR Szenen: {s2_sr.size().getInfo()}")
        print(f"   ☁️  Cloud-Prob Szenen: {s2_cloud.size().getInfo()}")

    join_filter = ee.Filter.equals(
        leftField="system:index",
        rightField="system:index",
    )

    inner_join = ee.Join.inner()
    joined = inner_join.apply(s2_sr, s2_cloud, join_filter)

    def _merge_pair(feat):
        primary = ee.Image(feat.get("primary"))
        secondary = ee.Image(feat.get("secondary"))
        cloud_prob = secondary.select("probability").rename("CLOUD_PROB")
        return (
            primary.addBands(cloud_prob)
            .copyProperties(primary, primary.propertyNames())
        )

    merged = ee.ImageCollection(joined.map(_merge_pair))

    if verbose:
        print(f"   🔗 Gemergte Szenen: {merged.size().getInfo()}")

    return merged


# --------------------------------------------------------------------
# 3) Monatsbild bauen (NDVI/NDWI + Counts)
# --------------------------------------------------------------------

def build_monthly_image(
    cfg: dict,
    year: int,
    month: int,
    min_obs: int = 1,
    cloud_prob_thresh: int = 60,
    max_cloud_pct: int = 80,
    verbose: bool = True,
) -> Tuple[ee.Image, ee.Geometry, str, float]:
    """
    Erzeugt ein monatliches Composite.

    Bänder im Output:
      - NDVI_MEAN
      - NDWI_MEAN
      - NDVI_COUNT
      - NDWI_COUNT

    Pixel-Logik:
      - Region: cfg.region.bbox_wgs84
      - SCL ∈ {4, 5, 6, 7}
      - CLOUD_PROB ≤ cloud_prob_thresh
      - mind. min_obs Beobachtungen pro Pixel
    """

    # --- Konfig auslesen ---
    bbox = _get_region_bbox(cfg)
    crs = cfg.get("gee", {}).get("crs", "EPSG:32633")
    scale = float(cfg.get("gee", {}).get("scale", 10))
    project_id = cfg.get("gee", {}).get("project_id")

    # --- GEE sicher initialisieren ---
    initialize_gee(project_id=project_id, verbose=verbose)

    region = ee.Geometry.Rectangle(bbox)
    start = ee.Date.fromYMD(int(year), int(month), 1)
    end = start.advance(1, "month")

    if verbose:
        print(f"\n🗓️ Baue Monats-Composite {year}-{month:02d}")
        print(f"   🗺️ Region (WGS84): {bbox}")
        print(f"   ⚙️ min_obs={min_obs}, cloud_prob_thresh={cloud_prob_thresh}, max_cloud_pct={max_cloud_pct}")
        print(f"   🧭 CRS={crs}, scale={scale}")

    # --- S2 + CloudProb joinen ---
    s2 = _build_s2_with_cloudprob(
        region=region,
        start=start,
        end=end,
        max_cloud_pct=max_cloud_pct,
        verbose=verbose,
    )

    if s2.size().getInfo() == 0:
        raise RuntimeError(f"❌ Keine joined Szenen für {year}-{month:02d}")

    # --- Masken anwenden ---
    s2_masked = (
        s2
        .map(_mask_s2_scl)
        .map(lambda img: img.updateMask(img.select("CLOUD_PROB").lte(cloud_prob_thresh)))
    )

    # --- NDVI/NDWI hinzufügen ---
    s2_idx = s2_masked.map(_add_indices)

    ndvi_coll = s2_idx.select("NDVI")
    ndwi_coll = s2_idx.select("NDWI")

    ndvi_count = ndvi_coll.count().rename("NDVI_COUNT")
    ndwi_count = ndwi_coll.count().rename("NDWI_COUNT")

    valid_mask = ndvi_count.gte(min_obs)

    ndvi_mean = ndvi_coll.mean().updateMask(valid_mask).rename("NDVI_MEAN")
    ndwi_mean = ndwi_coll.mean().updateMask(valid_mask).rename("NDWI_MEAN")

    # --- Optional: grobe Stats ---
    if verbose:
        try:
            stats = ndvi_mean.reduceRegion(
                reducer=ee.Reducer.minMax().combine(
                    ee.Reducer.mean(), sharedInputs=True
                ),
                geometry=region,
                scale=scale * 10,
                maxPixels=1e7,
                bestEffort=True,
            ).getInfo()
            print("   📊 NDVI_MEAN (rough stats):", stats)
        except Exception as e:
            print("   ⚠️ NDVI-Stats nicht berechenbar:", e)

    out = (
        ndvi_mean
        .addBands([ndwi_mean, ndvi_count, ndwi_count])
        .set("year", int(year))
        .set("month", int(month))
        .set("min_obs", int(min_obs))
        .set("cloud_prob_thresh", int(cloud_prob_thresh))
        .set("max_cloud_pct", int(max_cloud_pct))
        .set("region_bbox", bbox)
        .set("crs", crs)
        .set("scale", scale)
        .set("generator", "gee_monthly_composites_v2")
    )

    return out, region, crs, scale


# --------------------------------------------------------------------
# 4) Export als Asset
# --------------------------------------------------------------------

def create_monthly_composite(
    config: dict,
    year: int,
    month: int,
    min_obs: int = 1,
    suffix: str = "",
    cloud_prob_thresh: int = 60,
    max_cloud_pct: int = 80,
    verbose: bool = True,
):
    """
    Baut das Monatsbild und exportiert es als GEE-Asset.

    Asset-Name:
      projects/<project_id>/assets/monthly/<region_key>/<year>_<month><suffix>

    Beispiel:
      projects/inaturalist-474012/assets/monthly/berlin/2023_10_min0
    """

    project_id = config.get("gee", {}).get("project_id")
    if not project_id:
        raise ValueError("❌ config['gee']['project_id'] fehlt.")

    region_key = _get_region_key(config)

    img, region, crs, scale = build_monthly_image(
        cfg=config,
        year=year,
        month=month,
        min_obs=min_obs,
        cloud_prob_thresh=cloud_prob_thresh,
        max_cloud_pct=max_cloud_pct,
        verbose=verbose,
    )

    suffix = f"_{suffix}" if suffix else ""
    asset_id = (
        f"projects/{project_id}/assets/"
        f"{region_key}/{year}_{month:02d}{suffix}"
    )

    if verbose:
        print(f"🎯 Asset-ID: {asset_id}")

    task = ee.batch.Export.image.toAsset(
        image=img,
        description=f"monthly_{region_key}_{year}_{month:02d}{suffix}",
        assetId=asset_id,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
    )
    task.start()

    if verbose:
        print("🚀 Export gestartet.")

    return task, asset_id


# --------------------------------------------------------------------
# 5) Mini-Selbsttest (nur bei direktem Aufruf)
# --------------------------------------------------------------------

if __name__ == "__main__":
    print("Dieses Modul ist zum Import gedacht (z.B. in Notebooks),")
    print("nicht zum direkten Ausführen.")