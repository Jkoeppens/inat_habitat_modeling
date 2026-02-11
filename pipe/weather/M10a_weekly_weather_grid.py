# pipe/weather/m10a_weekly_weather_grid.py

import ee
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import date, timedelta

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
GRID_PATH = Path("/Volumes/Data/iNaturalist/weather/grid/grid_20km_DE.gpkg")
OUT_PATH  = Path("/Volumes/Data/iNaturalist/weather/derived/weather_grid_year_weekly_m10a.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

START_YEAR = 2017
END_YEAR   = 2024  # inclusive

SCALE_M = 10000
TILE_SCALE = 4

# ERA5-Land HOURLY (this is what we want)
ERA5_HOURLY = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")

# Bands (REAL names, from your band listing)
BANDS_MEAN = {
    "temp_2m": "temperature_2m",
    "soil_temp_l1": "soil_temperature_level_1",
    "soil_moist_l1": "volumetric_soil_water_layer_1",
    "snow_cover": "snow_cover",
}

BANDS_SUM = {
    "precip": "total_precipitation_hourly",
    "pet": "potential_evaporation_hourly",
    "evap": "total_evaporation_hourly",
}

# --------------------------------------------------
# EE INIT
# --------------------------------------------------
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def week_windows(year: int):
    """
    7-day blocks starting Jan 1.
    Last block may be shorter (end capped at Jan 1 next year).
    Returns list of (week_idx, start_date, end_date)
    """
    start = date(year, 1, 1)
    end_year = date(year + 1, 1, 1)
    windows = []
    wk = 1
    cur = start
    while cur < end_year:
        nxt = min(cur + timedelta(days=7), end_year)
        windows.append((wk, cur, nxt))
        wk += 1
        cur = nxt
    return windows


def build_grid_fc(grid_gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """
    Convert local grid polygons to an EE FeatureCollection.
    We use bounding boxes in EPSG:3035 and transform to EPSG:4326 (same trick as before).
    """
    feats = []
    for _, row in grid_gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds

        geom = (
            ee.Geometry.Rectangle(
                [minx, miny, maxx, maxy],
                proj="EPSG:3035",
                geodesic=False
            )
            .transform("EPSG:4326", 1)
        )

        feats.append(
            ee.Feature(geom, {"grid_id": str(row.grid_id)})
        )

    return ee.FeatureCollection(feats)


def weekly_image(start_dt: date, end_dt: date) -> ee.Image:
    """
    Build a weekly aggregated image from ERA5 HOURLY:
    - mean for some variables
    - sum for flux/accumulation variables
    """
    start_str = start_dt.isoformat()
    end_str   = end_dt.isoformat()

    col = ERA5_HOURLY.filterDate(start_str, end_str)

    # mean bands
    mean_imgs = []
    for out_name, band in BANDS_MEAN.items():
        img = col.select(band).mean().rename(out_name)
        mean_imgs.append(img)

    # sum bands
    sum_imgs = []
    for out_name, band in BANDS_SUM.items():
        img = col.select(band).sum().rename(out_name)
        sum_imgs.append(img)

    return ee.Image.cat(mean_imgs + sum_imgs)


def reduce_week_to_grid(img: ee.Image, grid_fc: ee.FeatureCollection, year: int, week: int, start_dt: date, end_dt: date):
    """
    Spatial mean over each grid cell for each band already aggregated temporally.
    """
    fc = img.reduceRegions(
        collection=grid_fc,
        reducer=ee.Reducer.mean(),
        scale=SCALE_M,
        tileScale=TILE_SCALE
    )

    start_str = start_dt.isoformat()
    end_str   = end_dt.isoformat()

    def add_meta(f):
        return f.set({
            "year": int(year),
            "week": int(week),
            "start_date": start_str,
            "end_date": end_str
        })

    return fc.map(add_meta)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print("🌍 Loading grid …")
    grid = gpd.read_file(GRID_PATH)[["grid_id", "geometry"]]
    print(f"🔲 Grid cells: {len(grid)}")

    print("🌐 Building EE grid FeatureCollection …")
    grid_fc = build_grid_fc(grid)

    rows = []
    years = list(range(START_YEAR, END_YEAR + 1))

    for y in years:
        print(f"\n📅 YEAR {y}")
        windows = week_windows(y)
        print(f"   weeks: {len(windows)}")

        for (wk, start_dt, end_dt) in windows:
            print(f"   🧊 week {wk:02d}: {start_dt} → {end_dt}")

            img = weekly_image(start_dt, end_dt)
            fc = reduce_week_to_grid(img, grid_fc, y, wk, start_dt, end_dt)

            # Pull to client (1019 features); one call per week, not per cell.
            out = fc.getInfo()

            for feat in out["features"]:
                prop = feat["properties"]

                # Keep only what we need (and normalize missing)
                rows.append({
                    "grid_id": prop.get("grid_id"),
                    "year": prop.get("year"),
                    "week": prop.get("week"),
                    "start_date": prop.get("start_date"),
                    "end_date": prop.get("end_date"),
                    "temp_2m": prop.get("temp_2m"),
                    "soil_temp_l1": prop.get("soil_temp_l1"),
                    "soil_moist_l1": prop.get("soil_moist_l1"),
                    "snow_cover": prop.get("snow_cover"),
                    "precip": prop.get("precip"),
                    "pet": prop.get("pet"),
                    "evap": prop.get("evap"),
                })

            # checkpoint every week (safe if kernel dies)
            df = pd.DataFrame(rows)
            df.to_parquet(OUT_PATH)
            print(f"      💾 checkpoint rows: {len(df):,}")

    print("\n✅ M10a complete")
    print("📦 Written:", OUT_PATH)


if __name__ == "__main__":
    main()