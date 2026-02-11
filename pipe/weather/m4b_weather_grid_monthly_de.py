import ee
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --------------------------------------------------
# Init
# --------------------------------------------------
ee.Initialize()

GRID_PATH = Path("/Volumes/Data/iNaturalist/weather/grid/grid_20km_DE.gpkg")
OUT_PATH  = Path("/Volumes/Data/iNaturalist/weather/derived/weather_grid_year_monthly.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

YEARS  = range(2017, 2025)
MONTHS = range(1, 13)

# --------------------------------------------------
# Load grid
# --------------------------------------------------
grid = gpd.read_file(GRID_PATH)[["grid_id", "geometry"]]

# --------------------------------------------------
# ERA5-Land MONTHLY_AGGR
# --------------------------------------------------
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")

def month_image(year, month):
    return (
        era5
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .filter(ee.Filter.calendarRange(month, month, "month"))
        .first()
    )

# --------------------------------------------------
# Main
# --------------------------------------------------
rows = []

for year in YEARS:
    print(f"🌦 Processing year {year}")

    for _, cell in grid.iterrows():

        minx, miny, maxx, maxy = cell.geometry.bounds

        geom = (
            ee.Geometry.Rectangle(
                [minx, miny, maxx, maxy],
                proj="EPSG:3035",
                geodesic=False
            )
            .transform("EPSG:4326", 1)
        )

        row = {
            "grid_id": cell.grid_id,
            "year": year,
        }

        for m in MONTHS:
            img = month_image(year, m)

            def safe_reduce(band):
                val = img.select(band).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geom,
                    scale=10000,
                    maxPixels=1e13
                ).get(band)
                return ee.Algorithms.If(val, val, None)

            temp_k = safe_reduce("temperature_2m")
            precip = safe_reduce("total_precipitation_sum")

            row[f"temp_m{m:02d}"] = ee.Algorithms.If(
                temp_k,
                ee.Number(temp_k).subtract(273.15),
                None
            )

            row[f"precip_m{m:02d}"] = ee.Algorithms.If(
                precip,
                ee.Number(precip).multiply(1000),
                None
            )

        rows.append(ee.Dictionary(row).getInfo())

    # checkpoint
    pd.DataFrame(rows).to_parquet(OUT_PATH)
    print(f"💾 checkpoint written ({len(rows)} rows)")

print("✅ M4b complete (monthly)")