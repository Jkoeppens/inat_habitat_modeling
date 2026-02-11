import ee
import pandas as pd
import numpy as np

# --------------------------------------------------
# Init Earth Engine
# --------------------------------------------------
ee.Initialize()

# --------------------------------------------------
# Germany geometry
# --------------------------------------------------
region = (
    ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    .filter(ee.Filter.eq("country_na", "Germany"))
    .geometry()
)

# --------------------------------------------------
# ERA5-Land MONTHLY_AGGR (NON-deprecated)
# --------------------------------------------------
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def get_month_image(year, month):
    col = (
        era5
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .filter(ee.Filter.calendarRange(month, month, "month"))
    )
    return ee.Image(col.first())

def reduce_safe(img, band, reducer):
    """
    Returns ee.Number or None (Python-side later)
    """
    return ee.Algorithms.If(
        img,
        img.select(band).reduceRegion(
            reducer=reducer,
            geometry=region,
            scale=10000,
            maxPixels=1e13
        ).get(band),
        None
    )

def aggregate_period(year, months):
    temp_sum = ee.Number(0)
    temp_n   = ee.Number(0)
    prec_sum = ee.Number(0)

    for m in months:
        img = get_month_image(year, m)

        t = reduce_safe(img, "temperature_2m", ee.Reducer.mean())
        p = reduce_safe(img, "total_precipitation_sum", ee.Reducer.mean())

        temp_sum = ee.Number(
            ee.Algorithms.If(t, temp_sum.add(t), temp_sum)
        )
        temp_n = ee.Number(
            ee.Algorithms.If(t, temp_n.add(1), temp_n)
        )
        prec_sum = ee.Number(
            ee.Algorithms.If(p, prec_sum.add(p), prec_sum)
        )

    temp_mean = ee.Algorithms.If(
        temp_n.gt(0),
        temp_sum.divide(temp_n),
        None
    )

    return {
        "temp": temp_mean,
        "precip": prec_sum
    }

# --------------------------------------------------
# Years to process
# --------------------------------------------------
years = list(range(2017, 2025))

# --------------------------------------------------
# Build weather table
# --------------------------------------------------
rows = []

for y in years:
    print(f"⛅ Processing year {y}")

    ann = aggregate_period(y, range(1, 13))
    mj  = aggregate_period(y, [5, 6])
    ja  = aggregate_period(y, [7, 8])
    so  = aggregate_period(y, [9, 10])

    ee_row = ee.Dictionary({
        "year": y,
        "temp_ann": ann["temp"],
        "precip_ann": ann["precip"],
        "temp_may_jun": mj["temp"],
        "precip_may_jun": mj["precip"],
        "temp_jul_aug": ja["temp"],
        "precip_jul_aug": ja["precip"],
        "temp_sep_oct": so["temp"],
        "precip_sep_oct": so["precip"],
    })

    row = ee_row.getInfo()
    rows.append(row)

# --------------------------------------------------
# Final DataFrame
# --------------------------------------------------
weather_df = (
    pd.DataFrame(rows)
    .set_index("year")
    .astype(float)
)

weather_df