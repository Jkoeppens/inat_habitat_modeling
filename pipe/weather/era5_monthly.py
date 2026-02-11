# ============================================================
# ERA5-Land Monthly Weather Export (GEE)
# ============================================================

import ee


# ------------------------------------------------------------
# Build yearly ERA5-Land image with monthly bands
# ------------------------------------------------------------
def build_yearly_weather_image(year):
    """
    Returns an ee.Image with 24 bands:
    precip_m01, temp_m01, ..., precip_m12, temp_m12
    Units:
      - precip: mm / month
      - temp: °C
    """

    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY")
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .select(["total_precipitation", "temperature_2m"])
    )

    def per_month(m):
        m = ee.Number(m)
        m_str = m.format("%02d")

        img = collection.filter(
            ee.Filter.calendarRange(m, m, "month")
        ).first()

        precip = (
            img.select("total_precipitation")
            .multiply(1000)  # m → mm
            .rename(ee.String("precip_m").cat(m_str))
        )

        temp = (
            img.select("temperature_2m")
            .subtract(273.15)  # K → °C
            .rename(ee.String("temp_m").cat(m_str))
        )

        return precip.addBands(temp)

    img = ee.ImageCollection(
        ee.List.sequence(1, 12).map(per_month)
    ).toBands()

    # --------------------------------------------------------
    # remove numeric prefixes added by toBands()
    # --------------------------------------------------------
    band_names = img.bandNames()

    def strip_prefix(name):
        name = ee.String(name)
        return name.split("_").slice(1).join("_")

    clean_names = band_names.map(strip_prefix)

    return img.rename(clean_names)


# ------------------------------------------------------------
# Export helpers
# ------------------------------------------------------------
def export_year_to_drive(
    year,
    region,
    out_name,
    folder="ERA5_LAND_WEATHER",
    scale=9000,
    crs="EPSG:4326",
):
    """
    Export one year to Google Drive as GeoTIFF.
    """

    img = build_yearly_weather_image(year).clip(region)

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=out_name,
        folder=folder,
        fileNamePrefix=out_name,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )

    task.start()
    return task


def export_batch_to_drive(
    years,
    region,
    name_prefix="weather",
    folder="ERA5_LAND_WEATHER",
    scale=9000,
    crs="EPSG:4326",
):
    """
    Export multiple years in a loop.
    """
    tasks = []
    for year in years:
        name = f"{name_prefix}_{year}"
        task = export_year_to_drive(
            year,
            region,
            out_name=name,
            folder=folder,
            scale=scale,
            crs=crs,
        )
        tasks.append(task)
        print(f"⬆️ started export: {name}")
    return tasks