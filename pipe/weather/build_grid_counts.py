"""
M3 – Build grid-level observation counts from GBIF (iNaturalist subset)

Input:
    - GBIF CSV (Germany, iNaturalist, 2017–2024)
    - 20 km Germany grid (EPSG:3035)

Output:
    - grid_counts.parquet:
        grid_id | year | month | n_parasol | n_controls | n_total
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sys

# --------------------------------------------------
# Config
# --------------------------------------------------
BASE_DIR = Path("/Volumes/Data/iNaturalist/weather")

GBIF_CSV = BASE_DIR / "gbif" / "raw" / "0017039-260108223611665.csv"
GRID_PATH = BASE_DIR / "grid" / "grid_20km_DE.gpkg"
OUT_PATH = BASE_DIR / "inat" / "processed" / "grid_counts.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET_SPECIES = "Macrolepiota procera"
CHUNKSIZE = 200_000

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    print("🌍 Loading grid …")
    grid = gpd.read_file(GRID_PATH)[["grid_id", "geometry"]]
    grid = grid.to_crs("EPSG:3035")

    print(f"🔲 Grid cells: {len(grid)}")
    print("📦 Starting GBIF stream …")

    agg_chunks = []

    usecols = [
        "decimalLatitude",
        "decimalLongitude",
        "year",
        "month",
        "species",
    ]

    for i, chunk in enumerate(
        pd.read_csv(
            GBIF_CSV,
            sep="\t",
            usecols=usecols,
            chunksize=CHUNKSIZE,
            low_memory=False,
        ),
        start=1,
    ):
        print(f"▶ Chunk {i}")

        # ------------------------------------------
        # Basic filtering
        # ------------------------------------------
        chunk = chunk.dropna(
            subset=["decimalLatitude", "decimalLongitude", "year", "month"]
        )

        if chunk.empty:
            continue

        # ------------------------------------------
        # Build GeoDataFrame
        # ------------------------------------------
        gdf = gpd.GeoDataFrame(
            chunk,
            geometry=[
                Point(xy)
                for xy in zip(
                    chunk["decimalLongitude"],
                    chunk["decimalLatitude"],
                )
            ],
            crs="EPSG:4326",
        ).to_crs("EPSG:3035")

        # ------------------------------------------
        # Spatial join → grid
        # ------------------------------------------
        joined = gpd.sjoin(
            gdf,
            grid,
            how="inner",
            predicate="within",
        )

        if joined.empty:
            continue

        # ------------------------------------------
        # Feature engineering
        # ------------------------------------------
        joined["is_parasol"] = joined["species"] == TARGET_SPECIES

        # ------------------------------------------
        # Aggregate
        # ------------------------------------------
        grouped = (
            joined.groupby(["grid_id", "year", "month"])
            .agg(
                n_parasol=("is_parasol", "sum"),
                n_total=("is_parasol", "size"),
            )
            .reset_index()
        )

        grouped["n_controls"] = grouped["n_total"] - grouped["n_parasol"]

        agg_chunks.append(grouped)

    if not agg_chunks:
        raise RuntimeError("❌ No data aggregated – check inputs.")

    print("🧮 Combining chunks …")
    df = pd.concat(agg_chunks, ignore_index=True)

    df = (
        df.groupby(["grid_id", "year", "month"], as_index=False)
        .sum()
        .sort_values(["year", "month", "grid_id"])
    )

    print(f"💾 Writing {OUT_PATH}")
    df.to_parquet(OUT_PATH, index=False)

    print("✅ M3 complete")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()