"""
M1 – Build 20 km grid for Germany

Creates a regular 20 km grid clipped to Germany
and stores it as a GeoPackage.

Output:
    /Volumes/Data/iNaturalist/weather/grid/grid_20km_DE.gpkg
"""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path("/Volumes/Data/iNaturalist/weather")
GRID_DIR = BASE_DIR / "grid"
GRID_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = GRID_DIR / "grid_20km_DE.gpkg"


# --------------------------------------------------
# Parameters
# --------------------------------------------------
GRID_SIZE_M = 20_000  # 20 km
TARGET_CRS = "EPSG:3035"
GRID_PREFIX = "DE20"


# --------------------------------------------------
# Load Germany geometry
# --------------------------------------------------
def load_germany():
    world = gpd.read_file(
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
    )
    germany = world[world["NAME"] == "Germany"]
    germany = germany.to_crs(TARGET_CRS)
    return germany


# --------------------------------------------------
# Build grid
# --------------------------------------------------
def build_grid(geometry, cell_size):
    minx, miny, maxx, maxy = geometry.total_bounds

    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    cells = []
    for x in xs:
        for y in ys:
            cells.append(
                box(x, y, x + cell_size, y + cell_size)
            )

    grid = gpd.GeoDataFrame(
        {"geometry": cells},
        crs=TARGET_CRS,
    )

    # Clip to Germany
    grid = gpd.clip(grid, geometry)

    return grid


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("🌍 Loading Germany geometry …")
    germany = load_germany()

    print("🔲 Building 20 km grid …")
    grid = build_grid(germany, GRID_SIZE_M)

    print("🆔 Assigning grid IDs …")
    grid = grid.reset_index(drop=True)
    grid["grid_id"] = [
        f"{GRID_PREFIX}_{i:06d}" for i in range(len(grid))
    ]

    grid = grid[["grid_id", "geometry"]]

    print(f"💾 Writing grid to {OUT_FILE}")
    grid.to_file(OUT_FILE, driver="GPKG")

    print(f"✅ Done. Cells: {len(grid)}")


if __name__ == "__main__":
    main()