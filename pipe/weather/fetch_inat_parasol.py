"""
Fetch iNaturalist observations for Macrolepiota procera (Parasol mushroom)
Germany, 2017–2024

Output:
    /Volumes/Data/iNaturalist/weather/inat/parasol/observations_YYYY.parquet
"""

from pathlib import Path
import requests
import pandas as pd
import time


# ==================================================
# CONFIG
# ==================================================
BASE_DIR = Path("/Volumes/Data/iNaturalist/weather")
OUT_DIR = BASE_DIR / "inat" / "parasol"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://api.inaturalist.org/v1/observations"

YEARS = range(2017, 2025)
TAXON_ID = 63401  # Macrolepiota procera
PER_PAGE = 200

# Germany bbox (WGS84)
BBOX = {
    "nelat": 55.1,
    "nelng": 15.5,
    "swlat": 47.2,
    "swlng": 5.8,
}

HEADERS = {
    "User-Agent": "inat-habitat-modeling (academic research)"
}


# ==================================================
# FETCH HELPERS
# ==================================================
def fetch_page(params):
    r = requests.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=60,
    )

    if r.status_code in (401, 403, 429):
        print(f"⛔ Rate limit ({r.status_code}) – sleeping 60s")
        time.sleep(60)
        return None

    r.raise_for_status()
    return r.json()


def fetch_year(year):
    print(f"\n⬇ Fetching parasol observations for {year}")

    rows = []
    page = 1

    while True:
        params = {
            "taxon_id": TAXON_ID,
            "quality_grade": "research",
            "d1": f"{year}-01-01",
            "d2": f"{year}-12-31",
            "per_page": PER_PAGE,
            "page": page,
            **BBOX,
        }

        data = fetch_page(params)
        if data is None:
            continue

        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            geo = obs.get("geojson")
            if not geo:
                continue

            coords = geo["coordinates"]
            observed_on = obs.get("observed_on")

            rows.append(
                {
                    "obs_id": obs.get("id"),
                    "observed_on": observed_on,
                    "year": year,
                    "month": int(observed_on.split("-")[1]) if observed_on else None,
                    "lon": coords[0],
                    "lat": coords[1],
                    "pos_accuracy_m": obs.get("positional_accuracy"),
                    "quality_grade": obs.get("quality_grade"),
                }
            )

        if len(results) < PER_PAGE:
            break

        page += 1
        time.sleep(1.0)

    df = pd.DataFrame(rows)
    out_file = OUT_DIR / f"observations_{year}.parquet"
    df.to_parquet(out_file, index=False)

    print(f"✔ {year}: {len(df)} parasol observations")


# ==================================================
# MAIN
# ==================================================
def main():
    print("=== iNaturalist Parasol fetch (Germany) ===")

    for year in YEARS:
        out_file = OUT_DIR / f"observations_{year}.parquet"
        if out_file.exists():
            print(f"✔ {year} exists, skipping")
            continue

        fetch_year(year)

    print("\n✅ Done")


if __name__ == "__main__":
    main()