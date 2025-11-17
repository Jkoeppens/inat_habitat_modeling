#!/usr/bin/env python3
"""
Check species observations in region defined by defaults.region.

Usage:
    python explore/check_species_observations.py --name "Amanita muscaria"
"""

import sys
import requests
from pathlib import Path
import argparse

# ---------------------------------------------------------
# 1. Projektwurzel erkennen & bootstrap importieren
# ---------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import init as bootstrap_init

# ---------------------------------------------------------
# 2. iNaturalist API Endpoints
# ---------------------------------------------------------
TAXON_URL = "https://api.inaturalist.org/v1/taxa"
OBS_URL = "https://api.inaturalist.org/v1/observations"


# ---------------------------------------------------------
# Suche taxon_id per scientific name
# ---------------------------------------------------------
def get_taxon_id(scientific_name: str):
    params = {
        "q": scientific_name,
        "rank": "species",
        "iconic_taxa": "Fungi",      # optional: einschränken auf Pilze
        "per_page": 5,
    }

    r = requests.get(TAXON_URL, params=params)
    r.raise_for_status()

    for t in r.json().get("results", []):
        if t["rank"] == "species" and t["name"].lower() == scientific_name.lower():
            return t["id"]

    return None


# ---------------------------------------------------------
# Zähle Beobachtungen in Bounding Box
# ---------------------------------------------------------
def count_observations_in_bbox(taxon_id: int, bbox: dict):
    params = {
        "taxon_id": taxon_id,
        "per_page": 1,
        **bbox,
    }

    r = requests.get(OBS_URL, params=params)
    r.raise_for_status()
    return r.json().get("total_results", 0)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Scientific name der Art, z.B. 'Amanita muscaria'"
    )
    args = parser.parse_args()

    # ---------------------------
    # Bootstrap laden
    # ---------------------------
    cfg = bootstrap_init(verbose=False)

    # Region laden
    region_name = cfg["defaults"]["region"]
    region_data = cfg["regions"][region_name]

    bbox = {
        "swlng": region_data["bbox_wgs84"][0],
        "swlat": region_data["bbox_wgs84"][1],
        "nelng": region_data["bbox_wgs84"][2],
        "nelat": region_data["bbox_wgs84"][3],
    }

    print(f"\n🌍 Region aus default: {region_name}")
    print(f"   BBOX = {bbox}\n")

    species = args.name
    print(f"🔍 Suche taxon_id für: {species}")

    taxon_id = get_taxon_id(species)
    if taxon_id is None:
        print("❌ taxon_id nicht gefunden!")
        return

    print(f"   ✔ taxon_id = {taxon_id}")

    # Beobachtungen zählen
    print(f"\n📊 Zähle Beobachtungen in {region_name}…")
    count = count_observations_in_bbox(taxon_id, bbox)

    print(f"\n🏆 Ergebnis:")
    print(f"   {species}: {count} Beobachtungen in Region '{region_name}'\n")


if __name__ == "__main__":
    main()