#!/usr/bin/env python3
"""
fetch_and_merge_inat.py

Ein einziger, sauberer, projektweiter Workflow:

1) iNat-Daten für:
   - target species
   - contrast species
   gemäß defaults.region + defaults.species + period

2) Robust cleaned → DataFrame

3) Zusammenführen + Labeln:
   1 = target species
   0 = contrast species

4) Speichern unter:
   <output_dir>/inat_merged_labeled.csv
"""

import sys
import os
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# Projektwurzel laden
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import init as bootstrap_init


# ==========================================================
# 1) Fetch-Funktionen
# ==========================================================
def fetch_inat_species(cfg, taxon_id, species_name, bbox, d1, d2, max_pages=50):
    """Lädt iNat-Beobachtungen einer Art → DataFrame."""

    base_url = "https://api.inaturalist.org/v1/observations"
    all_results = []

    params_base = {
        "taxon_id": taxon_id,
        "quality_grade": cfg["inat"]["quality_grade"],
        "acc_below_or_equal": cfg["inat"]["max_accuracy"],
        "nelat": bbox[3], "nelng": bbox[2],
        "swlat": bbox[1], "swlng": bbox[0],
        "d1": d1, "d2": d2,
        "per_page": 200,
        "order": "desc",
        "order_by": "observed_on"
    }

    print(f"\n🔍 Lade {species_name} (taxon {taxon_id})")

    for page in range(1, max_pages + 1):
        params = params_base.copy()
        params["page"] = page

        resp = requests.get(base_url, params=params, timeout=30)

        if resp.status_code == 429:
            print("⏳ Rate-limit – 30s warten…")
            time.sleep(30)
            continue

        if resp.status_code != 200:
            print(f"⚠️ API Fehler {resp.status_code}")
            break

        results = resp.json().get("results", [])
        if not results:
            break

        all_results.extend(results)
        time.sleep(0.6)

    print(f"   → {len(all_results)} Beobachtungen geladen.")
    return parse_results(all_results, species_name)


def parse_results(results, species_name):
    """iNat JSON → DataFrame mit robustem Datumsparsing."""
    records = []

    for obs in results:
        coords = obs.get("geojson", {}).get("coordinates", [None, None])
        lon, lat = coords[0], coords[1]

        observed_on = obs.get("observed_on")

        try:
            dt = datetime.fromisoformat(observed_on.replace("Z", ""))
        except:
            dt = None

        records.append({
            "species": species_name,
            "taxon_id": obs.get("taxon", {}).get("id"),
            "latitude": lat,
            "longitude": lon,
            "date": observed_on,
            "year": dt.year if dt else None,
            "month": dt.month if dt else None,
        })

    return pd.DataFrame(records)


# ==========================================================
# 2) Merging-Funktion
# ==========================================================
def build_merged(df_target, df_contrast, cfg):
    """Labelt + merged beide Arten."""
    df_target["label"] = 1
    df_contrast["label"] = 0

    cols = ["species", "label", "taxon_id", "latitude", "longitude", "date", "year", "month"]

    df = pd.concat([df_target[cols], df_contrast[cols]], ignore_index=True)
    df = df.dropna(subset=["latitude", "longitude", "year", "month"])
    df = df.sort_values(["year", "month"])
    return df


# ==========================================================
# 3) Main Workflow
# ==========================================================
def main():
    cfg = bootstrap_init(verbose=False)

    # welche Arten verwenden?
    sp_target_key = cfg["defaults"]["target_species"]
    sp_contrast_key = cfg["defaults"]["contrast_species"]

    target = cfg["species"][sp_target_key]
    contrast = cfg["species"][sp_contrast_key]

    period = cfg["inat"]["period"]
    bbox = cfg["region"]["bbox_wgs84"]

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📅 Zeitraum: {period['start']} → {period['end']}")
    print(f"🌍 Region: {cfg['defaults']['region']}  BBox={bbox}")
    print(f"🎯 Zielart:     {target['name']} (id={target['id']})")
    print(f"🪶 Kontrastart: {contrast['name']} (id={contrast['id']})")

    # --- Fetch target ---
    df_t = fetch_inat_species(
        cfg,
        taxon_id=target["id"],
        species_name=target["name"],
        bbox=bbox,
        d1=period["start"],
        d2=period["end"],
        max_pages=cfg["inat"]["max_pages"],
    )

    # --- Fetch contrast ---
    df_c = fetch_inat_species(
        cfg,
        taxon_id=contrast["id"],
        species_name=contrast["name"],
        bbox=bbox,
        d1=period["start"],
        d2=period["end"],
        max_pages=cfg["inat"]["max_pages"],
    )

    # RAW speichern
    df_t.to_csv(out_dir / f"inat_{target['name']}.csv", index=False)
    df_c.to_csv(out_dir / f"inat_{contrast['name']}.csv", index=False)


    # --- Merge ---
    df_all = build_merged(df_t, df_c, cfg)

    # dynamischer Dateiname basierend auf species
    tname = cfg["inat"]["target"]["name"].replace(" ", "_")
    cname = cfg["inat"]["contrast"]["name"].replace(" ", "_")
    merged_filename = f"inat_merged_{tname}_vs_{cname}.csv"

    merged_path = out_dir / merged_filename
    df_all.to_csv(merged_path, index=False)

    print("\n🎉 Fertig!")
    print(f"💾 Gespeichert: {merged_path}")
    print(f"🔢 Gesamt: {len[df_all]} Beobachtungen\n")


if __name__ == "__main__":
    main()