#!/usr/bin/env python3
"""
fetch_inat.py

Lädt iNaturalist-Beobachtungen gemäß YAML-Konfiguration.

Unterstützt:
- target_species = konkrete Art
- contrast_species = konkrete Art ODER "all" (Background)

Background wird:
- zeitlich gleichverteilt gesampelt
- auf max_total begrenzt
"""

import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# ------------------------------------------------------------
# Projekt & Bootstrap
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import init as bootstrap_init

BASE_URL = "https://api.inaturalist.org/v1/observations"


# ============================================================
# Hilfsfunktionen
# ============================================================
def _parse_results(results, species_name, label):
    rows = []

    for obs in results:
        coords = obs.get("geojson", {}).get("coordinates", [None, None])
        lon, lat = coords
        date_raw = obs.get("observed_on")

        try:
            dt = datetime.fromisoformat(date_raw)
        except Exception:
            dt = None

        rows.append({
            "species": species_name,
            "label": label,
            "taxon_id": obs.get("taxon", {}).get("id"),
            "latitude": lat,
            "longitude": lon,
            "date": date_raw,
            "year": dt.year if dt else None,
            "month": dt.month if dt else None,
        })

    return pd.DataFrame(rows)


def _inat_request(cfg, extra_params, max_pages):
    all_results = []

    params_base = {
        "quality_grade": cfg["inat"]["quality_grade"],
        "acc_below_or_equal": cfg["inat"]["max_accuracy"],
        "per_page": 200,
        "order": "desc",
        "order_by": "observed_on",
        **extra_params,
    }

    for page in range(1, max_pages + 1):
        params = params_base.copy()
        params["page"] = page

        r = requests.get(BASE_URL, params=params, timeout=30)

        if r.status_code == 429:
            time.sleep(30)
            continue
        if r.status_code != 200:
            break

        results = r.json().get("results", [])
        if not results:
            break

        all_results.extend(results)
        time.sleep(0.6)

    return all_results


# ============================================================
# Fetch-Funktionen
# ============================================================
def fetch_species(cfg, taxon_id, name, bbox, d1, d2):
    print(f"\n🔍 Lade Target: {name}")

    results = _inat_request(
        cfg,
        {
            "taxon_id": taxon_id,
            "nelat": bbox[3], "nelng": bbox[2],
            "swlat": bbox[1], "swlng": bbox[0],
            "d1": d1, "d2": d2,
        },
        cfg["inat"]["max_pages"],
    )

    return _parse_results(results, name, label=1)


def fetch_background(cfg, bbox, d1, d2, max_total=10000):
    print(f"\n🔍 Lade Background (all species, max={max_total})")

    results = _inat_request(
        cfg,
        {
            "nelat": bbox[3], "nelng": bbox[2],
            "swlat": bbox[1], "swlng": bbox[0],
            "d1": d1, "d2": d2,
        },
        cfg["inat"]["max_pages"],
    )

    df = _parse_results(results, "background", label=0)

    # ----------------------------------------
    # ⚖️ Zeitlich gleichverteilt samplen
    # ----------------------------------------
    df = df.dropna(subset=["year"])
    years = sorted(df["year"].unique())

    per_year = max_total // len(years)
    sampled = []

    for y in years:
        chunk = df[df["year"] == y]
        sampled.append(chunk.sample(min(per_year, len(chunk)), random_state=42))

    df = pd.concat(sampled).reset_index(drop=True)
    print(f"   → nach Sampling: {len(df)} Beobachtungen")

    return df


# ============================================================
# Zentrale Entry-Funktion
# ============================================================
def fetch_target_and_contrast(cfg=None):
    if cfg is None:
        cfg = bootstrap_init(verbose=True)

    region = cfg["defaults"]["region"]
    bbox = cfg["region"]["bbox_wgs84"]
    d1 = cfg["inat"]["period"]["start"]
    d2 = cfg["inat"]["period"]["end"]

    tkey = cfg["defaults"]["target_species"]
    ckey = cfg["defaults"]["contrast_species"]

    target = cfg["species"][tkey]
    df_target = fetch_species(cfg, target["id"], target["name"], bbox, d1, d2)

    if ckey == "all":
        df_contrast = fetch_background(
            cfg,
            bbox=bbox,
            d1=d1,
            d2=d2,
            max_total=cfg["inat"].get("background_max_total", 10000),
        )
        cname = "background"
    else:
        contrast = cfg["species"][ckey]
        df_contrast = fetch_species(cfg, contrast["id"], contrast["name"], bbox, d1, d2)
        df_contrast["label"] = 0
        cname = contrast["name"]

    # ----------------------------------------
    # 📊 Reporting
    # ----------------------------------------
    print("\n📊 Zusammenfassung:")
    print(f"Target ({target['name']}): {len(df_target)}")
    print(f"Contrast ({cname}): {len(df_contrast)}")
    print("Verhältnis:", len(df_target) / len(df_contrast))

    print("\n🗓️ Jahresverteilung (Target):")
    print(df_target["year"].value_counts().sort_index())

    print("\n🗓️ Jahresverteilung (Contrast):")
    print(df_contrast["year"].value_counts().sort_index())

    return df_target, df_contrast