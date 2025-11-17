#!/usr/bin/env python3
"""
explore_inat_fungi.py

Erster Exploration-Schritt für iNaturalist-Daten:
 - fetch: Beobachtungen (Fungi) laden
 - inspect: erste Statistiken + Tabellen
 - plot: Explorationsplots speichern
 - save: Rohdaten + bereinigte Daten exportieren

Nutzung:
    python explore_inat_fungi.py --fetch --limit 500
    python explore_inat_fungi.py --inspect
    python explore_inat_fungi.py --plot
    python explore_inat_fungi.py --all --limit 1000
"""
import numpy as np
import argparse
import requests
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Pfade
# -------------------------------------------------------
DATA_DIR = Path("data/inat_raw")
OUT_DIR = Path("data/inat_explore")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# Fetch aus iNaturalist
# -------------------------------------------------------
def fetch_inat_fungi(limit=500):
    print(f"📥 Lade iNaturalist (Fungi)… (limit={limit})")

    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_name": "Fungi",
        "quality_grade": "research",
        "per_page": 200,
        "order_by": "created_at",
    }

    pages = (limit // 200) + 1
    all_obs = []

    for p in range(1, pages + 1):
        print(f"   → Seite {p}/{pages}")
        params["page"] = p
        r = requests.get(url, params=params)
        r.raise_for_status()

        data = r.json()["results"]
        all_obs.extend(data)

        if len(all_obs) >= limit:
            break

    print(f"✔ Beobachtungen geladen: {len(all_obs)}")

    df = pd.json_normalize(all_obs)
    df.to_csv(DATA_DIR / "fungi_raw.csv", index=False)
    print("💾 Gespeichert: data/inat_raw/fungi_raw.csv")

    return df


# -------------------------------------------------------
# Bereinigen
# -------------------------------------------------------
import numpy as np

def clean(df):
    """
    Minimal-Cleaner für Exploration:
    - benutzt KEINE Koordinaten
    - ignoriert fehlende Spalten
    - extrahiert nur taxonomische Infos
    """

    # Relevante Spalten (koordinatenfrei!)
    keep = [
        "id",
        "observed_on",
        "quality_grade",
        "taxon.name",
        "taxon.rank",
        "taxon.preferred_common_name",
        "taxon.iconic_taxon_name",
        "user.login",
    ]

    missing = [c for c in keep if c not in df.columns]
    if missing:
        print(f"⚠️ Warnung: Fehlende Spalten im API-Result: {set(missing)}")
        # fehlen → füllen mit NaN
        for col in missing:
            df[col] = np.nan

    df = df[keep].copy()

    # Aufräumen
    df["taxon.name"] = df["taxon.name"].astype(str).str.strip()
    df["taxon.preferred_common_name"] = (
        df["taxon.preferred_common_name"]
        .astype(str)
        .str.replace(r"\[.*?\]", "", regex=True)
        .str.strip()
    )

    return df


# -------------------------------------------------------
# Inspect
# -------------------------------------------------------
def inspect(df):
    print("\n📊 Grundlegende Statistik")
    print("Anzahl Beobachtungen:", len(df))
    print("Eindeutige Arten:", df["scientific_name"].nunique())

    print("\n🔝 Top 10 Arten")
    print(df["scientific_name"].value_counts().head(10))

    print("\n📍 Missing Geo:", df["lat"].isna().sum(), "/", len(df))


# -------------------------------------------------------
# Plots erzeugen
# -------------------------------------------------------
def make_plots(df):
    print("📈 Erzeuge Plots…")

    # 1) Artenverteilung (Top 15)
    plt.figure(figsize=(8, 5))
    df["scientific_name"].value_counts().head(15).plot.bar()
    plt.title("Häufigste Pilzarten (Top 15)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "species_top15.png", dpi=200)
    plt.close()

    # 2) Monatsverlauf
    df["month"] = pd.to_datetime(df["observed_on"], errors="coerce").dt.month
    plt.figure(figsize=(7, 4))
    df["month"].dropna().value_counts().sort_index().plot.bar()
    plt.title("Saisonale Verteilung")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "monthly_distribution.png", dpi=200)
    plt.close()

    # 3) Scatter: Geo
    plt.figure(figsize=(6, 6))
    plt.scatter(df["lon"], df["lat"], s=1)
    plt.title("Geografische Verteilung (Rohpunkte)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "geo_scatter.png", dpi=200)
    plt.close()

    print("✔ Plots gespeichert in: data/inat_explore/")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fetch", action="store_true", help="iNat-Daten abrufen")
    parser.add_argument("--inspect", action="store_true", help="Statistik erstellen")
    parser.add_argument("--plot", action="store_true", help="Plots erzeugen")
    parser.add_argument("--all", action="store_true", help="fetch + inspect + plot")
    parser.add_argument("--limit", type=int, default=500)

    args = parser.parse_args()

    # Falls nur Inspect/Plot: bestehende Daten laden
    df = None
    clean_df = None

    if args.fetch or args.all:
        df = fetch_inat_fungi(limit=args.limit)
        clean_df = clean(df)

    if args.inspect or args.all:
        if clean_df is None:
            clean_df = pd.read_csv(DATA_DIR / "fungi_clean.csv")
        inspect(clean_df)

    if args.plot or args.all:
        if clean_df is None:
            clean_df = pd.read_csv(DATA_DIR / "fungi_clean.csv")
        make_plots(clean_df)