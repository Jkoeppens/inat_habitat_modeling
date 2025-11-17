#!/usr/bin/env python3
"""
fetch_top_edible_fungi_region.py

→ Lädt essbare Pilze ("edible_fungi_table.csv") von EXTERNER PLATTE.
→ Nutzt Bootstrap-Config → lädt Region-BBOX automatisch.
→ Sucht pro Art die taxon_id.
→ Zählt Beobachtungen in der Region (BBox).
→ Speichert Ergebnis ebenfalls auf EXTERNER PLATTE.

"""
#!/usr/bin/env python3
import sys
from pathlib import Path

# ============================================================
# Projektwurzel automatisch erkennen
# (= Ordner, der bootstrap.py enthält)
# ============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]        # inat_habitat_modeling/
BOOTSTRAP_FILE = PROJECT_ROOT / "bootstrap.py"

if not BOOTSTRAP_FILE.exists():
    raise RuntimeError(f"bootstrap.py nicht gefunden unter {BOOTSTRAP_FILE}")

# Projektwurzel zum Python-Pfad hinzufügen
sys.path.insert(0, str(PROJECT_ROOT))

# Jetzt ist der Import sicher!
from bootstrap import init as bootstrap_init
import pandas as pd
import requests
import time
from pathlib import Path
from bootstrap import init as bootstrap_init


# ============================================================
# 1. Speicherorte NUR auf externer Platte
# ============================================================

BASE_OUT = Path("/Volumes/Data/iNaturalist/outputs")

INPUT_TABLE = BASE_OUT / "fungi_raw.csv"
OUTPUT_TABLE = BASE_OUT / "top_edible_fungi_region.csv"


# ============================================================
# 2. API URLs
# ============================================================

OBS_URL = "https://api.inaturalist.org/v1/observations"
TAXON_URL = "https://api.inaturalist.org/v1/taxa"


# ============================================================
# 3. API Helper
# ============================================================

def get_taxon_id(scientific_name: str):
    """Suche taxon_id auf iNaturalist."""
    params = {
        "q": scientific_name,
        "rank": "species",
        "iconic_taxa": "Fungi",
        "per_page": 5,
    }

    r = requests.get(TAXON_URL, params=params)
    if r.status_code != 200:
        return None

    for t in r.json().get("results", []):
        if t["rank"] == "species" and t["name"].lower() == scientific_name.lower():
            return t["id"]

    return None


def count_observations_in_bbox(taxon_id: int, bbox: dict):
    """Zähle Beobachtungen im Bounding Box der Region aus der Config."""
    params = {
        "taxon_id": taxon_id,
        **bbox,
        "per_page": 1,   # Wir nutzen nur total_results
    }

    r = requests.get(OBS_URL, params=params)
    if r.status_code != 200:
        return 0

    return r.json().get("total_results", 0)


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("=============================================")
    print("🔧 Lade Region aus Bootstrap-Konfiguration…")
    print("=============================================")

    cfg = bootstrap_init(verbose=False)

    # Die Region enthält bereits swlng/swlat/nelng/nelat
    bbox = cfg["region"]["bbox_wgs84"]
    region_name = cfg["defaults"]["region"]

    print(f"🌍 Region: {region_name}")
    print(f"📦 bbox: {bbox}")

    # ---------------------------------------------------------
    # Eingabedatei prüfen
    # ---------------------------------------------------------
    if not INPUT_TABLE.exists():
        raise FileNotFoundError(
            f"❌ edible_fungi_table.csv nicht gefunden!\n"
            f"   Erwarteter Pfad: {INPUT_TABLE}"
        )

    print(f"\n📥 Lade essbare Pilzliste aus: {INPUT_TABLE}")
    df = pd.read_csv(INPUT_TABLE)

    if "Scientific name" not in df.columns:
        raise ValueError("❌ 'Scientific name' fehlt in der Tabelle!")

    fungi = df["Scientific name"].dropna().unique().tolist()
    print(f"🍄 Anzahl essbarer Arten: {len(fungi)}")

    # ---------------------------------------------------------
    # Ergebnisse sammeln
    # ---------------------------------------------------------
    results = []

    for name in fungi:
        print(f"\n🔍 taxon_id suchen: {name}")
        tid = get_taxon_id(name)

        if tid is None:
            print("   ⚠️ Keine taxon_id gefunden — übersprungen.")
            continue

        print(f"   ✔ taxon_id = {tid}")
        print("   → Beobachtungen in der Region zählen…")

        count = count_observations_in_bbox(tid, bbox)
        print(f"   ➝ Beobachtungen: {count}")

        results.append({
            "region": region_name,
            "scientific_name": name,
            "taxon_id": tid,
            "observations": count
        })

        time.sleep(0.3)  # freundlich zur API

    # ---------------------------------------------------------
    # Speichern
    # ---------------------------------------------------------
    out_df = (
        pd.DataFrame(results)
        .sort_values("observations", ascending=False)
    )

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_TABLE, index=False)

    print("\n💾 Ergebnis gespeichert unter:")
    print("   ", OUTPUT_TABLE)

    print("\n🏆 TOP 15 Arten:")
    print(out_df.head(15))


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()