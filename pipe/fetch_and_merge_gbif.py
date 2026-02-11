import pandas as pd
from pathlib import Path
from pyproj import Transformer
from bootstrap import init as bootstrap_init

# ----------------------------------------
# 🧪 Main
# ----------------------------------------
def main():
    cfg = bootstrap_init(verbose=True)

    # --- Konfiguration ---
    CSV_PATH = "/Volumes/Data/iNaturalist/gbif/0017039-260108223611665.csv"
    MAX_UNCERTAINTY_M = 50

    # --- Ziel & Kontrast ---
    tkey = cfg["defaults"]["target_species"]
    ckey = cfg["defaults"]["contrast_species"]
    target = cfg["species"][tkey]
    contrast = cfg["species"][ckey] if ckey in cfg["species"] else None

    region_key = cfg["defaults"]["region"]
    region = cfg["regions"][region_key]
    bbox = region["bbox_wgs84"]
    utm_crs = region["utm_crs"]
    period = cfg["inat"]["period"]

    print(f"\n🌍 Region: {region_key}  BBox={bbox}")
    print(f"🎯 Zielart: {target['name']} (taxonKey={target['id']})")
    if contrast:
        print(f"⚔️  Kontrastart: {contrast['name']} (taxonKey={contrast['id']})")
    else:
        print("⚔️  Kontrastart: Hintergrund (alle anderen Arten)")

    # --- Daten laden ---
    df = pd.read_csv(CSV_PATH, sep="\t", low_memory=False)

    # --- Zeit- & Raumfilter ---
    df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce")
    df = df[
        (df["eventDate"] >= period["start"]) &
        (df["eventDate"] <= period["end"]) &
        (df["decimalLatitude"].between(bbox[1], bbox[3])) &
        (df["decimalLongitude"].between(bbox[0], bbox[2]))
    ]

    # --- Genauigkeit ---
    if "coordinateUncertaintyInMeters" in df.columns:
        df = df[
            (df["coordinateUncertaintyInMeters"].isna()) |
            (df["coordinateUncertaintyInMeters"] <= MAX_UNCERTAINTY_M)
        ]

    # --- Zielart extrahieren ---
    df_target = df[df["taxonKey"] == target["id"]].copy()
    df_target["label"] = 1
    df_target["species"] = target["name"]

    # --- Kontrast: entweder echte Kontrast‑Art oder Background (alles andere) ---
    if contrast is not None and contrast.get("id") not in (None, ""):
        # echte Kontrast‑Art
        print(f"⚔️  Kontrastart: {contrast['name']} (taxonKey={contrast['id']})")
        df_contrast = df[df["taxonKey"] == contrast["id"]].copy()
        df_contrast["label"] = 0
        df_contrast["species"] = contrast["name"]
    else:
        # Hintergrund: alle anderen Arten außer Ziel
        print("⚙️  Kontrast='all' → Hintergrund aus Nicht‑Zielart erzeugen.")
        df_rest = df[df["taxonKey"] != target["id"]].copy()
        if len(df_rest) == 0:
            print("⚠️ Kein Hintergrund verfügbar nach Filter — df_rest ist leer!")
            df_contrast = df_rest  # bleibt leer
        else:
            max_bg = min(10000, 100 * len(df_target))
            df_contrast = df_rest.sample(n=min(max_bg, len(df_rest)), random_state=42)
            df_contrast["label"] = 0
            df_contrast["species"] = "background"

    # --- Vereinheitlichen ---
    def normalize(d):
        out = pd.DataFrame({
            "species": d["species"],
            "label": d["label"],
            "taxon_id": d["taxonKey"],
            "latitude": d["decimalLatitude"],
            "longitude": d["decimalLongitude"],
            "date": d["eventDate"],
        })
        out["year"] = out["date"].dt.year
        return out

    df_all = pd.concat([normalize(df_target), normalize(df_contrast)], ignore_index=True)
    df_all.dropna(subset=["latitude", "longitude", "year"], inplace=True)

    # --- UTM-Projektion ---
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x, y = transformer.transform(df_all["longitude"].values, df_all["latitude"].values)
    df_all["x_utm"] = x
    df_all["y_utm"] = y

    # --- Output ---
    tname = target["name"].replace(" ", "_").lower()
    cname = contrast["name"].replace(" ", "_").lower() if contrast else "background"
    species_out_dir = Path(cfg["paths"]["output_dir"]) / tname
    species_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = species_out_dir / f"inat_merged_{tname}_vs_{cname}.csv"

    df_all.to_csv(out_path, index=False)

    print(f"\n✅ Fertig")
    print(f"🔢 Beobachtungen gesamt: {len(df_all)}")
    print(f"💾 Gespeichert unter: {out_path}\n")


if __name__ == "__main__":
    main()