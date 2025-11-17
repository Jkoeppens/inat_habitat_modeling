#!/usr/bin/env python3
"""
diagnose_autocorr.py

Diagnose für Dateien vom Typ *_AUTOCORR.tif.
Unterstützt Region + Monate.
Erzeugt Histogramme und Clip-Previews.

Nutzung:
    python diagnose_autocorr.py --region berlin --month 08
    python diagnose_autocorr.py --month 10 --month 11
    python diagnose_autocorr.py --all
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Projektwurzel so bestimmen, dass bootstrap importierbar ist
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from bootstrap import init as bootstrap_init
except Exception as e:
    print("❌ Fehler: Konnte bootstrap.init nicht importieren!")
    print("PROJECT_ROOT =", ROOT)
    raise e


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def month_matches(fname: str, months):
    if months is None:
        return True
    return any(fname.endswith(f"_{m}_AUTOCORR.tif") for m in months)


def load_tiff(path: Path):
    with rasterio.open(path) as ds:
        data = ds.read()
        profile = ds.profile
    return data, profile


def plot_hist(values, title, outpath):
    plt.figure(figsize=(10, 4))
    plt.hist(values.flatten(), bins=200, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_clip(image, title, outpath):
    h, w = image.shape
    ch, cw = h // 2, w // 2
    crop = image[ch-1024:ch+1024, cw-1024:cw+1024]

    plt.figure(figsize=(6, 6))
    plt.imshow(crop, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ------------------------------------------------------------
# Diagnose
# ------------------------------------------------------------
def diagnose_file(path: Path, out_base: Path):
    fname = path.name
    print(f"\n🔍 Datei: {fname}")

    out_dir = out_base / fname.replace(".tif", "")
    out_dir.mkdir(exist_ok=True)

    data, profile = load_tiff(path)
    names = ["Moran NDVI", "Geary NDVI", "Moran NDWI", "Geary NDWI"]

    for i in range(4):
        band = data[i].astype(float)
        clean = band[np.isfinite(band)]

        if clean.size == 0:
            print(f"  ▸ {names[i]}: nur NaNs")
            continue

        print(f"  ▸ {names[i]}: min={clean.min():.4f}  max={clean.max():.4f}")

        plot_hist(clean, f"Histogram – {names[i]}",
                  out_dir / f"{names[i].replace(' ', '_')}_hist.png")

        plot_clip(band, f"Ausschnitt – {names[i]}",
                  out_dir / f"{names[i].replace(' ', '_')}_clip.png")

    print(f"   ✔ Diagnose gespeichert unter: {out_dir}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, required=False,
                        help="Region z. B. berlin")
    parser.add_argument("--month", action="append",
                        help="Monat(e): --month 08 --month 09")
    parser.add_argument("--all", action="store_true",
                        help="Alle Monate diagnostizieren")
    args = parser.parse_args()

    # --------------------------------------------------------
    # Konfiguration laden
    # --------------------------------------------------------
    cfg = bootstrap_init(verbose=False)

    # Region überschreiben, falls angegeben
    if args.region:
        if args.region not in cfg["regions"]:
            print(f"❌ Region '{args.region}' existiert nicht!")
            print("   Verfügbar:", list(cfg["regions"].keys()))
            return
        cfg["defaults"]["region"] = args.region
        # Region neu übernehmen
        cfg["region"] = cfg["regions"][args.region]

    region = cfg["defaults"]["region"]

    processed_dir = Path(cfg["paths"]["base_data_dir"]) / "processed" / region
    processed_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir = ROOT / "diagnostics" / region
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Monate verarbeiten
    months = None if args.all else args.month
    if months is not None:
        months = [m.zfill(2) for m in months]

    # --------------------------------------------------------
    # Dateien finden
    # --------------------------------------------------------
    tiffs = [f for f in processed_dir.iterdir()
             if f.name.endswith("_AUTOCORR.tif") and month_matches(f.name, months)]

    print(f"📦 Gefundene AUTOCORR-TIFFs in {region}: {len(tiffs)}")

    if not tiffs:
        print("⚠️ Keine passenden Dateien gefunden.")
        return

    # --------------------------------------------------------
    # Diagnose für jede Datei
    # --------------------------------------------------------
    for fp in tiffs:
        diagnose_file(fp, diagnostics_dir)

    print("\n✅ Fertig.")


if __name__ == "__main__":
    main()