#!/usr/bin/env python3
"""
qa_suitability_maps.py

Qualitätsanalyse für Suitability-Maps:
- Lädt Suitability (Band 1) + Data-Usage-Mask (Band 2)
- Zufällige Stichprobe ziehen
- Spearmann-Korrelation suitability ↔ mask (Daten-Qualität)
- Histogramme speichern
- Scatterplot speichern

Verkabelt auf:
    /Volumes/Data/iNaturalist/outputs/macrolepiota_procera
"""

import random
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# =========================================================
#  Pfade
# =========================================================
OUTPUT_ROOT = Path("/Volumes/Data/iNaturalist/outputs/macrolepiota_procera")
QA_DIR = OUTPUT_ROOT / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
#  Hilfsfunktion: QA für einzelne Datei
# =========================================================
def run_qa(file_path, n_samples=20000):
    file_path = Path(file_path)
    print(f"\n🔍 QA für: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # Output-Unterordner für dieses File
    out_dir = QA_DIR / file_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(file_path) as src:
        suitability = src.read(1)
        mask = src.read(2)

    H, W = suitability.shape
    N = H * W

    print(f"  Rastergröße: {W} × {H}  →  {N:,} Pixel")

    # -----------------------------------------------------
    # Stichprobe ziehen
    # -----------------------------------------------------
    if n_samples > N:
        n_samples = N

    idx = np.random.choice(N, size=n_samples, replace=False)

    suit_flat = suitability.reshape(-1)[idx]
    mask_flat = mask.reshape(-1)[idx]

    # -----------------------------------------------------
    # Korrelation
    # -----------------------------------------------------
    rho, p = spearmanr(mask_flat, suit_flat)
    print(f"  📈 Spearman r = {rho:.4f}   (p={p:.3g})")

    # -----------------------------------------------------
    # Speichern: Scatterplot
    # -----------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(mask_flat, suit_flat, s=3, alpha=0.3)
    plt.xlabel("Mask (0–1)")
    plt.ylabel("Suitability (0–1)")
    plt.title(f"Scatter: {file_path.name}\nr={rho:.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=150)
    plt.close()

    # -----------------------------------------------------
    # Speichern: Histogramme
    # -----------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.hist(mask_flat, bins=50, alpha=0.7)
    plt.title("Distribution of Mask Values")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_mask.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(suit_flat, bins=50, alpha=0.7)
    plt.title("Distribution of Suitability Values")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_suitability.png", dpi=150)
    plt.close()

    print(f"  📁 QA gespeichert in: {out_dir}")
    return rho, p


# =========================================================
# Batch: Alle Dateien im Ordner
# =========================================================
def run_qa_for_all(n_samples=20000):
    tifs = sorted(OUTPUT_ROOT.glob("suitability_*.tif"))
    print(f"\n🌍 QA für {len(tifs)} Dateien\n")

    results = []
    for fp in tifs:
        r, p = run_qa(fp, n_samples=n_samples)
        results.append((fp.name, r, p))

    return results


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QA für Suitability-Maps")
    parser.add_argument("--file", type=str, help="Einzelne Datei prüfen")
    parser.add_argument("--all", action="store_true", help="Alle Dateien prüfen")
    parser.add_argument("--samples", type=int, default=20000, help="Anzahl Stichproben")

    args = parser.parse_args()

    if args.file:
        run_qa(args.file, n_samples=args.samples)
    else:
        run_qa_for_all(n_samples=args.samples)