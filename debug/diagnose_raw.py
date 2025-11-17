import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

RAW_DIR = "/Volumes/Data/iNaturalist/raw/berlin"
OUT_DIR = "/Volumes/Data/iNaturalist/diagnostics_raw"
os.makedirs(OUT_DIR, exist_ok=True)

# Erlaubte Namensmuster:
# NDVI_NDWI_MEAN_YYYY_MM.tif  oder berlin_2023_10_test.tif  etc.
FILENAME_PATTERN = re.compile(
    r".*(\d{4})[_\-](\d{2}).*\.(tif|tiff)$",
    re.IGNORECASE
)

# -------------------------------------------------------------------
# IO Helpers
# -------------------------------------------------------------------

def load_tiff(path):
    """Lädt ein TIFF und gibt (numpy array, profile) zurück."""
    with rasterio.open(path) as ds:
        data = ds.read()
        profile = ds.profile
    return data, profile


def plot_valid_mask(mask, title, outpath):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_hist(values, title, outpath):
    plt.figure(figsize=(10, 4))
    plt.hist(values, bins=200)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# -------------------------------------------------------------------
# Diagnose einer Datei
# -------------------------------------------------------------------

def diagnose_raw_file(path):
    fname = os.path.basename(path)
    print(f"\n🔍 Datei: {fname}")

    # Ordner Name anhand des TIFFs
    out_dir = os.path.join(OUT_DIR, fname.replace(".tif", ""))
    os.makedirs(out_dir, exist_ok=True)

    # Datei laden
    data, profile = load_tiff(path)

    bands = data.shape[0]
    if bands < 2:
        print("❌ Nicht genug Bänder (min. 2 nötig: NDVI, NDWI).")
        return

    # NDVI/NDWI über Bände 0 und 1
    ndvi = data[0].astype(float)
    ndwi = data[1].astype(float)

    total_pix = ndvi.size

    valid_ndvi = np.isfinite(ndvi)
    valid_ndwi = np.isfinite(ndwi)

    share_ndvi = valid_ndvi.sum() / total_pix
    share_ndwi = valid_ndwi.sum() / total_pix

    print(f"   ▸ NDVI gültig: {share_ndvi*100:.2f}%")
    print(f"   ▸ NDWI gültig: {share_ndwi*100:.2f}%")

    # Masks plotten
    plot_valid_mask(
        valid_ndvi,
        f"Gültige Pixel – NDVI ({fname})",
        os.path.join(out_dir, "valid_ndvi.png")
    )

    plot_valid_mask(
        valid_ndwi,
        f"Gültige Pixel – NDWI ({fname})",
        os.path.join(out_dir, "valid_ndwi.png")
    )

    # Histogramme
    if valid_ndvi.any():
        plot_hist(ndvi[valid_ndvi],
                  "NDVI Verteilung",
                  os.path.join(out_dir, "hist_ndvi.png"))

    if valid_ndwi.any():
        plot_hist(ndwi[valid_ndwi],
                  "NDWI Verteilung",
                  os.path.join(out_dir, "hist_ndwi.png"))

    # Extra: Zähle extreme Werte
    print(f"   ▸ NDVI < -0.5 : {np.sum(ndvi < -0.5)} Pixel")
    print(f"   ▸ NDVI >  0.9 : {np.sum(ndvi > 0.9)} Pixel")

    print(f"   ✔ Rohdaten-Diagnose gespeichert in: {out_dir}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    all_files = os.listdir(RAW_DIR)
    tiffs = [f for f in all_files if f.lower().endswith(".tif")]

    print(f"📦 Gefundene TIFFs: {len(tiffs)}")

    for fname in sorted(tiffs):
        full = os.path.join(RAW_DIR, fname)

        # prüfen ob Name zu unserem Schema passt
        if not FILENAME_PATTERN.match(fname):
            print(f"⚠️ Überspringe (kein Monatsdatei-Muster): {fname}")
            continue

        diagnose_raw_file(full)


if __name__ == "__main__":
    main()