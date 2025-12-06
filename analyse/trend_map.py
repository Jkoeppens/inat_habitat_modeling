#!/usr/bin/env python3
"""
Trend Map Generator
-------------------

Berechnet eine Karte der langfristigen Veränderung der Suitability
über mehrere Jahre (2017–2024), basierend auf:

  - Mode A: cumulative difference (sum of deltas)
  - Mode B: linear regression slope per pixel

Nutzt nur echte Daten: Mask >= threshold (z. B. 0.8)

Outputs:
  - GeoTIFF (Trend-Werte)
  - PNG Preview
"""

import os
import numpy as np
import argparse
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--folder", required=True,
               help="Ordner mit suitability_YEAR_*.tif")
p.add_argument("--start", type=int, default=2017)
p.add_argument("--end", type=int, default=2024)
p.add_argument("--threshold", type=float, default=0.8,
               help="Mask-Grenzwert für echte Daten")
p.add_argument("--mode", type=str, default="slope",
               choices=["slope", "cumulative"],
               help="Trendmodus")
p.add_argument("--out", type=str, default="trend_map.tif")
args = p.parse_args()

years = list(range(args.start, args.end + 1))

# -------------------------------------------------------------
# Helper
# -------------------------------------------------------------

def path_for_year(folder, year):
    # Finde die Datei automatisch
    for fn in os.listdir(folder):
        if fn.startswith(f"suitability_{year}_") and fn.endswith(".tif"):
            return os.path.join(folder, fn)
    raise FileNotFoundError(f"Kein TIFF für Jahr {year}")

# -------------------------------------------------------------
# Datenstruktur vorbereiten
# -------------------------------------------------------------
print("📥 Lade alle Jahreskarten…")

stack_vals = []
stack_mask = []

profile = None

for y in years:
    tif = path_for_year(args.folder, y)
    print(f"➡ Lade {tif}")

    with rasterio.open(tif) as src:
        if profile is None:
            profile = src.profile.copy()
            H, W = src.height, src.width

        vals = src.read(1).astype("float32")
        mask = src.read(2).astype("float32")

    # Mask anwenden
    is_real = mask >= args.threshold

    vals = np.where(is_real, vals, np.nan)

    stack_vals.append(vals)
    stack_mask.append(is_real)

stack_vals = np.stack(stack_vals)    # Shape: (T, H, W)
stack_mask = np.stack(stack_mask)    # Shape: (T, H, W)

T = len(years)

# -------------------------------------------------------------
# Trendberechnung
# -------------------------------------------------------------
print(f"📈 Berechne Trendkarte ({args.mode})…")

trend = np.full((H, W), np.nan, dtype="float32")

if args.mode == "cumulative":
    # Summe aller jährlichen Veränderungen
    # S(t+1) - S(t), nur dort wo beide Jahre real sind
    deltas = []

    for i in range(T - 1):
        a = stack_vals[i]
        b = stack_vals[i + 1]
        both_real = np.isfinite(a) & np.isfinite(b)
        delta = np.where(both_real, b - a, np.nan)
        deltas.append(delta)

    deltas = np.stack(deltas)        # (T-1, H, W)
    trend = np.nanmean(deltas, axis=0) * (T - 1)

else:
    # Linear Regression pro Pixel
    t = np.array(years)

    for i in range(H):
        if i % 200 == 0:
            print(f"  Zeile {i}/{H}")

        row = stack_vals[:, i, :]       # (T, W)
        ok = np.isfinite(row)           # echte Daten per Pixel

        for j in range(W):
            r = row[:, j]
            valid = ok[:, j]

            if valid.sum() < 3:
                continue

            slope, intercept, _, _, _ = linregress(t[valid], r[valid])
            trend[i, j] = slope

# -------------------------------------------------------------
# Speichern: GeoTIFF
# -------------------------------------------------------------
out_tif = Path(args.out)
profile.update(count=1, dtype="float32", compress="deflate")

print(f"💾 Schreibe Trend-TIFF → {out_tif}")
with rasterio.open(out_tif, "w", **profile) as dst:
    dst.write(trend, 1)

# -------------------------------------------------------------
# PNG Preview
# -------------------------------------------------------------
png_path = out_tif.with_suffix(".png")

plt.figure(figsize=(10, 10))
plt.imshow(trend, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
plt.colorbar(label="Trend (neg = decline, pos = increase)")
plt.title(f"Suitability Trend {args.start}–{args.end} ({args.mode})")
plt.axis("off")
plt.tight_layout()
plt.savefig(png_path, dpi=150)
plt.close()

print(f"🖼 PNG gespeichert: {png_path}")
print("🎉 Fertig.")