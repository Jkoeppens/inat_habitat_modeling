#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trend-Spatial-Stats für große Raster (z.B. 11k x 11k).

Berechnet:

  • Globale QA-Maske aus allen Jahres-Suitability-Karten (2017–2024)
      → QA_mean = Mittelwert aus Band 2 über alle Jahre
      → gültig, wenn QA_mean >= qa_min (Standard: 0.5)
  • Moran’s I (global, auf Stichprobe aus allen gültigen Trendpixeln)
  • LISA-Karte (lokaler Moran's I)
  • Quadranten-Raster (HH/LL/LH/HL) aus LISA
  • Gi*-Hotspotkarte (Z-Scores, ohne Permutationstests)
  • Quantile-Karte (top/bottom 5% Trend)

Features:

  • QA-Schwelle (z.B. 0.5) über alle Jahre
  • Kachelverarbeitung (Tiles), um RAM im Zaum zu halten
  • Testmodus: nur ein zentraler Ausschnitt (max. 2048×2048) wird berechnet

Konsolen-Aufruf – FULL RUN:

  python analyse/spatial_trend_stats.py \
      --trend "/Volumes/Data/iNaturalist/outputs/trend_map_2017_2024.tif" \
      --qa_folder "/Volumes/Data/iNaturalist/outputs/macrolepiota_procera" \
      --qa_min 0.5 \
      --tile 512 \
      --out_prefix "/Volumes/Data/iNaturalist/outputs/trendstats_2017_2024"

Konsolen-Aufruf – TESTMODUS:

  python analyse/spatial_trend_stats.py \
      --trend "/Volumes/Data/iNaturalist/outputs/trend_map_2017_2024.tif" \
      --qa_folder "/Volumes/Data/iNaturalist/outputs/macrolepiota_procera" \
      --qa_min 0.5 \
      --tile 512 \
      --test \
      --out_prefix "/Volumes/Data/iNaturalist/outputs/trendstats_2017_2024_TEST"
"""

import os
import argparse

import numpy as np
import rasterio
from rasterio.windows import Window

from libpysal.weights import lat2W, WSP
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local


# =====================================================================
# Utility-Funktionen
# =====================================================================

def ensure_dir(path: str):
    """Stellt sicher, dass der Zielordner existiert."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_band(path: str, band: int = 1, dtype: str = "float64"):
    """Liest ein Band als numpy-Array und mappt NoData auf NaN."""
    with rasterio.open(path) as src:
        arr = src.read(band).astype(dtype)
        profile = src.profile
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return arr, profile


def write_tif(path: str, arr: np.ndarray, profile: dict, dtype: str = "float32"):
    """Schreibt ein Single-Band-TIF."""
    ensure_dir(path)
    prof = profile.copy()
    prof.update(
        dtype=dtype,
        count=1,
        compress="deflate",
        nodata=np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


# =====================================================================
# QA über alle Jahre
# =====================================================================

def build_trend_qa_mask(shape, qa_folder, years, qa_min):
    """
    Lädt für alle Jahre die QA (Band 2) der Suitability-Karten,
    mittelt sie und baut daraus eine Maske (True = gültig).

    Erwartetes Namensschema im qa_folder:
      suitability_<YEAR>_MONTHLY_Macrolepiota_procera_vs_Parus_major.tif
    """

    H, W = shape
    qa_stack = []

    print("📥 Lade QA aller Jahre für Trend-QA…")
    for year in years:
        qa_path = os.path.join(
            qa_folder,
            f"suitability_{year}_MONTHLY_Macrolepiota_procera_vs_Parus_major.tif"
        )
        print(f"  → {year}: {qa_path}")
        qa_year, _ = read_band(qa_path, band=2, dtype="float64")
        if qa_year.shape != shape:
            raise ValueError(
                f"QA-Shape {qa_year.shape} passt nicht zur Trend-Shape {shape} "
                f"(Jahr {year}, Datei {qa_path})."
            )
        qa_stack.append(qa_year)

    qa_arr = np.stack(qa_stack, axis=0)  # (n_years, H, W)
    qa_mean = np.nanmean(qa_arr, axis=0)

    valid = (qa_mean >= qa_min) & np.isfinite(qa_mean)

    n_valid = int(valid.sum())
    n_total = H * W
    print(f"✅ Gültige Trend-Pixel über alle Jahre (QA_mean >= {qa_min}): "
          f"{n_valid} / {n_total}")

    return valid


# =====================================================================
# Kernfunktion: Tiles verarbeiten
# =====================================================================

def process_tiles(trend_path,
                  qa_folder,
                  qa_min,
                  tile,
                  out_prefix,
                  years=(2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024),
                  test=False):
    """
    Hauptpipeline:
      - Trend laden
      - QA-Maske aus allen Jahren
      - optional Testmodus (zentraler Ausschnitt)
      - globale Quantile + globaler Moran (Stichprobe)
      - Tile-basierte Berechnung von:
          LISA (Moran_Local),
          Quadranten,
          Gi* (G_Local),
          Quantilen (global, 5%/95%)
    """

    # ---------------------------
    # Trend laden
    # ---------------------------
    trend_full, profile = read_band(trend_path, band=1, dtype="float64")
    H, W = trend_full.shape
    print("📥 Lade Trendkarte…")
    print(f"   Größe: {W} × {H}")

    # ---------------------------
    # QA-Maske (über alle Jahre)
    # ---------------------------
    valid_full = build_trend_qa_mask((H, W), qa_folder, years, qa_min)

    # ---------------------------
    # Optionaler Test-Ausschnitt
    # ---------------------------
    if test:
        max_side = 2048
        h = min(H, max_side)
        w = min(W, max_side)

        y0 = (H - h) // 2
        x0 = (W - w) // 2
        y1 = y0 + h
        x1 = x0 + w

        print(f"🧪 TESTMODUS: beschränke Analyse auf {w}×{h} "
              f"(y={y0}:{y1}, x={x0}:{x1})")

        # Ausschnitt
        trend_full = trend_full[y0:y1, x0:x1]
        valid_full = valid_full[y0:y1, x0:x1]
        H, W = trend_full.shape

        # Profil-Transform anpassen
        with rasterio.open(trend_path) as src:
            window = Window(col_off=x0, row_off=y0, width=W, height=H)
            transform = rasterio.windows.transform(window, src.transform)
        profile["transform"] = transform
        profile["height"] = H
        profile["width"] = W

    # ---------------------------
    # Trendwerte nach QA filtern + globale Quantile
    # ---------------------------
    print("📊 Berechne globale Trend-Quantile (5% / 95%)…")
    valid_trend = trend_full[valid_full]
    valid_trend = valid_trend[np.isfinite(valid_trend)]
    if valid_trend.size == 0:
        raise RuntimeError("Kein gültiger Trendwert nach QA/NaN-Filter!")

    q05 = float(np.nanquantile(valid_trend, 0.05))
    q95 = float(np.nanquantile(valid_trend, 0.95))
    print(f"   5%-Quantil: {q05:.5f}")
    print(f"   95%-Quantil: {q95:.5f}")

    # ---------------------------
    # Globales Moran's I (Stichprobe)
    # ---------------------------
    print("📊 Berechne globales Moran's I (Stichprobe)…")
    sample_size = min(250_000, valid_trend.size)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(valid_trend.size, size=sample_size, replace=False)
    sample_vals = valid_trend[sample_idx]

    side = int(np.sqrt(sample_size))
    side = max(4, side)
    cut = side * side
    sample_vals = sample_vals[:cut]

    W_sample = lat2W(side, side, rook=True)
    moran_global = Moran(sample_vals, W_sample, permutations=0)

    # ---------------------------
    # Output-Arrays
    # ---------------------------
    lisa_full = np.full((H, W), np.nan, dtype="float32")
    quad_full = np.zeros((H, W), dtype="uint8")
    gi_full = np.full((H, W), np.nan, dtype="float32")
    quant_full = np.zeros((H, W), dtype="float32")

    nx = (W + tile - 1) // tile
    ny = (H + tile - 1) // tile
    n_tiles = nx * ny

    print(f"📦 Prozessiere {n_tiles} Tiles ({nx}×{ny})…")

    # ---------------------------
    # Tiles durchlaufen
    # ---------------------------
    tile_counter = 0
    for ty in range(ny):
        for tx in range(nx):
            tile_counter += 1
            y0 = ty * tile
            x0 = tx * tile
            y1 = min(H, y0 + tile)
            x1 = min(W, x0 + tile)

            print(f"[Tile {tile_counter}/{n_tiles}] y={y0}:{y1} x={x0}:{x1}")

            t_tile = trend_full[y0:y1, x0:x1]
            v_tile = valid_full[y0:y1, x0:x1]

            flat_trend = t_tile.flatten()
            flat_valid = v_tile.flatten()

            # Indizes gültiger Pixel in diesem Tile
            good_mask = flat_valid & np.isfinite(flat_trend)
            n_good = int(good_mask.sum())
            if n_good < 10:
                continue

            vec = flat_trend[good_mask].astype("float64")

            # Spatial Weights für das komplette Tile (rechteckig)
            h_tile, w_tile = t_tile.shape
            W_full = lat2W(h_tile, w_tile, rook=True)

            # Weights auf gültige Pixel subsetten (WSP)
            # Achtung: Reihenfolge von lat2W entspricht Flatten-Reihenfolge (row-major)
            W_sparse_full = W_full.sparse
            W_sparse_sub = W_sparse_full[good_mask, :][:, good_mask]
            W_tile = WSP(W_sparse_sub)

            # ---------------------------
            # Lokaler Moran (LISA)
            # ---------------------------
            lisa = Moran_Local(vec, W_tile, permutations=0)
            Is = lisa.Is.astype("float32")
            q = lisa.q.astype("uint8")

            lisa_sub = lisa_full[y0:y1, x0:x1].flatten()
            quad_sub = quad_full[y0:y1, x0:x1].flatten()
            lisa_sub[good_mask] = Is
            quad_sub[good_mask] = q
            lisa_full[y0:y1, x0:x1] = lisa_sub.reshape(h_tile, w_tile)
            quad_full[y0:y1, x0:x1] = quad_sub.reshape(h_tile, w_tile)

            # ---------------------------
            # Gi* (lokaler Hotspot, Z-Score)
            # ---------------------------
            gi = G_Local(vec, W_tile, permutations=0).Zs.astype("float32")
            gi_sub = gi_full[y0:y1, x0:x1].flatten()
            gi_sub[good_mask] = gi
            gi_full[y0:y1, x0:x1] = gi_sub.reshape(h_tile, w_tile)

    # ---------------------------
    # Quantile-Karte (global, mit QA)
    # ---------------------------
    print("📊 Erzeuge globale Quantile-Karte…")
    mask_valid = valid_full & np.isfinite(trend_full)
    quant_full[(trend_full <= q05) & mask_valid] = -1.0
    quant_full[(trend_full >= q95) & mask_valid] = 1.0
    # Rest bleibt 0.0 (Mittelfeld oder ungültig)

    # =================================================================
    # Outputs schreiben
    # =================================================================
    moran_path = out_prefix + "_moran_global.txt"
    with open(moran_path, "w") as f:
        f.write("Moran's I (global, Stichprobe aus allen gültigen Trendpixeln)\n")
        f.write(f"I   = {moran_global.I}\n")
        f.write(f"E[I] = {moran_global.EI}\n")
        f.write(f"Var = {moran_global.VI_norm}\n")

    write_tif(out_prefix + "_lisa.tif", lisa_full, profile, dtype="float32")
    write_tif(out_prefix + "_quad.tif", quad_full, profile, dtype="uint8")
    write_tif(out_prefix + "_gi.tif", gi_full, profile, dtype="float32")
    write_tif(out_prefix + "_quantiles.tif", quant_full, profile, dtype="float32")

    print("🎉 Fertig! Dateien gespeichert unter:")
    print("  ", moran_path)
    print("  ", out_prefix + "_lisa.tif")
    print("  ", out_prefix + "_quad.tif")
    print("  ", out_prefix + "_gi.tif")
    print("  ", out_prefix + "_quantiles.tif")


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trend", required=True,
                    help="Pfad zur Trendkarte (z.B. trend_map_2017_2024.tif)")
    ap.add_argument("--qa_folder", required=True,
                    help="Ordner mit Jahres-Suitability-Karten")
    ap.add_argument("--qa_min", type=float, default=0.5,
                    help="QA-Schwelle über alle Jahre (Default 0.5)")
    ap.add_argument("--tile", type=int, default=512,
                    help="Tile-Größe in Pixeln (Default 512)")
    ap.add_argument("--out_prefix", required=True,
                    help="Prefix für Output-Dateien")
    ap.add_argument("--test", action="store_true",
                    help="Nur zentralen max. 2048×2048-Ausschnitt analysieren")
    args = ap.parse_args()

    process_tiles(
        trend_path=args.trend,
        qa_folder=args.qa_folder,
        qa_min=args.qa_min,
        tile=args.tile,
        out_prefix=args.out_prefix,
        years=(2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024),
        test=args.test
    )


if __name__ == "__main__":
    main()