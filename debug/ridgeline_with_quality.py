#!/usr/bin/env python3
"""
Plotly Ridgeline mit global normalisierter Suitability-Dichte (blau)
und Datenqualität (orange), mit Abstandsskalierung.
"""

import os
import argparse
import numpy as np
import rasterio
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

# ---------------------------------------------------------
# CLI Argumente
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--spacing", type=float, default=1.0,
                    help="Vertikale Dehnung der Jahresachse")
parser.add_argument("--html", action="store_true",
                    help="Als HTML speichern (transparent)")
parser.add_argument("--out", type=str, default="ridgeline_global_norm.html",
                    help="Output-Datei")
args = parser.parse_args()

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------
FOLDER = "/Volumes/Data/iNaturalist/outputs/macrolepiota_procera"
YEARS  = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
MAX_SAMPLES = 150_000
BINS = 250


def path_for_year(year):
    return f"{FOLDER}/suitability_{year}_MONTHLY_Macrolepiota_procera_vs_Parus_major.tif"


# ---------------------------------------------------------
# Daten laden
# ---------------------------------------------------------
def load_sampled_suitability_and_mask(path, max_samples=MAX_SAMPLES):
    with rasterio.open(path) as src:
        suit = src.read(1).astype("float32")
        mask = src.read(2).astype("float32")

    valid = np.isfinite(suit) & np.isfinite(mask)
    suit = suit[valid]
    mask = mask[valid]

    if len(suit) > max_samples:
        idx = np.random.choice(len(suit), max_samples, replace=False)
        suit = suit[idx]
        mask = mask[idx]

    return suit, mask


# ---------------------------------------------------------
# Dichteprofil
# ---------------------------------------------------------
def compute_kde(values, bins=BINS):
    kde = gaussian_kde(values)
    xs = np.linspace(0, 1, bins)
    ys = kde(xs)  # nicht normieren!
    return xs, ys


# ---------------------------------------------------------
# Qualitätsprofil (0–1 bleibt erhalten)
# ---------------------------------------------------------
def compute_quality_profile(suit, mask, bins=BINS):
    xs = np.linspace(0, 1, bins)
    digitized = np.digitize(suit, xs)
    qvals = np.zeros(bins)
    qvals[:] = np.nan

    for i in range(bins):
        m = mask[digitized == i]
        if len(m) > 0:
            qvals[i] = m.mean()

    qvals = np.nan_to_num(qvals, nan=0.0)
    return xs, qvals


# ---------------------------------------------------------
# Plot erstellen
# ---------------------------------------------------------
def plot_ridgeline_plotly(save_html=False):

    density_all_years = {}  # zur globalen Normierung
    quality_all_years = {}

    # ---------------------------------------
    # 1) Erst alle Dichten sammeln
    # ---------------------------------------
    for year in YEARS:
        path = path_for_year(year)
        if not os.path.exists(path):
            print("⚠️ Datei fehlt:", path)
            continue

        suit, mask = load_sampled_suitability_and_mask(path)
        xs_d, dens = compute_kde(suit)
        xs_q, qual = compute_quality_profile(suit, mask)

        density_all_years[year] = (xs_d, dens)
        quality_all_years[year] = (xs_q, qual)

    # globales Maximum der Dichte (für vergleichbare Skala)
    global_max_dens = max(d.max() for (_, d) in density_all_years.values())

    # ---------------------------------------
    # 2) Plot aufbauen
    # ---------------------------------------
    fig = go.Figure()

    for idx, year in enumerate(YEARS):
        if year not in density_all_years:
            continue

        xs_d, dens = density_all_years[year]
        xs_q, qual = quality_all_years[year]

        # Globale Normierung
        dens_norm = dens / global_max_dens

        yline = np.full_like(xs_d, idx * args.spacing)

        # Dichtekurve
        fig.add_trace(go.Scatter3d(
            x=xs_d,
            y=yline,
            z=dens_norm,
            mode="lines",
            line=dict(color="dodgerblue", width=5),
            name=f"{year} Density"
        ))

        # Qualitätskurve
        fig.add_trace(go.Scatter3d(
            x=xs_q,
            y=np.full_like(xs_q, idx * args.spacing),
            z=qual,
            mode="lines",
            line=dict(color="orange", width=3, dash="dash"),
            name=f"{year} Quality"
        ))

    # ---------------------------------------
    # Layout
    # ---------------------------------------
    fig.update_layout(
        title="Ridgeline (global normalisierte Dichte + Datenqualität)",
        scene=dict(
            xaxis_title="Suitability",
            yaxis_title="Year",
            zaxis_title="Suitability / Data Quality",
            xaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1]),
            yaxis=dict(
                tickmode="array",
                tickvals=[i * args.spacing for i in range(len(YEARS))],
                ticktext=[str(y) for y in YEARS]
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=args.spacing * 0.7, z=0.5),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        width=1200,
        height=900
    )

    if save_html:
        fig.write_html(args.out, include_plotlyjs="cdn")
        print("💾 HTML gespeichert:", args.out)
    else:
        fig.show()


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
plot_ridgeline_plotly(save_html=args.html)