#!/usr/bin/env python3
"""
Trendplot der echten Pixel (Mask >= threshold) mit 95%-Konfidenzintervall.
Output: interaktives HTML mit transparentem Hintergrund.
"""

import os
import argparse
import numpy as np
import rasterio
import plotly.graph_objects as go


# -------------------------------------------------------
# 1) Sampling + Qualitätsfilter
# -------------------------------------------------------

def load_real_pixels(path, max_samples, real_threshold):
    """Lädt Band 1 (Suitability) + Band 2 (Mask), filtert echte Pixel."""
    with rasterio.open(path) as src:
        suit = src.read(1).astype("float32")
        mask = src.read(2).astype("float32")

    valid = np.isfinite(suit) & np.isfinite(mask)
    suit = suit[valid]
    mask = mask[valid]

    real = mask >= real_threshold
    suit_real = suit[real]

    # Sampling
    if len(suit_real) > max_samples:
        idx = np.random.choice(len(suit_real), max_samples, replace=False)
        suit_real = suit_real[idx]

    return suit_real


# -------------------------------------------------------
# 2) Statistiken
# -------------------------------------------------------

def compute_mean_CI(values):
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    se = std / np.sqrt(n)
    ci95 = 1.96 * se
    return mean, ci95


# -------------------------------------------------------
# 3) Plot erstellen
# -------------------------------------------------------

def make_plot(years, means, cis, out_file):
    fig = go.Figure()

    # Linie
    fig.add_trace(go.Scatter(
        x=years,
        y=means,
        mode="lines+markers",
        line=dict(color="#3b82f6", width=4),
        marker=dict(size=10,
                    color="#3b82f6",
                    line=dict(width=2, color="white")),
        name="Mean suitability",
    ))

    # 95% Konfidenzband
    upper = np.array(means) + np.array(cis)
    lower = np.array(means) - np.array(cis)

    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(59,130,246,0.15)",  # leichter blau-Schimmer
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="95% CI",
    ))

    # Layout
    fig.update_layout(
        title="Trend der Habitat-Suitability (nur echte Pixel, 95% CI)",
        xaxis_title="Jahr",
        yaxis_title="Suitability (0–1)",
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=1100,
        height=650,
        font=dict(size=16),
    )

    # Y-Skala fixieren
    fig.update_yaxes(range=[0, 1])

    fig.write_html(out_file, include_plotlyjs="cdn", full_html=True)
    print(f"💾 Gespeichert nach: {out_file}")



# -------------------------------------------------------
# 4) CLI
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--real_threshold", type=float, default=0.8)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--contrast", type=str, required=True)
    parser.add_argument("--out", type=str, default="trend_real_CI.html")
    args = parser.parse_args()

    years = list(range(args.start, args.end + 1))
    means = []
    cis = []

    for year in years:
        path = f"{args.folder}/suitability_{year}_MONTHLY_{args.target}_vs_{args.contrast}.tif"

        if not os.path.exists(path):
            print(f"⚠️ Datei fehlt: {path}")
            means.append(np.nan)
            cis.append(np.nan)
            continue

        vals = load_real_pixels(path, args.samples, args.real_threshold)
        print(f"➡ Jahr {year}: {len(vals)} echte Pixel")

        mean, ci95 = compute_mean_CI(vals)
        means.append(mean)
        cis.append(ci95)

    make_plot(years, means, cis, args.out)


if __name__ == "__main__":
    main()