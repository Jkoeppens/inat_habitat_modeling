#!/usr/bin/env python3
"""
Violin-Plot der Suitability pro Jahr (nur echte Daten, Maskenfilter)

- X-Achse = Jahr
- Y-Achse = Suitability (0–1)
- Farbe = Viridis (wie in der Karte)
- Hintergrund transparent
- Output: interaktive HTML-Datei
"""

import os
import argparse
import numpy as np
import rasterio
import plotly.graph_objects as go
import plotly.express as px


# ---------------------------------------------------------
# Daten laden (Suitability + Mask)
# ---------------------------------------------------------

def load_real_values(path, max_samples=200_000, mask_threshold=0.8):
    """Lädt Suitability und filtert nach echter Datenqualität."""
    with rasterio.open(path) as src:
        suit = src.read(1).astype("float32")
        mask = src.read(2).astype("float32")

    valid = np.isfinite(suit) & np.isfinite(mask)
    suit = suit[valid]
    mask = mask[valid]

    # Nur echte Daten
    real_idx = mask >= mask_threshold
    suit_real = suit[real_idx]

    if len(suit_real) > max_samples:
        idx = np.random.choice(len(suit_real), max_samples, replace=False)
        suit_real = suit_real[idx]

    return suit_real


# ---------------------------------------------------------
# Hauptfunktion: Violin-Plot bauen
# ---------------------------------------------------------

def plot_violin(folder, years, mask_threshold, samples, output_html):

    # Viridis Farbschema
    colors = px.colors.sequential.Viridis

    fig = go.Figure()

    for i, year in enumerate(years):
        path = f"{folder}/suitability_{year}_MONTHLY_Macrolepiota_procera_vs_Parus_major.tif"
        if not os.path.exists(path):
            print(f"⚠️ Datei fehlt: {path}")
            continue

        vals = load_real_values(path, max_samples=samples, mask_threshold=mask_threshold)
        print(f"➡ Jahr {year}: {len(vals)} echte Pixel")

        fig.add_trace(go.Violin(
            x=[year] * len(vals),     # Jahr auf X-Achse
            y=vals,                   # Suitability
            line_color=colors[int(i / len(years) * (len(colors) - 1))],
            fillcolor=colors[int(i / len(years) * (len(colors) - 1))],
            opacity=0.6,
            box_visible=True,
            meanline_visible=True,
            spanmode="hard",
            name=str(year)
        ))

    fig.update_layout(
        title="Violin-Plot der Suitability pro Jahr (nur echte Daten)",
        xaxis_title="Jahr",
        yaxis_title="Suitability (0–1)",
        template="plotly_white",
        width=1400,
        height=800,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)
    print(f"💾 HTML gespeichert: {output_html}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--start", type=int, default=2017)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Masken-Schwelle für echte Daten (0–1)")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--out", type=str, default="violin_realdata.html")
    args = parser.parse_args()

    years = list(range(args.start, args.end + 1))

    plot_violin(
        folder=args.folder,
        years=years,
        mask_threshold=args.threshold,
        samples=args.samples,
        output_html=args.out
    )


if __name__ == "__main__":
    main()