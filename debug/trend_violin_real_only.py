#!/usr/bin/env python3
"""
Violin-Plot der jährlichen Suitability-Verteilungen (nur echte Daten)

- X-Achse = Jahr
- Y-Achse = Suitability
- Nur Pixel, deren Datenqualität >= real_threshold
- Farben angelehnt an Viridis
- HTML-Ausgabe mit komplett transparentem Hintergrund
"""

import os
import argparse
import numpy as np
import rasterio
import plotly.graph_objects as go


# ---------------------------------------------------
# Daten laden (Suitability + Maske)
# ---------------------------------------------------
def load_real_pixels(path, real_threshold=0.95, max_samples=200_000):
    """Lädt nur echte Pixel entsprechend der Data-Quality-Maske."""
    with rasterio.open(path) as src:
        suit = src.read(1).astype("float32")
        mask = src.read(2).astype("float32")

    valid = np.isfinite(suit) & np.isfinite(mask)
    suit = suit[valid]
    mask = mask[valid]

    real_pixels = suit[mask >= real_threshold]

    print(f"➡ {os.path.basename(path)}: {len(real_pixels)} echte Pixel")

    # Sampling
    if len(real_pixels) > max_samples:
        idx = np.random.choice(len(real_pixels), max_samples, replace=False)
        real_pixels = real_pixels[idx]

    return real_pixels


# ---------------------------------------------------
# Plot erzeugen
# ---------------------------------------------------
def make_violin_plot(all_years, pixel_values, output_html):

    fig = go.Figure()

    # Viridis-Farben pro Jahr
    from plotly.colors import sample_colorscale
    colors = sample_colorscale("Viridis", np.linspace(0, 1, len(all_years)))

    for i, year in enumerate(all_years):
        vals = pixel_values[i]
        fig.add_trace(go.Violin(
            x=np.full_like(vals, year, dtype=float),
            y=vals,
            name=str(year),
            line_color=colors[i],
            fillcolor=colors[i].replace("rgb", "rgba").replace(")", ",0.4)"),
            meanline_visible=True,
            box_visible=True
        ))

    fig.update_layout(
        title="Verteilung der Suitability (nur echte Daten)",
        xaxis_title="Jahr",
        yaxis_title="Suitability (0–1)",
        template="plotly_white",

        # Transparenter Hintergrund
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=1200,
        height=700,
    )

    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)
    print(f"💾 HTML gespeichert unter: {output_html}")

    fig.show()


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--real_threshold", type=float, default=0.95,
                        help="Mask-Wert ab dem ein Pixel als echt gilt")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--contrast", type=str, required=True)
    parser.add_argument("--out", type=str, default="violin_real_only.html")
    args = parser.parse_args()

    years = list(range(args.start, args.end + 1))

    pixel_values = []

    for year in years:
        path = f"{args.folder}/suitability_{year}_MONTHLY_{args.target}_vs_{args.contrast}.tif"

        if not os.path.exists(path):
            print(f"⚠️ fehlt: {path}")
            pixel_values.append(np.array([]))
            continue

        vals = load_real_pixels(path, real_threshold=args.real_threshold,
                                max_samples=args.samples)
        pixel_values.append(vals)

    make_violin_plot(years, pixel_values, args.out)


if __name__ == "__main__":
    main()