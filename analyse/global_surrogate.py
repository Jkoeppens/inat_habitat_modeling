#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global Surrogate Tree für ein XGBoost-Modell + D3-Visualisierung.

Aufruf z.B.:

python analyse/global_surrogate.py \
  --model "/Volumes/Data/iNaturalist/outputs/macrolepiota_procera/model_MONTHLY_Macrolepiota_procera_vs_Parus_major.json" \
  --data  "/Volumes/Data/iNaturalist/features/Macrolepiota_procera/inat_with_climatology_Macrolepiota_procera_vs_Parus_major.csv" \
  --out   surrogate_tree.html \
  --depth 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

# -------------------------------
# Farbpaletten & Ranges (Python)
# -------------------------------

NDVI_PALETTE = ["#f2f2f2", "#a3c586", "#2f6b3a"]
NDWI_PALETTE = ["#f7fbff", "#6baed6", "#08519c"]
MORAN_PALETTE = ["#fee8c8", "#fdbb84", "#e34a33"]
GEARY_PALETTE = ["#f7f4f9", "#998ec3", "#542788"]

RANGES = {
    "ndvi_mean": (0.0, 1.0),
    "ndwi_mean": (-1.0, 1.0),
    "moran": (-0.2, 4.0),
    "geary": (0.0, 1.5),
}

# relativ feine Viridis-Skala für die Suitability-Bar
VIRIDIS_COLORS = [
    "#440154", "#472f7d", "#3b518b", "#2c718e",
    "#21918c", "#27ad81", "#5cc863", "#aadc32",
    "#fde725"
]


# -------------------------------
# Semantik / Feature-Typ
# -------------------------------

def infer_semantics(feature_name: str):
    """
    Nimmt z.B.
      m12_ndvi_mean
      m08_geary_ndwi
      m10_moran_ndvi
    und liefert Label, Palette, Range.
    """
    if not (feature_name.startswith("m") and "_" in feature_name):
        return {
            "label": feature_name,
            "palette": ["#dddddd", "#aaaaaa", "#666666"],
            "range": (0.0, 1.0),
        }

    try:
        month = int(feature_name[1:3])
        rest = feature_name[4:]
    except Exception:
        return {
            "label": feature_name,
            "palette": ["#dddddd", "#aaaaaa", "#666666"],
            "range": (0.0, 1.0),
        }

    if month in (7, 8, 9):
        season = "Sommer"
    elif month in (10, 11, 12):
        season = "Herbst"
    else:
        season = "Saison"

    label = feature_name
    palette = ["#dddddd", "#aaaaaa", "#666666"]
    rng = (0.0, 1.0)

    if rest.startswith("ndvi_mean"):
        label = f"Vegetationsdichte ({season}, NDVI)"
        palette = NDVI_PALETTE
        rng = RANGES["ndvi_mean"]
    elif rest.startswith("ndwi_mean"):
        label = f"Feuchtigkeit ({season}, NDWI)"
        palette = NDWI_PALETTE
        rng = RANGES["ndwi_mean"]
    elif rest.startswith("moran_ndvi"):
        label = f"Vegetations-Cluster ({season}, Moran)"
        palette = MORAN_PALETTE
        rng = RANGES["moran"]
    elif rest.startswith("moran_ndwi"):
        label = f"Feuchtigkeits-Cluster ({season}, Moran)"
        palette = MORAN_PALETTE
        rng = RANGES["moran"]
    elif rest.startswith("geary_ndvi"):
        label = f"Vegetations-Heterogenität ({season}, Geary)"
        palette = GEARY_PALETTE
        rng = RANGES["geary"]
    elif rest.startswith("geary_ndwi"):
        label = f"Feuchtigkeits-Heterogenität ({season}, Geary)"
        palette = GEARY_PALETTE
        rng = RANGES["geary"]

    return {
        "label": label,
        "palette": palette,
        "range": rng,
    }


# -------------------------------
# 1) Surrogate-Tree trainieren
# -------------------------------

def train_surrogate(model_path: str, data_path: str, max_depth: int = 4):
    print("→ Lade XGBoost-Modell…")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    booster = model.get_booster()
    feature_names = booster.feature_names
    if feature_names is None:
        raise ValueError("❌ Modell enthält keine Feature-Namen im Booster!")

    print(f"→ Modell hat {len(feature_names)} Features")

    print("→ Lade Daten…")
    df = pd.read_csv(data_path)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"❌ CSV enthält nicht alle Modell-Features. Fehlend: {missing}")

    X = df[feature_names].copy()

    # Objektspalten in float, wenn möglich
    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = X[col].astype(float)
            except Exception:
                raise ValueError(f"❌ Spalte {col} ist object und lässt sich nicht in float casten.")

    print("→ Berechne Modell-Predictions (P(1))…")
    y_pred = model.predict_proba(X)[:, 1]

    print("→ Trainiere Surrogate DecisionTreeRegressor…")
    surrogate = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=50,
        random_state=42
    )
    surrogate.fit(X, y_pred)

    return surrogate, feature_names


# -------------------------------
# 2) Sklearn-Tree → JSON-Baum
# -------------------------------

def tree_to_json(tree, feature_names):
    """
    Konvertiert sklearn.tree_ in rekursives JSON:
    - Splits: {feature, threshold, yes, no, label, palette, range}
    - Leaf:   {leaf, suit}
    suit = mean prediction im Leaf
    """

    def recurse(node_id: int):
        # Leaf?
        if tree.feature[node_id] == -2:  # sklearn: -2 = LEAF
            val = float(tree.value[node_id][0][0])
            return {
                "leaf": val,
                "suit": val
            }

        feat_idx = int(tree.feature[node_id])
        feat_name = feature_names[feat_idx]
        thr = float(tree.threshold[node_id])

        semantics = infer_semantics(feat_name)

        left_id = int(tree.children_left[node_id])
        right_id = int(tree.children_right[node_id])

        left = recurse(left_id)
        right = recurse(right_id)

        node = {
            "feature": feat_name,
            "threshold": thr,
            "yes": right,   # Konvention: "Ja" = rechts (>= threshold)
            "no": left,     # "Nein" = links  (< threshold)
        }
        node.update(semantics)
        return node

    return recurse(0)


# -------------------------------
# 3) HTML-Template (D3)
# -------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Global Surrogate Tree</title>
<style>
  body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0;
    padding: 1rem;
    background: #f7f7f7;
  }
  .label {
    font-size: 12px;
    text-anchor: middle;
  }
  .edge-label {
    font-size: 11px;
    fill: #555;
  }
</style>

<svg id="tree-svg"></svg>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>

// ----- Eingebettete Daten -----
const treeData = REPLACE_TREE_JSON;
const viridisColors = REPLACE_VIRIDIS_COLORS;

const svg = d3.select("#tree-svg");

// Layout-Parameter
const dx = 260;    // horizontal spacing zwischen Nodes
const dy = 200;    // vertikal
const nodeW = 260;
const nodeH = 80;

// Hierarchie aus yes/no
const root = d3.hierarchy(treeData, d => {
  const c = [];
  if (d.yes) c.push(Object.assign({_edge: "Ja"}, d.yes));
  if (d.no)  c.push(Object.assign({_edge: "Nein"}, d.no));
  return c.length ? c : null;
});

// Baum-Layout
d3.tree().nodeSize([dx, dy])(root);

// Extents bestimmen
let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
root.each(d => {
  if (d.x < minX) minX = d.x;
  if (d.x > maxX) maxX = d.x;
  if (d.y < minY) minY = d.y;
  if (d.y > maxY) maxY = d.y;
});

const padding = 260;

svg
  .attr("width",  (maxX - minX) + 2 * padding)
  .attr("height", (maxY - minY) + 3 * padding);

const xOffset = padding - minX;
const yOffset = padding - minY;

const g = svg.append("g")
  .attr("transform", "translate(" + xOffset + "," + yOffset + ")");

// Gradient-Defs
const defs = svg.append("defs");

// Hilfsfunktion: Threshold-X in Node-Koordinaten
function thresholdXLocal(data) {
  if (!data.range || data.threshold == null) {
    return 0;
  }
  const min = data.range[0];
  const max = data.range[1];
  const t = data.threshold;
  let rel = (t - min) / (max - min);
  if (!Number.isFinite(rel)) rel = 0.5;
  rel = Math.max(0.05, Math.min(0.95, rel));   // nicht ganz an den Rand
  return -nodeW/2 + rel * nodeW;
}

// Knoten-spezifische Palette → Gradient-ID
function ensureGradientForNode(data) {
  const pal = data.palette || ["#eeeeee", "#cccccc", "#888888"];
  const id = "grad_" + Math.random().toString(36).slice(2);
  const grad = defs.append("linearGradient")
    .attr("id", id)
    .attr("x1", "0%").attr("x2", "100%")
    .attr("y1", "0%").attr("y2", "0%");
  pal.forEach((c, i) => {
    grad.append("stop")
      .attr("offset", (i/(pal.length-1))*100 + "%")
      .attr("stop-color", c);
  });
  return id;
}

// Bezier-Funktion für weiche Kurven (Option C)
function bezierPath(sx, sy, tx, ty) {
  const c1y = sy + (ty - sy) * 0.3;
  const c2y = sy + (ty - sy) * 0.7;
  const c1x = sx;
  const c2x = tx;
  return "M " + sx + "," + sy +
         " C " + c1x + "," + c1y +
         " " + c2x + "," + c2y +
         " " + tx + "," + ty;
}

// Startposition der Kante an der Eltern-Box (rechts/links neben Threshold)
function edgeStartX(sourceNode, side) {
  const data = sourceNode.data;
  const txLocal = thresholdXLocal(data);
  const margin = nodeW * 0.12;
  let x = txLocal + (side === "yes" ? +margin : -margin);
  // zur Sicherheit innerhalb der Box halten
  const min = -nodeW/2 + 8;
  const max =  nodeW/2 - 8;
  x = Math.max(min, Math.min(max, x));
  return sourceNode.x + x;
}

// ---------------- Links zwischen Splits ----------------
const splitLinks = root.links().filter(d => {
  // nur Kanten, deren Ziel kein Leaf ist
  return d.target.data.leaf === undefined;
});

g.append("g")
  .selectAll("path")
  .data(splitLinks)
  .join("path")
  .attr("fill", "none")
  .attr("stroke", "#999")
  .attr("stroke-width", 1.2)
  .attr("d", d => {
    const side = d.target.data._edge === "Ja" ? "yes" : "no";
    const sx = edgeStartX(d.source, side);
    const sy = d.source.y + nodeH/2;
    const tx = d.target.x;
    const ty = d.target.y - nodeH/2;
    return bezierPath(sx, sy, tx, ty);
  });

// Edge-Labels (Ja/Nein) ungefähr auf der Mitte der Splits
g.append("g")
  .selectAll("text")
  .data(splitLinks)
  .join("text")
  .attr("class", "edge-label")
  .attr("x", d => (d.source.x + d.target.x)/2)
  .attr("y", d => (d.source.y + d.target.y)/2 - 6)
  .attr("text-anchor", "middle")
  .text(d => d.target.data._edge);

// ---------------- Knoten ----------------
const node = g.append("g")
  .selectAll("g")
  .data(root.descendants())
  .join("g")
  .attr("transform", d => "translate(" + d.x + "," + d.y + ")");

node.each(function(d) {
  const n = d3.select(this);

  // Leaf: nur Suitability-Box
  if (d.data.leaf !== undefined) {
    n.append("rect")
      .attr("x", -70)
      .attr("y", -24)
      .attr("width", 140)
      .attr("height", 48)
      .attr("rx", 8)
      .attr("fill", "#e6f5e6")
      .attr("stroke", "#999");

    const suit = d.data.suit;
    n.append("text")
      .attr("class", "label")
      .attr("dy", 4)
      .text("suit ≈ " + suit.toFixed(3));

    return;
  }

  // Decision Node mit Gradient
  const gradId = ensureGradientForNode(d.data);

  n.append("rect")
    .attr("x", -nodeW/2)
    .attr("y", -nodeH/2)
    .attr("width", nodeW)
    .attr("height", nodeH)
    .attr("rx", 20)
    .attr("fill", "url(#" + gradId + ")")
    .attr("stroke", "#333");

  // Threshold-Linie
  const txLocal = thresholdXLocal(d.data);
  n.append("line")
    .attr("x1", txLocal)
    .attr("x2", txLocal)
    .attr("y1", -nodeH/2)
    .attr("y2", nodeH/2)
    .attr("stroke", "black")
    .attr("stroke-width", 2);

  // Label oben
  n.append("text")
    .attr("class", "label")
    .attr("y", -nodeH/2 - 10)
    .text(d.data.label || d.data.feature);

  // Threshold-Wert unten
  if (d.data.threshold != null) {
    n.append("text")
      .attr("class", "label")
      .attr("y", nodeH/2 + 16)
      .text("Schwelle: " + d.data.threshold.toFixed(3));
  }
});

// ---------------- Suitability-Bar & Leaf-Linien ----------------
const leaves = root.leaves();

let maxYleaf = -Infinity;
leaves.forEach(d => { if (d.y > maxYleaf) maxYleaf = d.y; });

const barY = yOffset + maxYleaf + 200;
const barX = 120;
const barW = parseFloat(svg.attr("width")) - 2*barX;
const barH = 40;

// Viridis-Gradient
const gradSuit = defs.append("linearGradient")
  .attr("id", "gradSuit")
  .attr("x1", "0%").attr("x2", "100%")
  .attr("y1", "0%").attr("y2", "0%");

viridisColors.forEach((c,i) => {
  gradSuit.append("stop")
    .attr("offset", (i/(viridisColors.length-1))*100 + "%")
    .attr("stop-color", c);
});

svg.append("rect")
  .attr("x", barX)
  .attr("y", barY)
  .attr("width", barW)
  .attr("height", barH)
  .attr("fill", "url(#gradSuit)")
  .attr("stroke", "#333");

svg.append("text")
  .attr("x", barX)
  .attr("y", barY - 10)
  .text("Suitability (0 → 1, Viridis)");

// Range der Suitability-Werte aus den Leafs
const suitMin = d3.min(leaves, d => d.data.suit);
const suitMax = d3.max(leaves, d => d.data.suit);
const suitScale = d3.scaleLinear()
  .domain([0, 1])   // explizit 0–1
  .range([0, 1]);

// Leaf-Linien nach unten
svg.append("g")
  .attr("transform", "translate(" + xOffset + "," + yOffset + ")")
  .selectAll("path")
  .data(leaves)
  .join("path")
  .attr("fill", "none")
  .attr("stroke", "#b0b0b0")
  .attr("stroke-width", 1)
  .attr("d", d => {
    const sx = d.x;
    const sy = d.y + nodeH/2;

    let s = d.data.suit;
    if (!Number.isFinite(s)) s = 0.5;
    s = Math.max(0, Math.min(1, s));
    const rel = suitScale(s);

    const tx = barX + rel * barW - xOffset;
    const ty = barY - yOffset;

    return bezierPath(sx, sy, tx, ty);
  });

</script>
"""


# -------------------------------
# 4) HTML exportieren
# -------------------------------

def export_html(tree_dict, out_path: str):
    out_path = Path(out_path)
    tree_json = json.dumps(tree_dict, ensure_ascii=False)
    viridis_json = json.dumps(VIRIDIS_COLORS)

    html = (HTML_TEMPLATE
            .replace("REPLACE_TREE_JSON", tree_json)
            .replace("REPLACE_VIRIDIS_COLORS", viridis_json))

    out_path.write_text(html, encoding="utf-8")
    print(f"✓ HTML exportiert nach: {out_path}")


# -------------------------------
# MAIN
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Pfad zum XGBoost-JSON-Modell")
    parser.add_argument("--data", required=True, help="CSV mit Features")
    parser.add_argument("--out", default="surrogate_tree.html", help="HTML-Ausgabedatei")
    parser.add_argument("--depth", type=int, default=4, help="max_depth des Surrogate Trees")
    args = parser.parse_args()

    surrogate, featnames = train_surrogate(args.model, args.data, args.depth)
    tree_dict = tree_to_json(surrogate.tree_, featnames)
    export_html(tree_dict, args.out)

    print("=== DONE ===")


if __name__ == "__main__":
    main()