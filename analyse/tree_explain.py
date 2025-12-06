#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tree_explain.py – erzeugt eine D3-Visualisierung eines XGBoost-Baumes
mit:

- semantischen Feature-Knoten (NDVI/NDWI/Moran/Geary, mit Farbverlauf & Range)
- neutralen Leaf-Knoten (nur Text, keine Suitability-Farbe)
- globaler Viridis-Skala (0–1) am unteren Rand
- hervorgehobenem Pfad: Leaf mit größtem Leaf-Wert
- Verbindung dieses Leafs mit einem Punkt auf der Skala:
    suit = sigmoid(Leaf)

Wichtig: Das ist eine *bauminterne* „Suitability“, kein Gesamtmodell-Output.
"""

import json
import argparse
from pathlib import Path

# ---------------------------------------------------------
# Farbpaletten & Ranges für Features
# ---------------------------------------------------------

NDVI_PALETTE = ["#f2f2f2", "#a3c586", "#2f6b3a"]
NDWI_PALETTE = ["#f7fbff", "#6baed6", "#08519c"]
MORAN_PALETTE = ["#fee8c8", "#fdbb84", "#e34a33"]
GEARY_PALETTE = ["#f7f4f9", "#998ec3", "#542788"]

RANGES = {
    "ndvi_mean": [0.0, 1.0],
    "ndwi_mean": [-1.0, 1.0],
    "moran": [-0.2, 4.0],
    "geary": [0.0, 1.5],
}

# Viridis-Skala für Suitability (0–1)
VIRIDIS = [
    "#440154", "#482475", "#414487", "#355F8D", "#2A788E",
    "#21918C", "#22A884", "#44BF70", "#7AD151", "#BDDF26", "#FDE725"
]

# ---------------------------------------------------------
# 1) XGBoost-Dump laden
# ---------------------------------------------------------

def load_xgb_tree(dump_json_path, tree_index=0):
    """
    Erwartet einen JSON-Dump im Format:
    [
      {"nodeid": 0, "depth": 0, "split": ..., "children": [...]},
      ...
    ]
    """
    print("=== LOAD_XGB_TREE ===")
    dump_json_path = Path(dump_json_path)
    print(f"→ Lade Dump: {dump_json_path}")

    with dump_json_path.open("r") as f:
        dump = json.load(f)

    if not isinstance(dump, list):
        raise ValueError("❌ Dump ist kein list-Format – erwarte Liste von Bäumen.")

    print(f"✓ Dump enthält {len(dump)} Bäume")

    if tree_index < 0 or tree_index >= len(dump):
        raise IndexError(f"Tree-Index {tree_index} außerhalb des gültigen Bereichs.")

    tree = dump[tree_index]
    print(f"✓ Nutze Baum Nr. {tree_index}, root nodeid = {tree.get('nodeid')}")
    return tree

# ---------------------------------------------------------
# 2) XGBoost-Knoten → generischer Entscheidungsbaum
# ---------------------------------------------------------

def convert_xgb_tree(node):
    """
    Konvertiert XGBoost-Node-Struktur in:
      - Leaf:   {"leaf": float}
      - Split:  {"feature": str, "threshold": float, "yes": {...}, "no": {...}}
    """

    # Leaf
    if "leaf" in node:
        leaf_val = float(node["leaf"])
        print(f"Leaf erkannt: {leaf_val}")
        return {"leaf": leaf_val}

    # Split
    children = node.get("children", [])
    if len(children) != 2:
        print(f"⚠ WARNUNG: Node {node.get('nodeid')} hat {len(children)} Kinder → als Leaf degradiert")
        return {"leaf": 0.0}

    return {
        "feature": node["split"],
        "threshold": float(node["split_condition"]),
        "yes": convert_xgb_tree(children[0]),
        "no":  convert_xgb_tree(children[1]),
    }

# ---------------------------------------------------------
# 3) Feature-Semantik (Labels, Paletten, Ranges)
# ---------------------------------------------------------

def infer_semantics(raw_feature: str):
    """
    Erwartet Feature-Namen wie:
      m07_ndvi_mean
      m12_ndwi_mean
      m10_moran_ndvi
      m09_geary_ndwi
    und leitet Label, Palette, Range ab.
    """
    print(f"→ Semantik für Feature: {raw_feature}")

    if not raw_feature.startswith("m") or "_" not in raw_feature:
        # Fallback
        return {
            "label": raw_feature,
            "palette": ["#dddddd", "#aaaaaa", "#666666"],
            "range": [0.0, 1.0],
        }

    try:
        month = int(raw_feature[1:3])
        rest = raw_feature[4:]  # Teil nach 'mXX_'
    except Exception:
        return {
            "label": raw_feature,
            "palette": ["#dddddd", "#aaaaaa", "#666666"],
            "range": [0.0, 1.0],
        }

    # Saison
    if month in (7, 8, 9):
        season = "Sommer"
    elif month in (10, 11, 12):
        season = "Herbst"
    else:
        season = "Saison"

    # Defaults
    label = raw_feature
    palette = ["#dddddd", "#aaaaaa", "#666666"]
    rng = [0.0, 1.0]

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

def enrich_tree(node):
    """
    Fügt jedem Split-Knoten:
      - label
      - palette
      - range
    hinzu.
    Leafs bleiben unverändert.
    """
    if "leaf" in node:
        return node

    info = infer_semantics(node["feature"])
    node.update(info)
    node["yes"] = enrich_tree(node["yes"])
    node["no"]  = enrich_tree(node["no"])
    return node

# ---------------------------------------------------------
# 4) HTML mit D3.js erzeugen (inkl. Viridis-Skala + Pfad)
# ---------------------------------------------------------

def export_tree_html(tree_data, out_path="decision_tree_vertical.html"):
    """
    Schreibt ein eigenständiges HTML mit eingebettetem D3-Code.
    tree_data: bereits angereicherter Baum (label, palette, range, yes/no/leaf).
    """
    print("=== EXPORT_HTML ===")
    out_path = Path(out_path)
    print(f"→ Schreibe HTML nach: {out_path}")

    tree_json = json.dumps(tree_data, ensure_ascii=False)
    viridis_json = json.dumps(VIRIDIS)

    html = """<!DOCTYPE html>
<meta charset="utf-8">
<title>Decision Tree – ML-Erklärung</title>
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
// ---------------- Daten ----------------
const treeData = """ + tree_json + """;
const suitabilityPalette = """ + viridis_json + """;

// ----------- Setup: SVG dynamisch -----------

const svg = d3.select("#tree-svg");

// Knotenabstände
const dx = 260;   // horizontaler Abstand
const dy = 240;   // vertikaler Abstand
const nodeW = 240;
const nodeH = 70;

// Hierarchie aufbauen (yes/no → Kinder)
const root = d3.hierarchy(treeData, d => {
  const c = [];
  if (d.yes) c.push(Object.assign({ "_edge": "Ja" }, d.yes));
  if (d.no)  c.push(Object.assign({ "_edge": "Nein" }, d.no));
  return c.length ? c : null;
});

// D3-Layout (x = horizontal, y = vertikal)
d3.tree().nodeSize([dx, dy])(root);

// Dynamische SVG-Größe bestimmen
let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
root.each(d => {
  if (d.x < minX) minX = d.x;
  if (d.x > maxX) maxX = d.x;
  if (d.y < minY) minY = d.y;
  if (d.y > maxY) maxY = d.y;
});

const padding = 300;

// SVG final setzen
svg.attr("width",  (maxX - minX) + 2 * padding);
svg.attr("height", (maxY - minY) + 3 * padding);

// Verschiebung, damit alles im Bild liegt
const xOffset = padding - minX;
const yOffset = padding - minY;

// Hauptgruppe
const g = svg.append("g")
  .attr("transform", `translate(${xOffset}, ${yOffset})`);

// ---------------- Links ----------------
const links = root.links();

g.append("g")
  .selectAll("path")
  .data(links)
  .join("path")
    .attr("fill", "none")
    .attr("stroke", "#444")
    .attr("stroke-width", 1.5)
    .attr("d", d3.linkVertical()
      .x(d => d.x)
      .y(d => d.y)
    );

// ---------------- Edge-Labels (Ja/Nein) ----------------
g.append("g")
  .selectAll("text")
  .data(links)
  .join("text")
    .attr("class", "edge-label")
    .attr("x", d => (d.source.x + d.target.x) / 2)
    .attr("y", d => (d.source.y + d.target.y) / 2 - 4)
    .attr("text-anchor", "middle")
    .text(d => d.target.data._edge || "");

// ---------------- Knoten ----------------
const defs = svg.append("defs");

function thresholdX(d) {
  const data = d.data;
  if (!data.range || data.threshold == null) return 0;
  const min = data.range[0];
  const max = data.range[1];
  const t   = data.threshold;
  const rel = Math.max(0, Math.min(1, (t - min) / (max - min)));
  return -nodeW/2 + rel * nodeW;
}

const node = g.append("g")
  .selectAll("g")
  .data(root.descendants())
  .join("g")
    .attr("transform", d => `translate(${d.x},${d.y})`);

node.each(function(d){
  const gNode = d3.select(this);

  // Leaf-Knoten: neutraler Kasten mit Text, KEINE Suitability-Farbe
  if (d.data.leaf !== undefined) {
    gNode.append("rect")
      .attr("x", -90)
      .attr("y", -20)
      .attr("width", 180)
      .attr("height", 40)
      .attr("rx", 8)
      .attr("fill", "#ffffff")
      .attr("stroke", "#999");

    gNode.append("text")
      .attr("class", "label")
      .attr("dy", 4)
      .text("Leaf: " + d.data.leaf.toFixed(3));

    return;
  }

  // Decision-Knoten: Feature-Verlauf als Gradient
  const pal = d.data.palette || ["#dddddd", "#aaaaaa", "#666666"];
  const gradId = "grad_" + Math.random().toString(36).slice(2);

  const grad = defs.append("linearGradient")
    .attr("id", gradId)
    .attr("x1", "0%").attr("x2", "100%")
    .attr("y1", "0%").attr("y2", "0%");

  // 3-Stützfarben → kontinuierlicher Verlauf
  pal.forEach((c, i) => {
    grad.append("stop")
      .attr("offset", (i / (pal.length - 1)) * 100 + "%")
      .attr("stop-color", c);
  });

  // Hintergrund-Rechteck des Knotens
  gNode.append("rect")
    .attr("x", -nodeW / 2)
    .attr("y", -nodeH / 2)
    .attr("width", nodeW)
    .attr("height", nodeH)
    .attr("rx", 10)
    .attr("fill", "url(#" + gradId + ")")
    .attr("stroke", "#333")
    .attr("stroke-width", 1);

  // Threshold-Linie
  const tx = thresholdX(d);
  gNode.append("line")
    .attr("x1", tx)
    .attr("x2", tx)
    .attr("y1", -nodeH / 2)
    .attr("y2", nodeH / 2)
    .attr("stroke", "black")
    .attr("stroke-width", 2);

  // Label oben
  gNode.append("text")
    .attr("class", "label")
    .attr("y", -nodeH / 2 - 8)
    .text(d.data.label || d.data.feature);

  // Threshold-Wert unten
  if (d.data.threshold != null) {
    gNode.append("text")
      .attr("class", "label")
      .attr("y", nodeH / 2 + 14)
      .text("Schwelle: " + d.data.threshold.toFixed(3));
  }
});

// --------------------------------------------------
// Globale Viridis-Skala (0–1) für „bauminterne Suitability“
// --------------------------------------------------
const leaves = root.leaves();
let maxY_leaf = -Infinity;
leaves.forEach(d => { if (d.y > maxY_leaf) maxY_leaf = d.y; });

// Position der Skala unterhalb der Leaves
const barY = yOffset + maxY_leaf + 200;
const barX = 120;
const barW = svg.attr("width") - 240;
const barH = 40;

// Viridis-Gradient
const gradSuit = defs.append("linearGradient")
  .attr("id", "gradSuit")
  .attr("x1", "0%").attr("x2", "100%")
  .attr("y1", "0%").attr("y2", "0%");

suitabilityPalette.forEach((c, i) => {
  gradSuit.append("stop")
    .attr("offset", (i / (suitabilityPalette.length - 1)) * 100 + "%")
    .attr("stop-color", c);
});

// Rechteck für Skala
svg.append("rect")
  .attr("x", barX)
  .attr("y", barY)
  .attr("width", barW)
  .attr("height", barH)
  .attr("fill", "url(#gradSuit)")
  .attr("stroke", "#333")
  .attr("stroke-width", 1);

// Achsenbeschriftung
svg.append("text")
  .attr("x", barX)
  .attr("y", barY - 10)
  .text("Bauminterne Suitability (sigmoid(Leaf))");

svg.append("text")
  .attr("x", barX)
  .attr("y", barY + barH + 14)
  .text("0.0");

svg.append("text")
  .attr("x", barX + barW)
  .attr("y", barY + barH + 14)
  .attr("text-anchor", "end")
  .text("1.0");

// --------------------------------------------------
// Pfad-Auswahl & Verbindung: Leaf mit maximalem Leaf-Wert
// --------------------------------------------------

// 1) Leaf mit maximalem Leaf-Wert
let highlightLeaf = null;
let maxLeafVal = -Infinity;
leaves.forEach(d => {
  if (d.data.leaf > maxLeafVal) {
    maxLeafVal = d.data.leaf;
    highlightLeaf = d;
  }
});

// 2) Sigmoid-Mapping: leaf → p in [0,1]
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

let suit = 0.5;
if (highlightLeaf) {
  suit = sigmoid(highlightLeaf.data.leaf);
}

// 3) Pfad-Knoten: root → highlightLeaf
const pathNodes = new Set();
let cur = highlightLeaf;
while (cur) {
  pathNodes.add(cur);
  cur = cur.parent;
}

// 4) Overlay-Links für den Pfad
g.append("g")
  .selectAll("path.highlight-link")
  .data(links.filter(l => pathNodes.has(l.source) && pathNodes.has(l.target)))
  .join("path")
    .attr("class", "highlight-link")
    .attr("fill", "none")
    .attr("stroke", "black")
    .attr("stroke-width", 3)
    .attr("d", d3.linkVertical()
      .x(d => d.x)
      .y(d => d.y)
    );

// 5) Marker auf der Suitability-Skala
const markerX = barX + suit * barW;

svg.append("line")
  .attr("x1", markerX)
  .attr("x2", markerX)
  .attr("y1", barY - 10)
  .attr("y2", barY + barH + 10)
  .attr("stroke", "black")
  .attr("stroke-width", 2);

svg.append("text")
  .attr("x", markerX)
  .attr("y", barY + barH + 28)
  .attr("text-anchor", "middle")
  .text("sigmoid(Leaf) ≈ " + suit.toFixed(2));

// 6) Verbindungslinie vom Leaf zur Skala
if (highlightLeaf) {
  const leafX_svg = xOffset + highlightLeaf.x;
  const leafY_svg = yOffset + highlightLeaf.y + nodeH / 2;

  svg.append("path")
    .attr("d", `
      M ${leafX_svg} ${leafY_svg}
      L ${leafX_svg} ${barY - 40}
      L ${markerX} ${barY}
    `)
    .attr("fill", "none")
    .attr("stroke", "black")
    .attr("stroke-width", 1.5)
    .attr("stroke-dasharray", "4,3");
}

</script>
"""

    out_path.write_text(html, encoding="utf-8")
    print("✓ HTML gespeichert.")


# ---------------------------------------------------------
# 5) Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump", required=True,
        help="Pfad zum XGBoost JSON-Dump (dump_format='json')"
    )
    parser.add_argument(
        "--out", default="decision_tree_vertical.html",
        help="Pfad zur HTML-Ausgabe"
    )
    parser.add_argument(
        "--tree-index", type=int, default=0,
        help="Index des Baums im Dump (default: 0)"
    )
    args = parser.parse_args()

    raw = load_xgb_tree(args.dump, tree_index=args.tree_index)
    clean = convert_xgb_tree(raw)
    enriched = enrich_tree(clean)
    export_tree_html(enriched, args.out)

    print("=== DONE ===")


if __name__ == "__main__":
    main()