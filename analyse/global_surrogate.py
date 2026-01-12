#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor



# ---------------------------------------------------------
# 1) Surrogate Tree trainieren
# ---------------------------------------------------------

def train_surrogate(model_path, data_path, max_depth=4):
    print("→ Lade XGBoost-Modell…")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    booster = model.get_booster()
    feature_names = booster.feature_names
    print(f"→ Modell hat {len(feature_names)} Features")

    print("→ Lade Daten…")
    df = pd.read_csv(data_path)

    X = df[feature_names].copy()

    # numerische Typen erzwingen
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype(float)

    print("→ Berechne Modell-Predictions…")
    y_pred = model.predict_proba(X)[:, 1]

    surrogate = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=50,
        random_state=42
    )

    print("→ Trainiere Surrogate Tree…")
    surrogate.fit(X, y_pred)

    return surrogate, feature_names



# ---------------------------------------------------------
# 2) Baum → JSON
# ---------------------------------------------------------

def tree_to_json(tree, feature_names):

    def recurse(node_id):

        # Leaf?
        if tree.children_left[node_id] == -1:
            return {"leaf": float(tree.value[node_id][0][0])}

        feat = feature_names[tree.feature[node_id]]
        thr = float(tree.threshold[node_id])

        return {
            "feature": feat,
            "threshold": thr,
            "yes": recurse(tree.children_left[node_id]),
            "no": recurse(tree.children_right[node_id])
        }

    return recurse(0)



# ---------------------------------------------------------
# 3) HTML export ohne f-Strings
# ---------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<meta charset="utf-8">
<title>Global Surrogate Tree</title>
<style>
body {{
  font-family: system-ui, sans-serif;
  margin: 0; padding: 1rem;
}}
.label {{
  font-size: 12px;
  text-anchor: middle;
}}
.edge-label {{
  fill: #444;
  font-size: 11px;
}}
</style>

<svg id="tree-svg"></svg>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>

const treeData = __TREE_JSON__;

const svg = d3.select("#tree-svg");
const dx = 220, dy = 180;
const nodeW = 240, nodeH = 70;

const root = d3.hierarchy(treeData, d => {{
  const c = [];
  if (d.yes) c.push(Object.assign({{_edge:"Ja"}}, d.yes));
  if (d.no)  c.push(Object.assign({{_edge:"Nein"}}, d.no));
  return c.length ? c : null;
}});

d3.tree().nodeSize([dx, dy])(root);

// SVG Größe bestimmen
let minX=1e9, maxX=-1e9, minY=1e9, maxY=-1e9;
root.each(d => {{
  if (d.x < minX) minX = d.x;
  if (d.x > maxX) maxX = d.x;
  if (d.y < minY) minY = d.y;
  if (d.y > maxY) maxY = d.y;
}});

const padding = 200;
svg.attr("width",(maxX-minX)+2*padding);
svg.attr("height",(maxY-minY)+3*padding);

const g = svg.append("g")
  .attr("transform","translate("+(padding-minX)+","+(padding-minY)+")");

// Links
g.append("g")
 .selectAll("path")
 .data(root.links())
 .join("path")
 .attr("fill","none")
 .attr("stroke","#444")
 .attr("stroke-width",1.5)
 .attr("d", d3.linkVertical().x(d=>d.x).y(d=>d.y));

// Edge labels
g.append("g")
 .selectAll("text")
 .data(root.links())
 .join("text")
 .attr("class","edge-label")
 .attr("x", d => (d.source.x+d.target.x)/2)
 .attr("y", d => (d.source.y+d.target.y)/2 - 4)
 .text(d => d.target.data._edge);

// Nodes
const node = g.append("g")
 .selectAll("g")
 .data(root.descendants())
 .join("g")
 .attr("transform", d => `translate(${d.x},${d.y})`);

node.each(function(d){{
  const n = d3.select(this);

  if (d.data.leaf !== undefined) {{
    n.append("rect")
     .attr("x",-90).attr("y",-25)
     .attr("width",180).attr("height",50)
     .attr("rx",10).attr("fill","#fdfcd4").attr("stroke","#888");
    n.append("text").attr("class","label").attr("dy",4)
     .text("Leaf: "+d.data.leaf.toFixed(3));
    return;
  }}

  n.append("rect")
   .attr("x",-nodeW/2).attr("y",-nodeH/2)
   .attr("width",nodeW).attr("height",nodeH)
   .attr("rx",10).attr("fill","#eee").attr("stroke","#333");

  n.append("text").attr("class","label")
   .attr("y",-nodeH/2 - 8)
   .text(d.data.feature);

  n.append("text").attr("class","label")
   .attr("y",nodeH/2 + 14)
   .text("Threshold: "+d.data.threshold.toFixed(3));
}});

</script>
"""



def export_html(tree_json_path, out_html):
    tree_text = Path(tree_json_path).read_text()

    html = HTML_TEMPLATE.replace("__TREE_JSON__", tree_text)

    Path(out_html).write_text(html, encoding="utf-8")
    print(f"✓ HTML gespeichert: {out_html}")



# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="surrogate_tree.html")
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()

    surrogate, feature_names = train_surrogate(args.model, args.data, args.depth)

    print("→ Konvertiere Baum nach JSON…")
    tree_json = tree_to_json(surrogate.tree_, feature_names)

    json_path = Path(args.out).with_suffix(".json")
    json_path.write_text(json.dumps(tree_json, indent=2), encoding="utf-8")
    print(f"✓ JSON gespeichert: {json_path}")

    export_html(json_path, args.out)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()