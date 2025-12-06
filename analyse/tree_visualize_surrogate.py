#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualisiert den Surrogate Tree als HTML/D3.
"""

import argparse
import json
from pathlib import Path


def export_html(tree_json, out_path):
    out_path = Path(out_path)

    # JSON als Text für D3
    tree_json_str = json.dumps(tree_json, ensure_ascii=False)

    # ----------------------------------------
    # ⚠️ Nur EIN f-string-Bereich – JSON wird eingesetzt.
    # Rest ist RAW STRING → JavaScript bleibt unberührt.
    # ----------------------------------------
    html = r"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Surrogate Tree – ML-Erklärung</title>

<style>
body {
    font-family: system-ui, sans-serif;
    margin: 0;
    padding: 1rem;
    background: #fafafa;
}
.label {
    font-size: 12px;
    text-anchor: middle;
}
.edge-label {
    fill: #666;
    font-size: 11px;
}
</style>

<svg id="svg"></svg>

<script src="https://d3js.org/d3.v7.min.js"></script>

<script>

// -----------------------------
// JSON-Daten werden unten eingesetzt
// -----------------------------
const treeData = TREE_JSON_DATA;

// Farbpaletten für Featuretypen
const NDVI  = ["#f2f2f2", "#a3c586", "#2f6b3a"];
const NDWI  = ["#f7fbff", "#6baed6", "#08519c"];
const MORAN = ["#fee8c8", "#fdbb84", "#e34a33"];
const GEARY = ["#f7f4f9", "#998ec3", "#542788"];

const VIRIDIS = [
    "#440154", "#482878","#3e4989","#31688e","#26828e",
    "#1f9e89","#35b779","#6ece58","#b5de2b","#fde725"
];

// Hilfsfunktionen
function sigmoid(x){ return 1/(1+Math.exp(-x)); }

function paletteForFeature(name){
    if(!name) return ["#eee","#ccc","#999"];
    const n = name.toLowerCase();

    if(n.includes("ndvi"))  return NDVI;
    if(n.includes("ndwi"))  return NDWI;
    if(n.includes("moran")) return MORAN;
    if(n.includes("geary")) return GEARY;

    return ["#eee","#ccc","#999"];
}

// SVG Setup
const svg = d3.select("#svg");
const dx = 260;
const dy = 200;
const nodeW = 260;
const nodeH = 70;
const padding = 200;

// Baumhierarchie
const root = d3.hierarchy(treeData, d => {
    const c=[];
    if(d.yes) c.push(Object.assign({_edge:"Ja"}, d.yes));
    if(d.no)  c.push(Object.assign({_edge:"Nein"},d.no));
    return c.length?c:null;
});

d3.tree().nodeSize([dx,dy])(root);

// Autosize SVG
let minX=1e9,maxX=-1e9, minY=1e9,maxY=-1e9;
root.each(d=>{
    minX=Math.min(minX,d.x);
    maxX=Math.max(maxX,d.x);
    minY=Math.min(minY,d.y);
    maxY=Math.max(maxY,d.y);
});

svg.attr("width",(maxX-minX)+2*padding);
svg.attr("height",(maxY-minY)+3*padding);

const g = svg.append("g")
    .attr("transform",`translate(${padding-minX},${padding-minY})`);

const defs = svg.append("defs");

// Startpunkt der Links an Split-Linie
function edgeStartX(node, side){
    const thr = node.data.threshold ?? 0.5;
    const x_t = -nodeW/2 + thr*nodeW;
    return side==="yes" ? x_t-6 : x_t+6;
}

// Bezier-Link
const link = d3.linkVertical()
    .x(d=>d.x)
    .y(d=>d.y);

// -----------------------------
// Kanten zeichnen
// -----------------------------
g.append("g")
 .selectAll("path")
 .data(root.links())
 .join("path")
 .attr("fill","none")
 .attr("stroke","#777")
 .attr("stroke-width",1.5)
 .attr("d", d => {
     const side = (d.target.data._edge==="Ja")?"yes":"no";
     const sx = edgeStartX(d.source, side);
     const sy = d.source.y;

     return link({
        source:{x:sx,y:sy},
        target:{x:d.target.x,y:d.target.y}
     });
 });

// Edge-Labels
g.append("g")
 .selectAll("text")
 .data(root.links())
 .join("text")
 .attr("class","edge-label")
 .attr("x", d=>(d.source.x+d.target.x)/2)
 .attr("y", d=>(d.source.y+d.target.y)/2 - 6)
 .text(d=>d.target.data._edge);

// -----------------------------
// Knoten
// -----------------------------
const node = g.append("g")
 .selectAll("g")
 .data(root.descendants())
 .join("g")
 .attr("transform", d=>`translate(${d.x},${d.y})`);

node.each(function(d){
    const n = d3.select(this);

    // -------- LEAF ----------
    if(d.data.leaf !== undefined){
        const suit = sigmoid(d.data.leaf);
        d.data.suit = suit;

        n.append("rect")
         .attr("x",-70).attr("y",-25)
         .attr("width",140).attr("height",50)
         .attr("rx",10)
         .attr("fill",VIRIDIS[Math.floor(suit*(VIRIDIS.length-1))])
         .attr("stroke","#333");

        n.append("text")
         .attr("class","label")
         .attr("dy",4)
         .text(`suit = ${suit.toFixed(3)}`);

        return;
    }

    // -------- SPLIT NODE ----------
    const pal = paletteForFeature(d.data.feature);

    const gradId = "grad_"+Math.random().toString(36).slice(2);
    const grad = defs.append("linearGradient")
        .attr("id",gradId)
        .attr("x1","0%").attr("x2","100%")
        .attr("y1","0%").attr("y2","0%");

    pal.forEach((c,i)=>{
        grad.append("stop")
            .attr("offset",(i/(pal.length-1))*100+"%")
            .attr("stop-color",c);
    });

    n.append("rect")
     .attr("x",-nodeW/2).attr("y",-nodeH/2)
     .attr("width",nodeW).attr("height",nodeH)
     .attr("rx",12)
     .attr("fill",`url(#${gradId})`)
     .attr("stroke","#333");

    n.append("text")
     .attr("class","label")
     .attr("y",-nodeH/2 - 10)
     .text(d.data.feature);

    n.append("text")
     .attr("class","label")
     .attr("y",nodeH/2 + 14)
     .text(`Schwelle: ${d.data.threshold.toFixed(3)}`);

    // split-line
    const rel = d.data.threshold;
    const tX  = -nodeW/2 + rel*nodeW;

    n.append("line")
     .attr("x1",tX).attr("x2",tX)
     .attr("y1",-nodeH/2).attr("y2",nodeH/2)
     .attr("stroke","black").attr("stroke-width",2);
});

// -----------------------------
// SUITABILITY-SKALA
// -----------------------------
const leaves = root.leaves();

const barY = maxY + 160;
const barX = minX + 80;
const barW = (maxX-minX) - 160;
const barH = 35;

const gradSuit = defs.append("linearGradient")
 .attr("id","gradSuit")
 .attr("x1","0%").attr("x2","100%")
 .attr("y1","0%").attr("y2","0%");

VIRIDIS.forEach((c,i)=>{
    gradSuit.append("stop")
     .attr("offset",(i/(VIRIDIS.length-1))*100+"%")
     .attr("stop-color",c);
});

g.append("rect")
 .attr("x",barX)
 .attr("y",barY)
 .attr("width",barW)
 .attr("height",barH)
 .attr("fill","url(#gradSuit)")
 .attr("stroke","#333");

g.append("text")
 .attr("x",barX)
 .attr("y",barY - 12)
 .text("Suitability (0 → 1, Viridis)");

// -----------------------------
// LEAF → BAR Verbindungen
// -----------------------------
leaves.forEach(d=>{
    const sx = d.x;
    const sy = d.y + 35;

    const tx = barX + d.data.suit * barW;
    const ty = barY;

    g.append("path")
     .attr("fill","none")
     .attr("stroke","#aaa")
     .attr("stroke-width",1.2)
     .attr("d",`
        M ${sx},${sy}
        C ${sx},${(sy+ty)/2}  ${tx},${(sy+ty)/2}  ${tx},${ty}
     `);
});

</script>
"""

    # JSON einsetzen
    html = html.replace("TREE_JSON_DATA", tree_json_str)

    out_path.write_text(html, encoding="utf-8")
    print(f"✓ Surrogate Tree HTML exportiert nach: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", default="surrogate_tree.html")
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text())
    export_html(data, args.out)


if __name__ == "__main__":
    main()