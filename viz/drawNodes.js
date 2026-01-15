// drawNodes.js — FINAL (klar, minimal, konsistent)
//
// Verantwortlich NUR für:
// - Node-Rendering
// - Node-Hover → Highlight + Entscheidung DIESES Nodes

import {
  showDecisionExplanation,
  clearDecisionExplanation
} from "./edgeExplanation.js";

export function drawNodes({
  g,
  root,
  layout,
  config,
  gradients,
  highlightFromNode,
  clearHighlight,
  edgeLabelLayer
}) {

  // ==================================================
  // 1) Hilfsfunktion: Label umbrechen
  // ==================================================
  function splitLabel(label) {
    if (!label) return [];
    const match = label.match(/^(.+?)\s*\((.+)\)$/);
    if (match) return [match[1], `(${match[2]})`];
    return [label];
  }

  // ==================================================
  // 2) Node-Gruppen + INTERAKTION
  // ==================================================
  const nodeG = g
    .selectAll("g.node")
    .data(root.descendants().filter(d => !d.data.leaf))
    .join("g")
    .attr("class", "node")
    .attr("transform", d => `translate(${d.x},${d.y})`)
    .on("mouseover", (_, d) => {
      // 🔦 visuelles Highlight
      highlightFromNode(d.data._nid);

      // 🧠 Entscheidung = dieser Node selbst
      showDecisionExplanation({
        node: d,
        nodeById: layout.nodeById,
        edgeSegments: layout.edgeSegments,
        layer: edgeLabelLayer
      });
    })
    .on("mouseout", () => {
      clearHighlight();
      clearDecisionExplanation(edgeLabelLayer);
    });

  // ==================================================
  // 3) ClipPath pro Node
  // ==================================================
  nodeG.append("clipPath")
    .attr("id", d => `clip-node-${d.data._nid}`)
    .append("rect")
      .attr("x", -config.nodeW / 2)
      .attr("y", -config.nodeH / 2)
      .attr("width", config.nodeW)
      .attr("height", config.nodeH)
      .attr("rx", 6);

  // ==================================================
  // 4) PNG-Layer (Moran / Geary)
  // ==================================================
  nodeG.append("image")
    .attr("href", d => {
      const f = (d.data.feature || "").toLowerCase();

      const isMean  = f.includes("mean");
      const isNDVI  = f.includes("ndvi");
      const isNDWI  = f.includes("ndwi");
      const isMoran = f.includes("moran");
      const isGeary = f.includes("geary");

      if (isMean) return null;

      if ((isMoran || isGeary) && (isNDVI || isNDWI)) {
        const type = isMoran ? "moran" : "geary";
        const band = isNDVI ? "ndvi" : "ndwi";
        return `patterns/${type}_${band}.png`;
      }

      return null;
    })
    .attr("x", -config.nodeW / 2)
    .attr("y", -config.nodeH / 2)
    .attr("width", config.nodeW)
    .attr("height", config.nodeH)
    .attr("clip-path", d => `url(#clip-node-${d.data._nid})`)
    .style("pointer-events", "none");

  // ==================================================
  // 5) Gradient-Layer (Mean)
  // ==================================================
  nodeG.append("rect")
    .attr("x", -config.nodeW / 2)
    .attr("y", -config.nodeH / 2)
    .attr("width", config.nodeW)
    .attr("height", config.nodeH)
    .attr("rx", 6)
    .attr("fill", d => {
      const f = (d.data.feature || "").toLowerCase();
      if (f.includes("ndvi") && f.includes("mean")) return "url(#gradient-ndvi)";
      if (f.includes("ndwi") && f.includes("mean")) return "url(#gradient-ndwi)";
      return "none";
    });

  // ==================================================
  // 6) Threshold-Linie
  // ==================================================
  const THRESHOLD_OVERHANG = 4;

  nodeG
    .filter(d => typeof d.data.threshold === "number")
    .append("line")
      .attr("class", "node-threshold-line")
      .attr("x1", -config.nodeW / 2 - THRESHOLD_OVERHANG)
      .attr("x2",  config.nodeW / 2 + THRESHOLD_OVERHANG)
      .attr("y1", d =>
        (1 - d.data.threshold) * config.nodeH - config.nodeH / 2
      )
      .attr("y2", d =>
        (1 - d.data.threshold) * config.nodeH - config.nodeH / 2
      )
      .attr("pointer-events", "none");

  // ==================================================
  // 7) Threshold-Zahl
  // ==================================================
  nodeG
    .filter(d => typeof d.data.threshold === "number")
    .append("text")
      .attr("class", "node-threshold-label")
      .attr("x", config.nodeW / 2 + 6)
      .attr("y", d =>
        (1 - d.data.threshold) * config.nodeH - config.nodeH / 2
      )
      .text(d => d.data.threshold.toFixed(2))
      .attr("dominant-baseline", "middle")
      .attr("pointer-events", "none");

  // ==================================================
  // 8) Labels rechts außerhalb
  // ==================================================
  nodeG.each(function(d) {
    if (!d.data.label) return;

    const lines = splitLabel(d.data.label);

    const gLabel = d3.select(this)
      .append("g")
      .attr("class", `node-label-outside node-label-${d.data._nid}`)
      .attr(
        "transform",
        `translate(${config.nodeW / 2 + 12}, ${-config.nodeH / 2 + 14})`
      );

    lines.forEach((line, i) => {
      gLabel.append("text")
        .attr("x", 0)
        .attr("y", i * 14)
        .text(line)
        .attr(
          "class",
          `${i === 0 ? "node-label-main" : "node-label-sub"} node-label-${d.data._nid}`
        )
        .attr("pointer-events", "none");
    });
  });

  // ==================================================
  // 9) Rahmen + Hit-Area
  // ==================================================
  nodeG.append("rect")
    .attr("x", -config.nodeW / 2)
    .attr("y", -config.nodeH / 2)
    .attr("width", config.nodeW)
    .attr("height", config.nodeH)
    .attr("rx", 6)
    .attr("fill", "transparent")
    .attr("stroke", "#333")
    .attr("pointer-events", "all");
}