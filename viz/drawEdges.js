// drawEdges.js — FINAL
// Verantwortlich NUR für:
// - Edge-Zeichnung
// - Edge-Hover → Highlight + Erklärung DIESER Edge

import {
  showDecisionExplanation,
  clearDecisionExplanation
} from "./edgeExplanation.js";

export function drawEdges({
  g,
  layout,
  highlightFromEdge,
  clearHighlight,
  edgeLabelLayer
}) {
  const edgeG = g.append("g").attr("class", "edges");
  const hitG  = g.append("g").attr("class", "edge-hits");

  // ----------------------------------------------
  // Pfad-Generator
  // ----------------------------------------------
  const pathD = (d) => {
    const { x: x0, y: y0 } = d.source;
    const { x: x1, y: y1 } = d.target;

    if (![x0, y0, x1, y1].every(Number.isFinite)) return "";

    const dx = x1 - x0;
    const c  = 0.35;

    return `
      M ${x0},${y0}
      C ${x0 + dx * c},${y0}
        ${x1 - dx * c},${y1}
        ${x1},${y1}
    `;
  };

  // ==================================================
  // 1) Sichtbare Edges
  // ==================================================
  edgeG
    .selectAll("path.edge")
    .data(layout.edgeSegments)
    .join("path")
    .attr("class", d => `edge ${d.cls ?? ""}`)
    .attr("fill", "none")
    .attr("d", pathD);

// ==================================================
// 2) Hit-Area (Interaktion)
// ==================================================
hitG
  .selectAll("path.edge-hit")
  .data(layout.edgeSegments)
  .join("path")
  .attr("class", "edge-hit")
  .attr("fill", "none")
  .attr("stroke", "transparent")
  .attr("stroke-width", 18)
  .attr("pointer-events", "stroke")
  .attr("d", pathD)
  .on("mouseover", (_, edge) => {
    // 1) Visuelles Highlight
    highlightFromEdge(edge);

    // 2) Entscheidung anzeigen (vom Ziel-Node gelesen)
    showDecisionExplanation({
      edge,
      nodeById: layout.nodeById,   // 🔑 wichtig
      layer: edgeLabelLayer
    });
  })
  .on("mouseout", () => {
    clearHighlight();
    clearDecisionExplanation(edgeLabelLayer);
  });
}