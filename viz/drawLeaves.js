// drawLeaves.js — FIXED

import { clearDecisionExplanation } from "./edgeExplanation.js";

export function drawLeaves({
  g,
  layout,
  viridis,
  highlightFromEdge,
  clearHighlight,
  showDecisionExplanation,
  edgeLabelLayer
}) {
  console.log("🟨 drawLeaves called");

  const { suitToY, barX, barW, edgeSegments } = layout;

  const scaleEdges = edgeSegments.filter(
    d => d.isScaleEdge || d.cls?.includes("scale")
  );

  console.log("🟩 scaleEdges:", scaleEdges);

  if (!scaleEdges.length) {
    console.warn("❌ no scaleEdges – abort drawLeaves");
    return;
  }

  const colorScale = d3.scaleLinear()
    .domain([0, 1])
    .range(viridis);

  // ===============================
  // Dreiecke
  // ===============================
  const triG = g.append("g").attr("class", "suit-triangles");

  triG.selectAll("path.leaf-tri")
    .data(scaleEdges)
    .join("path")
    .attr("class", d =>
      `leaf-tri leaf-${d.fromId} leaf-${d.toId.replace("suit_", "")}`
    )
    .attr("fill", d => colorScale(d.suit ?? 0.5))
    .attr("stroke", "#000")
    .attr("stroke-width", 0.5)
    .attr("d", d => {
      const s = Math.max(0, Math.min(1, d.suit ?? 0.5));
      const cy = suitToY(s);
      const cx = barX + barW / 2;
      const t = 8;
      return `M ${cx - t},${cy}
              L ${cx + t},${cy - t}
              L ${cx + t},${cy + t} Z`;
    })
    .on("mouseover", (_, d) => {
      highlightFromEdge(d);
      showDecisionExplanation({ edge: d, layer: edgeLabelLayer });
    })
    .on("mouseout", () => {
      clearHighlight();
      clearDecisionExplanation(edgeLabelLayer);
    });

  // ===============================
  // Zahlen
  // ===============================
  const labelG = g.append("g").attr("class", "suit-labels");

  labelG.selectAll("text.suit-label")
    .data(scaleEdges)
    .join("text")
    .attr("class", "suit-label")
    .attr("x", barX + barW + 6)
    .attr("y", d => suitToY(d.suit ?? 0.5) + 4)
    .attr("font-size", 11)
    .attr("fill", "#333")
    .text(d => (d.suit ?? 0.5).toFixed(3))
    .on("mouseover", (_, d) => {
      highlightFromEdge(d);
      showDecisionExplanation({ edge: d, layer: edgeLabelLayer });
    })
    .on("mouseout", () => {
      clearHighlight();
      clearDecisionExplanation(edgeLabelLayer);
    });
}