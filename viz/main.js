// main.js — DEBUG VERSION (clean, single source wiring)

import { setupSvg } from "./setupSvg.js";
import { buildLayout } from "./layout.js";
import { makeGradients } from "./makeGradients.js";
import { makeNodeGradients } from "./makeNodePatterns.js";
import { drawEdges } from "./drawEdges.js";
import { drawNodes } from "./drawNodes.js";
import { drawScale } from "./drawScale.js";
import { drawLeaves } from "./drawLeaves.js";
import { setupHighlighting } from "./highlighting.js";
import { config, viridis } from "./constants.js";
import {
  showDecisionExplanation,
  clearDecisionExplanation
} from "./edgeExplanation.js";

console.log("✅ main.js geladen (DEBUG)");

window.addEventListener("DOMContentLoaded", async () => {
  console.time("⏱️ Rendering abgeschlossen in");

  // ==================================================
  // 1️⃣ Daten laden
  // ==================================================
  const res = await fetch("./surrogate_tree.json");
  const treeData = await res.json();

  // ==================================================
  // 2️⃣ Layout berechnen
  // ==================================================
  const layout = buildLayout(treeData, config);
  window.__layout = layout; // 🔥 DEBUG ONLY

  // ==================================================
  // 3️⃣ SVG Setup
  // ==================================================
  const { svg, g, defs } = setupSvg(d3.select("#tree-svg"));

  svg
    .attr(
      "viewBox",
      `0 0 ${layout.maxX + 3 * config.PAD} ${(layout.maxY - layout.minX) + 2 * config.PAD}`
    )
    .attr("preserveAspectRatio", "xMinYMid meet");

  g.attr(
    "transform",
    `translate(${config.PAD}, ${config.PAD - layout.minX})`
  );

  // ==================================================
  // 4️⃣ Gradients & Patterns
  // ==================================================
  const [suitGradientId] = makeGradients(defs, [viridis]);
  const gradients = { suit: suitGradientId };

  makeNodeGradients(defs);

  // ==================================================
  // 5️⃣ ZENTRALER Edge-Label-Layer (Single Source)
  // ==================================================
  const edgeLabelLayer = g
    .append("g")
    .attr("class", "edge-label-layer");

  // ==================================================
  // 6️⃣ Highlighting (LOGIK – MUSS ZUERST KOMMEN)
  // ==================================================
  const {
    highlightFromNode,
    highlightFromEdge,
    clearHighlight
  } = setupHighlighting(layout.root, layout.edgeSegments);

  // ==================================================
  // 7️⃣ Zeichnen (GENAU EINMAL)
  // ==================================================
  drawEdges({
    g,
    layout,
    highlightFromEdge,
    clearHighlight,
    edgeLabelLayer
  });

  drawNodes({
    g,
    root: layout.root,
    layout,
    config,
    gradients,
    highlightFromNode,
    clearHighlight,
    edgeLabelLayer
  });

  drawScale({
    g,
    layout,
    config,
    gradientId: suitGradientId
  });

  drawLeaves({
    g,
    layout,
    viridis,
    highlightFromEdge,
    clearHighlight,
    showDecisionExplanation,
    edgeLabelLayer
  });

  // ==================================================
  // 8️⃣ Done
  // ==================================================
  console.timeEnd("⏱️ Rendering abgeschlossen in");
  console.log("🌳 Baum korrekt gerendert (DEBUG)");
});