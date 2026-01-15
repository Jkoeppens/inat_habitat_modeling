// edgeExplanation.js – DEBUG-AUSGABE

// ======================================
// Label-Mappings
// ======================================
const FEATURE_LABELS = {
  ndvi: "NDVI",
  ndwi: "NDWI"
};

const STAT_LABELS = {
  moran: "Moran I",
  geary: "Geary C",
  mean: "Mittelwert"
};

// ======================================
// Utils
// ======================================
function fmt(x, d = 2) {
  return Number.isFinite(x) ? x.toFixed(d) : "";
}

function norm(s) {
  return (s ?? "").toString().toLowerCase();
}

function parseFeature(feature) {
  if (!feature) return {};
  const parts = feature.split("_");
  return {
    season: parts[0],
    stat: parts[1],
    band: parts[2]
  };
}

// ======================================
// TEXT BUILDER
// ======================================
function buildDecisionText(decisionNode, childNode, edge) {
  if (!decisionNode?.data) return null;
  const { feature, threshold } = decisionNode.data;
  if (!feature || threshold == null) return null;

  const { stat, band, season } = parseFeature(feature);
  const statLabel   = STAT_LABELS[norm(stat)] ?? stat?.toUpperCase() ?? "";
  const bandLabel   = FEATURE_LABELS[norm(band)] ?? band?.toUpperCase() ?? "";
  const seasonLabel = season ? ` (${season})` : "";

  let op = "?";
  if (childNode?.data?._edge === "Ja")        op = "≥";
  else if (childNode?.data?._edge === "Nein") op = "<";
  else if (edge?.toId?.startsWith("suit_"))   op = "≥";  // Skalen-Edges

  return `${statLabel} ${bandLabel}${seasonLabel} ${op} ${fmt(threshold)}`;
}

function buildDecisionPathText(node, nodeById, edgeSegments) {
  if (!node?.parent) return [];

  const pathTexts = [];

  let current = node;
  let parent  = current.parent;

  while (parent) {
    // passende Edge suchen (nur für Anzeige / Richtung)
    const edge = edgeSegments?.find(e =>
      e.fromId === parent.data._nid &&
      (e.toId === current.data._nid ||
       e.toId === "suit_" + current.data._nid)
    );

    const line = buildDecisionText(parent, current, edge);
    if (line) pathTexts.push(line);

    current = parent;
    parent = current.parent;
  }

  return pathTexts.reverse(); // von Wurzel zu Blatt
}

// ======================================
// TEXT-POSITION AN EDGE-START
// ======================================
function getTextPositionForEdge(edge) {
  const { x: x0, y: y0 } = edge.source;
  const { y: y1 }         = edge.target;

  const offsetX = 6;
  const offsetY = y1 < y0 ? 14 : -6;  // Ziel ist höher → Text drüber

  return {
    x: x0 + offsetX,
    y: y0 + offsetY
  };
}

// ======================================
// MAIN FUNCTION
// ======================================
export function showDecisionExplanation({
  edge,
  node,
  nodeById,
  edgeSegments,
  layer
}) {
  console.group("🧠 showDecisionExplanation");

  console.log("input.edge:", edge);
  console.log("input.node:", node);
  console.log("input.nodeById:", !!nodeById);
  console.log("input.edgeSegments:", !!edgeSegments);
  console.log("input.layer:", !!layer);

  if (!layer) {
    console.warn("⛔ Kein gültiger Layer → abbrechen");
    console.groupEnd();
    return;
  }

  let decisionNode = null;
  let childNode    = null;

  // ============================
  // NODE-HOVER
  // ============================
  if (node && nodeById && edgeSegments) {
    decisionNode = node.parent ?? null;
    childNode    = node;

    console.log("🔍 Modus: NODE-HOVER");
    console.log("→ Parent gesucht für:", node.data?._nid);

    if (decisionNode) {
      edge = edgeSegments.find(e => {
        const isDirect = e.toId === node.data._nid &&
                         e.fromId === decisionNode.data._nid;

        const isScale = node.data._nid &&
                        e.toId === "suit_" + node.data._nid;

        return isDirect || isScale;
      });

      console.log("→ gefundene Edge:", edge);
    }
  }

  // ============================
  // EDGE-HOVER
  // ============================
  else if (edge && nodeById) {
    console.log("🔍 Modus: EDGE-HOVER");
    decisionNode = nodeById[edge.fromId] ?? null;
    const toId = edge.toId?.startsWith("suit_") ? edge.toId.replace("suit_", "") : edge.toId;
        childNode = nodeById[toId] ?? null;
  }

  if (!decisionNode || !edge) {
    console.warn("❌ Kein decisionNode oder edge → abbrechen");
    console.groupEnd();
    return;
  }

  console.log("🎯 decisionNode:", decisionNode?.data?._nid);
  console.log("🎯 childNode:", childNode?.data?._nid);

  // ============================
  // TEXT GENERIEREN
  // ============================

    const textLines = (node || childNode)
    ? buildDecisionPathText(node ?? childNode, nodeById, edgeSegments)
    : [];

  if (!textLines.length) {
    console.warn("❌ Kein gültiger Entscheidungstext");
    console.groupEnd();
    return;
  }

  console.log("📝 decisionText:", textLines.join(" / "));

  const pos = getTextPositionForEdge(edge, textLines);
  console.log("📍 Textposition:", pos);

// ============================
// RENDER (ohne Debug-Text)
// ============================
layer.selectAll(".decision-hover-label").remove();

const isAbove = edge?.target?.y < edge?.source?.y;
const lineHeightEm = 1.2;
const lineCount = textLines.length;

// Wenn Edge nach oben geht → Text darunter, also kein Offset nötig.
// Wenn Edge nach unten geht → Text darüber, also raufziehen.
const dyOffset = isAbove ? 0 : -lineHeightEm * (lineCount - 1);

const textEl = layer.append("text")
  .attr("class", "decision-hover-label")
  .attr("x", pos.x)
  .attr("y", pos.y)
  .attr("text-anchor", "start")
  .attr("font-size", 12)
  .attr("font-weight", 500)
  .attr("fill", "#222")
  .attr("pointer-events", "none");

// Erste Zeile mit initialem Offset
textLines.forEach((line, i) => {
  textEl.append("tspan")
    .attr("x", pos.x)
    .attr("dy", i === 0 ? `${dyOffset}em` : `${lineHeightEm}em`)
    .text(line);
});
}

// ======================================
// CLEAR
// ======================================
export function clearDecisionExplanation(layer) {
  if (!layer) return;
  layer.selectAll(".decision-hover-label").remove();
}