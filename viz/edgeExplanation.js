// edgeExplanation.js
// SINGLE SOURCE – DEBUG VERSION

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

// feature string: m12_geary_ndwi
function parseFeature(feature) {
  if (!feature) return {};

  const parts = feature.split("_"); // ["m12","geary","ndwi"]
  const season = parts[0];
  const stat   = parts[1];
  const band   = parts[2];

  return {
    season,
    stat,
    band
  };
}

// ======================================
// 1) ZENTRALE TEXTFUNKTION
// ======================================
export function buildDecisionText(node) {
  console.log("🧠 buildDecisionText()");
  console.log("   node:", node);

  if (!node?.data) {
    console.warn("⚠️ no node.data");
    return "Decision Path";
  }

  const { feature, threshold, yes } = node.data;

  if (!feature || threshold == null) {
    console.warn("⚠️ missing feature/threshold", node.data);
    return "Decision Path";
  }

  const { stat, band, season } = parseFeature(feature);

  const statLabel =
    STAT_LABELS[norm(stat)] ?? stat?.toUpperCase() ?? "";

  const bandLabel =
    FEATURE_LABELS[norm(band)] ?? band?.toUpperCase() ?? "";

  const opLabel = yes ? "≥" : "<";

  const seasonLabel = season ? ` (${season})` : "";

  const text = `${statLabel} ${bandLabel}${seasonLabel} ${opLabel} ${fmt(threshold)}`;

  console.log("✅ decision text:", text);
  return text;
}

// ======================================
// 2) POSITION (einheitlich)
// ======================================
function decisionLabelPosition(node) {
  if (!node?.x || !node?.y) return null;

  return {
    x: node.x + 24,
    y: node.y - 24
  };
}

// ======================================
// 3) HAUPTFUNKTION (EDGE ODER NODE)
// ======================================
export function showDecisionExplanation({
  edge,
  node,
  nodeById,
  layer
}) {
  console.log("🟣 showDecisionExplanation()");
  console.log("   edge:", edge);
  console.log("   node:", node);

  if (!layer || !nodeById) {
    console.warn("❌ missing layer or nodeById");
    return;
  }

  // ----------------------------------
  // 1️⃣ Ziel-Node bestimmen
  // ----------------------------------
  let decisionNode = null;

  if (node) {
    decisionNode = node;
    console.log("➡️ using direct node");
  } else if (edge?.toId) {
    decisionNode = nodeById[edge.toId];
    console.log("➡️ resolved node from edge.toId:", edge.toId);
  }

  if (!decisionNode) {
    console.warn("❌ could not resolve decision node");
    return;
  }

  // ----------------------------------
  // 2️⃣ Text bauen
  // ----------------------------------
  const text = buildDecisionText(decisionNode);
  if (!text) return;

  // ----------------------------------
  // 3️⃣ Position
  // ----------------------------------
  const pos = decisionLabelPosition(decisionNode);
  if (!pos) return;

  // ----------------------------------
  // 4️⃣ Render (immer genau eins)
  // ----------------------------------
  layer.selectAll(".decision-hover-label").remove();

  layer.append("text")
    .attr("class", "decision-hover-label")
    .attr("x", pos.x)
    .attr("y", pos.y)
    .attr("font-size", 12)
    .attr("font-weight", 500)
    .attr("fill", "#222")
    .attr("pointer-events", "none")
    .text(text);
}

// ======================================
// 4) CLEAR
// ======================================
export function clearDecisionExplanation(layer) {
  if (!layer) return;
  layer.selectAll(".decision-hover-label").remove();
}