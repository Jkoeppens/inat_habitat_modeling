// 🛠️ utils.js

// 🎨 Farbpaletten je nach Feature-Name
export function paletteForFeature(name) {
  if (!name) return ["#ddd", "#bbb", "#999"];
  const n = name.toLowerCase();

  if (n.includes("ndvi")) return ["#f2f2f2", "#a3c586", "#2f6b3a"];   // weiß → grün
  if (n.includes("ndwi")) return ["#f7fbff", "#6baed6", "#08519c"];  // weiß → blau
  if (n.includes("moran")) return ["#fee8c8", "#fdbb84", "#e34a33"]; // beige → orange
  if (n.includes("geary")) return ["#f7f4f9", "#998ec3", "#542788"]; // hell → violett

  return ["#eee", "#ccc", "#aaa"]; // fallback
}

// 📈 Wertebereiche je nach Feature-Name
export function rangeForFeature(name) {
  const n = name.toLowerCase();

  if (n.includes("ndvi")) return [0, 1];
  if (n.includes("ndwi")) return [-1, 1];
  if (n.includes("moran")) return [-0.2, 4];
  if (n.includes("geary")) return [0, 1.5];

  return [0, 1]; // fallback
}

// 🔽 Berechnet Y-Position des Thresholds relativ zum Node
export function thresholdYLocal(d, nodeH) {
  const label = d.label || d.feature || "";
  const [min, max] = d.range || rangeForFeature(label);
  const rel = (d.threshold - min) / (max - min);

  return -nodeH / 2 + Math.max(0.05, Math.min(0.95, rel)) * nodeH;
}

// 🌈 Erzeugt ein vertikales SVG-Gradient mit gegebener Palette
export function makeGradient(svgDefs, palette) {
  if (!palette) {
    console.warn("⚠️ Keine Palette übergeben, nutze Fallback");
    palette = ["#ddd", "#bbb", "#999"];
  }

  const id = "grad_" + Math.random().toString(36).slice(2);

  const gr = svgDefs.append("linearGradient")
    .attr("id", id)
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "0%")
    .attr("y2", "100%");

  [...palette].reverse().forEach((c, i) => {
    gr.append("stop")
      .attr("offset", `${(i / (palette.length - 1)) * 100}%`)
      .attr("stop-color", c);
  });

  return id;
}

// 🧭 Bestimmt den Indikator-Typ aus dem Namen
export function indicatorType(name) {
  if (!name) return "other";

  const n = name.toLowerCase();

  if (n.includes("moran")) return "moran";
  if (n.includes("geary")) return "geary";
  if (n.includes("ndvi")) return "ndvi";
  if (n.includes("ndwi")) return "ndwi";

  return "other";
}