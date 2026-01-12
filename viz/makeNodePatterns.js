// makeNodePatterns.js

export function makeNodeGradients(defs) {
  // 🌱 NDVI – vertikaler Verlauf von grün (oben) nach hellgrau (unten)
  const ndvi = defs.append("linearGradient")
    .attr("id", "gradient-ndvi")
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "0%")
    .attr("y2", "100%");

  ndvi.append("stop").attr("offset", "0%").attr("stop-color", "#2f6b3a");
  ndvi.append("stop").attr("offset", "50%").attr("stop-color", "#a3c586");
  ndvi.append("stop").attr("offset", "100%").attr("stop-color", "#f2f2f2");

  // 💧 NDWI – vertikaler Verlauf von blau (oben) nach fast weiß (unten)
  const ndwi = defs.append("linearGradient")
    .attr("id", "gradient-ndwi")
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "0%")
    .attr("y2", "100%");

  ndwi.append("stop").attr("offset", "0%").attr("stop-color", "#08519c");
  ndwi.append("stop").attr("offset", "50%").attr("stop-color", "#6baed6");
  ndwi.append("stop").attr("offset", "100%").attr("stop-color", "#f7fbff");
}