// setupSvg.js
export function setupSvg(svgSelection) {
  svgSelection
    .attr("width", "100%")
    .attr("height", "100%");

  const defs = svgSelection.append("defs");
  const g = svgSelection.append("g");

  return { svg: svgSelection, g, defs };
}