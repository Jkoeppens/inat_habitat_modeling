export function drawScale({ g, layout, config, gradientId }) {
  const { barX, barY, barW, barH } = layout;

  g.append("rect")
    .attr("x", barX)
    .attr("y", barY)
    .attr("width", barW)
    .attr("height", barH)
    .attr("fill", `url(#${gradientId})`)
    .attr("stroke", "#333");

  g.append("text")
    .attr("x", barX)
    .attr("y", barY - 10)
    .attr("font-size", 11)
    .text("Suitability (0 → 1, Viridis)");
}