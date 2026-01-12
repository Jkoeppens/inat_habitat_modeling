export function makeGradients(defs, palettes) {
  return palettes.map((palette, i) => {
    const id = i === 0 ? "gradSuit" : "grad_" + i;
    const gr = defs.append("linearGradient")
      .attr("id", id)
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "0%")
      .attr("y2", "100%");

    [...palette].reverse().forEach((c, i) =>
      gr.append("stop")
        .attr("offset", (i / (palette.length - 1)) * 100 + "%")
        .attr("stop-color", c)
    );

    return id;
  });
}