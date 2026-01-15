""// script.js – Horizontale D3-Visualisierung des Surrogate Trees
console.log("✅ script.js geladen");

window.addEventListener("DOMContentLoaded", () => {
  if (typeof TREE_DATA === "undefined") {
    console.error("❌ TREE_DATA ist nicht definiert");
    return;
  }

  console.time("⏱️ Rendering abgeschlossen in");

  const treeData = TREE_DATA;
  const svg = d3.select("#tree-svg");

  /* =====================
     Layout-Parameter
  ===================== */
  const nodeW = 50;
  const nodeH = 140;
  const xSpacing = 300;
  const ySpacing =80;
  const PAD = 100;
  const marginX = 6;
  const offsetY = 16;

  /* =====================
     Farben
  ===================== */
  const viridis = [
    "#440154", "#472f7d", "#3b518b", "#2c718e",
    "#21918c", "#27ad81", "#5cc863", "#aadc32", "#fde725"
  ];

  const paletteForFeature = name => {
    if (!name) return ["#ddd", "#bbb", "#999"];
    const n = name.toLowerCase();
    if (n.includes("ndvi"))  return ["#f2f2f2", "#a3c586", "#2f6b3a"];
    if (n.includes("ndwi"))  return ["#f7fbff", "#6baed6", "#08519c"];
    if (n.includes("moran")) return ["#fee8c8", "#fdbb84", "#e34a33"];
    if (n.includes("geary")) return ["#f7f4f9", "#998ec3", "#542788"];
    return ["#eee", "#ccc", "#aaa"];
  };

  const rangeForFeature = name => {
    const n = name.toLowerCase();
    if (n.includes("ndvi"))  return [0, 1];
    if (n.includes("ndwi"))  return [-1, 1];
    if (n.includes("moran")) return [-0.2, 4];
    if (n.includes("geary")) return [0, 1.5];
    return [0, 1];
  };

  /* =====================
     Hierarchie
  ===================== */
  const root = d3.hierarchy(treeData, d => {
    const kids = [];
    if (d.no)  kids.push({ ...d.no, _edge: "Nein" });
    if (d.yes) kids.push({ ...d.yes, _edge: "Ja" });
    return kids.length ? kids : null;
  });

  (function assignDepth(n, depth) {
    n.depth = depth;
    n.children?.forEach(c => assignDepth(c, depth + 1));
  })(root, 0);

// =====================
// Sortierung & Layout
// =====================

// 1. Leaf-Knoten nach Suitability sortieren (höchste Eignung oben)
  const leaves = root.leaves()
    .sort((a, b) => (b.data.suit ?? 0.5) - (a.data.suit ?? 0.5));

  // 2. xIndex (→ y-Position später) für Leaf-Knoten setzen
  leaves.forEach((leaf, i) => {
    leaf.xIndex = i;
  });

  // 3. Interne Knoten auf mittleren xIndex ihrer Kinder setzen
  (function assignInternalX(n) {
    if (n.children) {
      n.children.forEach(assignInternalX);
      n.xIndex = d3.mean(n.children, c => c.xIndex);
    }
  })(root);

  // 4. Konkrete Koordinaten berechnen
  let minX = 1e9, maxX = -1e9, maxY = -1e9;
  let nodeCounter = 0;

  root.each(d => {
    d.y = d.xIndex * ySpacing;
    d.x = d.depth * xSpacing;
    d.data._nid = "n" + (nodeCounter++);
    minX = Math.min(minX, d.x);
    maxX = Math.max(maxX, d.x);
    maxY = Math.max(maxY, d.y);
  });

  const svgWidth  = maxX + 3 * PAD;
  const svgHeight = (maxY - minX) + 2 * PAD;

  svg.attr("viewBox", `0 0 ${svgWidth} ${svgHeight}`)
     .attr("preserveAspectRatio", "xMinYMid meet");

  const g = svg.append("g")
    .attr("transform", `translate(${PAD}, ${PAD - minX})`);

  /* =====================
     Gradients
  ===================== */
  const defs = svg.append("defs");

  const gradientIdFor = palette => {
    const id = "grad_" + Math.random().toString(36).slice(2);
    const gr = defs.append("linearGradient")
      .attr("id", id)
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "0%")
      .attr("y2", "100%");

    // Umkehrung der Palette: höherer Wert oben, niedriger unten
    [...palette].reverse().forEach((c, i) =>
      gr.append("stop")
        .attr("offset", (i / (palette.length - 1)) * 100 + "%")
        .attr("stop-color", c)
    );

    return id;
  };

  const thresholdYLocal = d => {
    const label = d.label || d.feature || "";
    const [min, max] = d.range || rangeForFeature(label);
    const rel = (d.threshold - min) / (max - min);
    return -nodeH / 2 + Math.max(0.05, Math.min(0.95, rel)) * nodeH;
  };

  function collectAncestors(startId, edges, out) {
    edges.forEach(e => {
      if (e.toId === startId && !out.has(e)) {
        out.add(e);
        collectAncestors(e.fromId, edges, out);
      }
    });
  }

  function collectDescendants(startId, edges, out) {
    edges.forEach(e => {
      if (e.fromId === startId && !out.has(e)) {
        console.log("✅ Gefundene Kante:", e.fromId, "→", e.toId, "| isScaleEdge:", e.isScaleEdge);
        out.add(e);
        collectDescendants(e.toId, edges, out);

        // 🔽 NEU: Skala-Kante? Dann virtuell weitergehen
        if (e.isScaleEdge) {
          collectDescendants(e.toId, edges, out);
        }
      }
    });
  }

  function highlightFromNode(nodeId) {
    console.log("➡️ highlightFromNode:", nodeId);

    const targets = new Set();
    collectAncestors(nodeId, edgeSegments, targets);
    collectDescendants(nodeId, edgeSegments, targets);

    console.log("🎯 Targets:");
    console.table([...targets].map(e => ({
      from: e.fromId,
      to: e.toId,
      isScaleEdge: e.isScaleEdge
    })));

    d3.selectAll("path.edge")
      .classed("highlighted", d => targets.has(d))
      .classed("faded", d => !targets.has(d));
  }

  function targetsHasLeaf(leafId, targets) {
    for (const e of targets) {
      if (e.toId === leafId) return true;
    }
    return false;
  }

  function clearHighlight() {
    d3.selectAll("path.edge")
      .classed("highlighted", false)
      .classed("faded", false);
  }


  const nodeG = g.selectAll("g.node")
    .data(root.descendants().filter(d => !d.data.leaf))
    .join("g")
    .attr("class", "node")
    .attr("transform", d => `translate(${d.x},${d.y})`);

  nodeG
    .on("mouseover", (ev, d) => {
      highlightFromNode(d.data._nid);
    })
    .on("mouseout", clearHighlight);
    
  nodeG.each(function(d) {
    const gsel = d3.select(this);
    const grad = gradientIdFor(paletteForFeature(d.data.label || d.data.feature));

    gsel.append("rect")
      .attr("x", -nodeW / 2)
      .attr("y", -nodeH / 2)
      .attr("width", nodeW)
      .attr("height", nodeH)
      .attr("rx", 18)
      .attr("fill", `url(#${grad})`)
      .attr("stroke", "#333");

    gsel.append("line")
      .attr("x1", -nodeW / 2)
      .attr("x2", nodeW / 2)
      .attr("y1", thresholdYLocal(d.data))
      .attr("y2", thresholdYLocal(d.data))
      .attr("stroke", "#000")
      .attr("stroke-width", 2);

    gsel.append("text")
      .attr("y", -nodeH / 2 - 10)
      .attr("text-anchor", "middle")
      .text(d.data.label || d.data.feature);
  });

  
  /* =====================
     Pfade (Node → Node)
  ===================== */
  const edgeSegments = [];

  root.each(n => {
    if (!n.children || n.data.leaf) return;

    n.children.forEach(child => {
      if (child.data.leaf) return; // <- Leaf überspringen!

      const side = child.data._edge === "Ja" ? +1 : -1;
      const y0 = n.y + thresholdYLocal(n.data) + side * offsetY;

      edgeSegments.push({
        source: { x: n.x + nodeW / 2 + marginX, y: y0 },
        target: { x: child.x - nodeW / 2 - marginX, y: child.y },

        fromId: n.data._nid,
        toId: child.data._nid,

        cls: `edge node-${n.data._nid}`,
      });
    });
  });

  /* =====================
     Suitability-Skala
  ===================== */
  const barX = maxX + 140;
  const barY = minX + 30;
  const barH = (maxY - minX) - 80;
  const barW = 40;

  const suitToY = d3.scaleLinear()
    .domain([0, 1])
    .range([barY + barH, barY]);

  const gradSuit = defs.append("linearGradient")
    .attr("id", "gradSuit")
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "0%")
    .attr("y2", "100%");

  [...viridis].reverse().forEach((c, i) =>
    gradSuit.append("stop")
      .attr("offset", (i / (viridis.length - 1)) * 100 + "%")
      .attr("stop-color", c)
  );

  g.append("rect")
    .attr("x", barX)
    .attr("y", barY)
    .attr("width", barW)
    .attr("height", barH)
    .attr("fill", "url(#gradSuit)")
    .attr("stroke", "#333");

  g.append("text")
    .attr("x", barX)
    .attr("y", barY - 10)
    .attr("font-size", 11)
    .text("Suitability (0 → 1, Viridis)");

  /* =====================
    Pfade (Leaf → Skala als virtueller Node)
  ===================== */
  leaves.forEach(lf => {
    const suit = Math.max(0, Math.min(1, lf.data.suit ?? 0.5));
    const scaleNodeId = `${lf.data._nid}_scale`;
    lf.data._scaleNodeId = scaleNodeId;

    // Neue Kante: Leaf → virtueller Skala-Node
    edgeSegments.push({
      source: { x: lf.x, y: lf.y },
      target: { x: barX, y: suitToY(suit) },

      fromId: lf.data._nid,
      toId: scaleNodeId,

      isScaleEdge: true,
      cls: `edge node-${lf.data._nid} node-${scaleNodeId} leaf-scale`,
    });
  });
  
  console.log("📦 Alle Skalenkanten:");
  console.table(edgeSegments.filter(e => e.isScaleEdge));

  const lineGen = d3.line()
    .x(d => d.x)
    .y(d => d.y)
    .curve(d3.curveCatmullRom.alpha(0.2)); // leichte Kurve

  g.append("g")
    .selectAll("path.edge")
    .data(edgeSegments)
    .join("path")
    .attr("class", d => d.cls)
    .attr("fill", "none")
    .attr("stroke", "#999")
    .attr("stroke-width", 1.1)
    .attr("opacity", 0.6)
    .attr("d", d => {
      const { x: x0, y: y0 } = d.source;
      const { x: x1, y: y1 } = d.target;
      const dx = x1 - x0;
      const curveStrength = 0.5;
      const cx1 = x0 + dx * curveStrength;
      const cx2 = x1 - dx * curveStrength;
      const path = d3.path();
      path.moveTo(x0, y0);
      path.bezierCurveTo(cx1, y0, cx2, y1, x1, y1);
      return path.toString();
    })

    .on("mouseover", (ev, d) => {
      highlightFromNode(d.fromId);
    })
    .on("mouseout", clearHighlight);

    console.log("📊 EdgeSegments:");
    console.table(edgeSegments.map(e => ({
      from: e.fromId,
      to: e.toId,
      isScaleEdge: e.isScaleEdge || false,
      cls: e.cls
    })));

  /* =====================
     Dreiecke & Werte
  ===================== */
  g.append("g")
    .selectAll("path.tri")
    .data(leaves)
    .join("path")
    .attr("class", d => `leaf-tri leaf-${d.data._nid}`)
    .attr("fill", d => d3.scaleLinear().domain([0,1]).range(viridis)(d.data.suit))
    .attr("stroke", "#000")
    .attr("stroke-width", 0.5)
    .attr("d", d => {
      const s = Math.max(0, Math.min(1, d.data.suit ?? 0.5));
      const cy = suitToY(s);
      const cx = barX + barW / 2;
      const t = 8;
      return `M ${cx - t},${cy} L ${cx + t},${cy - t} L ${cx + t},${cy + t} Z`;
    });

  g.append("g")
    .selectAll("text")
    .data(leaves)
    .join("text")
    .attr("class", "suit-label")
    .attr("text-anchor", "start")
    .attr("font-size", 11)
    .attr("fill", "#333")
    .attr("x", barX + barW + 6)
    .attr("y", d => suitToY(Math.max(0, Math.min(1, d.data.suit ?? 0.5))) + 4)
    .text(d => (d.data.suit ?? 0.5).toFixed(3));

  console.timeEnd("⏱️ Rendering abgeschlossen in");
  console.log("🌳 Baum korrekt gerendert");
});
