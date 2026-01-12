// layout.js – SAUBERE VERSION (Parent-basiert, konsistent)

export function buildLayout(treeData, config) {
  const { xSpacing, ySpacing, nodeW, nodeH, marginX } = config;

  const root = d3.hierarchy(treeData, d => {
    const kids = [];
    if (d.no) kids.push({ ...d.no, _edge: "Nein" });
    if (d.yes) kids.push({ ...d.yes, _edge: "Ja" });
    return kids.length ? kids : null;
  });

  // -------------------------
  // Struktur & Positionen
  // -------------------------
  let nodeCounter = 0;

  (function assignDepth(n, depth) {
    n.depth = depth;
    n.children?.forEach(c => assignDepth(c, depth + 1));
  })(root, 0);

  const leaves = root.leaves()
    .sort((a, b) => (b.data.suit ?? 0.5) - (a.data.suit ?? 0.5));

  leaves.forEach((leaf, i) => {
    leaf.xIndex = i;
  });

  (function assignInternalX(n) {
    if (n.children) {
      n.children.forEach(assignInternalX);
      n.xIndex = d3.mean(n.children, c => c.xIndex);
    }
  })(root);

  let minX = Infinity, maxX = -Infinity, maxY = -Infinity;

  root.each(d => {
    d.y = d.xIndex * ySpacing;
    d.x = d.depth * xSpacing;
    d.data._nid = "n" + (nodeCounter++);
    minX = Math.min(minX, d.x);
    maxX = Math.max(maxX, d.x);
    maxY = Math.max(maxY, d.y);
  });


// -----------------------------------
// Vertikale Spalten-Kompaktierung (centered)
// -----------------------------------

const MIN_NODE_GAP = nodeH + 20;

// Nodes nach Tiefe gruppieren
const nodesByDepth = d3.group(root.descendants(), d => d.depth);

nodesByDepth.forEach(nodes => {
  // nach ursprünglichem y sortieren
  nodes.sort((a, b) => a.y - b.y);

  // 🔒 Originale Mitte merken
  const originalTop = nodes[0].y;
  const originalBottom = nodes[nodes.length - 1].y;
  const originalCenter = (originalTop + originalBottom) / 2;

  // === Pass A: Push-down (Kollisionen vermeiden)
  for (let i = 1; i < nodes.length; i++) {
    const prev = nodes[i - 1];
    const curr = nodes[i];
    const minY = prev.y + MIN_NODE_GAP;
    if (curr.y < minY) {
      curr.y = minY;
    }
  }

  // === Pass B: Pull-up (Lücken schließen)
  for (let i = nodes.length - 2; i >= 0; i--) {
    const next = nodes[i + 1];
    const curr = nodes[i];
    const maxY = next.y - MIN_NODE_GAP;
    if (curr.y > maxY) {
      curr.y = maxY;
    }
  }

  // 🔒 Neue Mitte berechnen
  const newTop = nodes[0].y;
  const newBottom = nodes[nodes.length - 1].y;
  const newCenter = (newTop + newBottom) / 2;

  // === Pass C: Spalte um ursprüngliche Mitte zentrieren
  const delta = originalCenter - newCenter;

  nodes.forEach(n => {
    n.y += delta;
  });
});



  // -------------------------
  // KANTEN (Parent-basiert!)
  // -------------------------
  const edgeSegments = [];

  root.each(n => {
    if (!n.children) return;

    const threshold = n.data.threshold ?? 0.5;

    const thresholdY = n.y + (0.5 - threshold) * nodeH;

    const exitOffset = nodeH * 0.25;

    console.group(`🟦 Node ${n.data._nid}`);
    console.log("threshold:", threshold);
    console.log("thresholdY:", thresholdY, "-> innerhalb Node mit Höhe", nodeH);    console.log("thresholdY:", thresholdY);
    console.groupEnd();

    n.children.forEach(child => {
      const isYes = child.data._edge === "Ja";
        let fromY = isYes
        ? thresholdY - exitOffset
        : thresholdY + exitOffset;

        // ⛔ Begrenzung erzwingen (nicht außerhalb des Node-Rechtecks)
        const yMin = n.y - nodeH / 2 + 4;
        const yMax = n.y + nodeH / 2 - 4;
        fromY = Math.max(yMin, Math.min(fromY, yMax));

      if (child.data.leaf) {
        // 👉 DIREKT ZUR SKALA
        edgeSegments.push({
          source: {
            x: n.x + nodeW / 2 + marginX,
            y: fromY
          },
          target: { x: null, y: null },
          fromId: n.data._nid,
          toId: "suit_" + child.data._nid,
          cls: "edge scale-edge",
          isScaleEdge: true,
          suit: child.data.suit
        });
      } else {
        // 👉 ZUM NÄCHSTEN NODE
        edgeSegments.push({
          source: {
            x: n.x + nodeW / 2 + marginX,
            y: fromY
          },
          target: {
            x: child.x - nodeW / 2 - marginX,
            y: child.y
          },
          fromId: n.data._nid,
          toId: child.data._nid,
          cls: "edge"
        });
      }
    });
  });

  // -------------------------
  // SKALA
  // -------------------------
  const barX = maxX + xSpacing * 0.9;
  const barY = minX + 30;
  const barH = (maxY - minX) - 80;
  const barW = 40;

  const suitToY = d3.scaleLinear()
    .domain([0, 1])
    .range([barY + barH, barY]);

  edgeSegments.forEach(e => {
    if (e.isScaleEdge && e.suit != null) {
      e.target.x = barX;
      e.target.y = suitToY(e.suit);
    }
  });

  return {
    root,
    leaves,
    minX,
    maxX,
    maxY,
    barX,
    barY,
    barW,
    barH,
    suitToY,
    edgeSegments
  };
}