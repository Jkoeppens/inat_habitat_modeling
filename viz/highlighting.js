// highlighting.js — FINAL (klare Semantik, keine Seiteneffekte)

export function setupHighlighting(root, edgeSegments) {

  // ===============================
  // PUBLIC API
  // ===============================

  function highlightFromNode(nodeId) {
    console.log("🟢 NODE HOVER", nodeId);

    const edgeSet = new Set();
    const leafIds = new Set();

    collectAncestors(nodeId, edgeSet);
    collectDescendants(nodeId, edgeSet);

    // 🔑 DAS FEHLTE
    collectLeavesFromEdges(edgeSet, leafIds);

    applyHighlights(edgeSet, leafIds);
  }

  function highlightFromEdge(edge) {
    console.log("🔵 EDGE HOVER", edge);

    const edgeSet = new Set();
    const leafIds = new Set();

    if (edge.isScaleEdge) {
      // 🔑 WICHTIG: Scale-Edge = EIN Pfad
      collectPathToRoot(edge, edgeSet);
      collectLeafFromScaleEdge(edge, leafIds);
    } else {
      // normale Tree-Edge
      collectPathThroughEdge(edge, edgeSet);
      collectLeavesFromEdges(edgeSet, leafIds);
    }

    applyHighlights(edgeSet, leafIds);
  }

  function clearHighlight() {
    d3.selectAll(".highlighted").classed("highlighted", false);
    d3.selectAll(".faded").classed("faded", false);
  }

  return {
    highlightFromNode,
    highlightFromEdge,
    clearHighlight
  };

  // ===============================
  // TRAVERSAL
  // ===============================

  function collectAncestors(nodeId, out) {
    edgeSegments.forEach(e => {
      if (e.toId === nodeId && !out.has(e)) {
        out.add(e);
        collectAncestors(e.fromId, out);
      }
    });
  }

  function collectDescendants(nodeId, out) {
    edgeSegments.forEach(e => {
      if (e.fromId === nodeId && !out.has(e)) {
        out.add(e);
        collectDescendants(e.toId, out);
      }
    });
  }

  function collectPathThroughEdge(edge, out) {
    out.add(edge);
    collectAncestors(edge.fromId, out);
    collectDescendants(edge.toId, out);
  }

  function collectPathToRoot(edge, out) {
    out.add(edge);
    collectAncestors(edge.fromId, out);
  }

  function collectLeafFromScaleEdge(edge, leafIds) {
    if (edge.toId?.startsWith("suit_")) {
      leafIds.add(edge.toId.replace("suit_", ""));
    }
  }

  function collectLeavesFromEdges(edgeSet, leafIds) {
    edgeSet.forEach(e => {
      if (e.toId?.startsWith("suit_")) {
        leafIds.add(e.toId.replace("suit_", ""));
      }
    });
  }

  // ===============================
  // APPLY
  // ===============================

  function applyHighlights(edgeSet, leafIds) {
    // Edges
    d3.selectAll("path.edge")
      .classed("highlighted", d => edgeSet.has(d))
      .classed("faded", d => !edgeSet.has(d));

    // Leaves
    d3.selectAll(".leaf-tri, .suit-label")
      .classed("highlighted", false)
      .classed("faded", true);

    leafIds.forEach(id => {
      d3.selectAll(`.leaf-${id}`)
        .classed("highlighted", true)
        .classed("faded", false);
    });

    // Node-Labels
    d3.selectAll(".node-label-outside")
      .classed("highlighted", false)
      .classed("faded", true);

    edgeSet.forEach(e => {
      d3.selectAll(`.node-label-${e.fromId}`)
        .classed("highlighted", true)
        .classed("faded", false);
      d3.selectAll(`.node-label-${e.toId}`)
        .classed("highlighted", true)
        .classed("faded", false);
    });
  }
}