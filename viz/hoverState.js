// hoverState.js

let state = {
  type: null,        // "edge" | "node"
  edge: null,
  nodeId: null,
  explanation: null
};

const listeners = new Set();

export function setHoverState(next) {
  state = { ...state, ...next };
  listeners.forEach(fn => fn(state));
}

export function clearHoverState() {
  state = {
    type: null,
    edge: null,
    nodeId: null,
    explanation: null
  };
  listeners.forEach(fn => fn(state));
}

export function onHoverStateChange(fn) {
  listeners.add(fn);
}