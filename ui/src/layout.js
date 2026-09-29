// Tiling 2x2 viz layout. Two splits (split_x, split_y) divide the container
// into four quadrants TL, TR, BL, BR; `quadrants` assigns a card id to each.
// Cards always tile the container fully — empty space is impossible by
// construction. Dragging an internal edge moves a split (and therefore the
// neighbor on the other side too). Dragging the title onto another card
// swaps their quadrant assignments.

import { send } from "./ws.js";

const CARDS = ["bars", "lines", "scene", "fft"];
// Quadrant indices in the `quadrants` array.
const TL = 0, TR = 1, BL = 2, BR = 3;
const SPLIT_MIN = 0.1;
const SPLIT_MAX = 0.9;

let container = null;
let cardEls = {};   // id -> { card, title, handles: {} }
let centerHandle = null;
let layout = null;  // { split_x, split_y, quadrants } — null until first meta
let dragging = null;
let lastKey = "";   // fingerprint of the last applied server layout

// Below this width the CSS stacks the cards in one column (see style.css
// `@media (max-width: 760px)`), so tiling drags are disabled there.
const stackedMq = window.matchMedia ? window.matchMedia("(max-width: 760px)") : null;
const isStacked = () => !!(stackedMq && stackedMq.matches);

export function setupLayout() {
  container = document.querySelector(".viz-grid");
  if (!container) return;
  for (const id of CARDS) {
    const card = container.querySelector(`[data-card="${id}"]`);
    if (!card) continue;
    const title = card.querySelector(".viz-title");
    const handles = createHandles(card, id);
    cardEls[id] = { card, title, handles };
    addTitleDrag(card, title, id);
  }
  centerHandle = document.createElement("div");
  centerHandle.className = "viz-resize-center";
  centerHandle.addEventListener("pointerdown", beginCenterDrag);
  container.appendChild(centerHandle);
  window.addEventListener("blur", () => endDrag(null));
}

/** Apply the server's saved layout. Cheap no-op when it hasn't changed
 *  since the last call (meta arrives on every control change). */
export function applyLayout(next) {
  if (!container || !next) return;
  if (dragging) return; // don't clobber in-flight drag with the meta echo
  if (typeof next.split_x !== "number" || typeof next.split_y !== "number" || !Array.isArray(next.quadrants)) return;
  const q = next.quadrants;
  if (q.length !== 4 || !CARDS.every((id) => q.includes(id))) return;
  const key = `${next.split_x}|${next.split_y}|${q.join(",")}`;
  if (key === lastKey && layout) return;
  lastKey = key;
  layout = { split_x: next.split_x, split_y: next.split_y, quadrants: q.slice() };
  if (!container.classList.contains("free-layout")) {
    container.classList.add("free-layout");
  }
  paintAll();
}

function paintAll() {
  for (const id of CARDS) {
    const els = cardEls[id]; if (!els) continue;
    const r = rectFor(id);
    paint(els.card, r);
    setHandleVisibility(els.handles, quadrantOf(id));
  }
  if (centerHandle) {
    centerHandle.style.left = `${(layout.split_x * 100).toFixed(4)}%`;
    centerHandle.style.top  = `${(layout.split_y * 100).toFixed(4)}%`;
  }
}

function quadrantOf(id) { return layout.quadrants.indexOf(id); }

function rectFor(id) {
  const q = quadrantOf(id);
  const sx = layout.split_x, sy = layout.split_y;
  switch (q) {
    case TL: return { x: 0,  y: 0,  w: sx,     h: sy };
    case TR: return { x: sx, y: 0,  w: 1 - sx, h: sy };
    case BL: return { x: 0,  y: sy, w: sx,     h: 1 - sy };
    case BR: return { x: sx, y: sy, w: 1 - sx, h: 1 - sy };
  }
  return { x: 0, y: 0, w: 1, h: 1 };
}

// Visible inset per side, in CSS px. The visible gap between two adjacent
// quadrants is 2 * GAP_PX; outer cards sit GAP_PX from the container edge.
const GAP_PX = 5;

function paint(card, r) {
  card.style.left   = `calc(${(r.x * 100).toFixed(4)}% + ${GAP_PX}px)`;
  card.style.top    = `calc(${(r.y * 100).toFixed(4)}% + ${GAP_PX}px)`;
  card.style.width  = `calc(${(r.w * 100).toFixed(4)}% - ${2 * GAP_PX}px)`;
  card.style.height = `calc(${(r.h * 100).toFixed(4)}% - ${2 * GAP_PX}px)`;
}

function pushLayout(commit) {
  if (!layout) return;
  lastKey = `${layout.split_x}|${layout.split_y}|${layout.quadrants.join(",")}`;
  send({ type: "set_ui_layout", layout, commit });
}

// A drag may start only with the primary button, once the server's layout
// has arrived (before that the cards sit in the CSS fallback grid and there
// is nothing to resize), and not in the stacked narrow layout.
function canDrag(e) {
  if (!layout || !container) return false;
  if (e.pointerType === "mouse" && e.button !== 0) return false;
  return !isStacked();
}

// ---------- handles ----------

// Each card has up to 4 internal-edge handles. We create all 4 up front and
// hide the ones that touch the container boundary based on the card's current
// quadrant. The east edge of TL is split_x; the south edge of TL is split_y;
// etc. Dragging any internal edge moves the corresponding split.
function createHandles(card, id) {
  const handles = {};
  for (const edge of ["n", "s", "e", "w"]) {
    const el = document.createElement("div");
    el.className = `viz-resize viz-resize-${edge}`;
    el.addEventListener("pointerdown", (e) => beginSplitDrag(e, edge, id));
    card.appendChild(el);
    handles[edge] = el;
  }
  return handles;
}

function setHandleVisibility(handles, q) {
  // Show only the edges that face the interior of the container.
  // TL -> e, s.   TR -> w, s.   BL -> e, n.   BR -> w, n.
  const visible = {
    [TL]: { n: false, s: true,  e: true,  w: false },
    [TR]: { n: false, s: true,  e: false, w: true  },
    [BL]: { n: true,  s: false, e: true,  w: false },
    [BR]: { n: true,  s: false, e: false, w: true  },
  }[q];
  for (const edge of ["n", "s", "e", "w"]) {
    handles[edge].style.display = visible[edge] ? "" : "none";
  }
}

function beginSplitDrag(e, edge, id) {
  if (!canDrag(e)) return;
  e.preventDefault();
  e.stopPropagation();
  // Resolve which split this edge moves.
  // Vertical edges (e/w) move split_x. Horizontal edges (n/s) move split_y.
  const axis = (edge === "e" || edge === "w") ? "split_x" : "split_y";
  const rect = container.getBoundingClientRect();
  dragging = {
    kind: "split",
    axis,
    rect,
    startSplit: layout[axis],
    originX: e.clientX,
    originY: e.clientY,
    pointerId: e.pointerId,
  };
  document.body.classList.add("layout-dragging");
  attachWindowDrag();
}

function beginCenterDrag(e) {
  if (!canDrag(e)) return;
  e.preventDefault();
  e.stopPropagation();
  const rect = container.getBoundingClientRect();
  dragging = {
    kind: "center",
    rect,
    startX: layout.split_x,
    startY: layout.split_y,
    originX: e.clientX,
    originY: e.clientY,
    pointerId: e.pointerId,
  };
  centerHandle.classList.add("dragging");
  document.body.classList.add("layout-dragging");
  attachWindowDrag();
}

// ---------- title drag (swap) ----------

function addTitleDrag(card, title, id) {
  if (!title) return;
  title.addEventListener("pointerdown", (e) => {
    const tag = e.target.tagName;
    if (tag === "INPUT" || tag === "LABEL" || e.target.closest("label")) return;
    if (!canDrag(e)) return;
    e.preventDefault();
    const rect = container.getBoundingClientRect();
    dragging = {
      kind: "swap",
      id,
      rect,
      originX: e.clientX,
      originY: e.clientY,
      moved: false,
      pointerId: e.pointerId,
    };
    card.classList.add("dragging");
    attachWindowDrag();
  });
}

// ---------- window-level drag plumbing ----------

function attachWindowDrag() {
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("pointercancel", onCancel);
}

function detachWindowDrag() {
  window.removeEventListener("pointermove", onMove);
  window.removeEventListener("pointerup", onUp);
  window.removeEventListener("pointercancel", onCancel);
}

function onMove(e) {
  if (!dragging || !layout || e.pointerId !== dragging.pointerId) return;
  if (dragging.kind === "split") {
    const dxFrac = (e.clientX - dragging.originX) / dragging.rect.width;
    const dyFrac = (e.clientY - dragging.originY) / dragging.rect.height;
    const delta = dragging.axis === "split_x" ? dxFrac : dyFrac;
    const next = clamp(dragging.startSplit + delta, SPLIT_MIN, SPLIT_MAX);
    if (next !== layout[dragging.axis]) {
      layout[dragging.axis] = next;
      paintAll();
      pushLayout(false);
    }
  } else if (dragging.kind === "center") {
    const dxFrac = (e.clientX - dragging.originX) / dragging.rect.width;
    const dyFrac = (e.clientY - dragging.originY) / dragging.rect.height;
    const nx = clamp(dragging.startX + dxFrac, SPLIT_MIN, SPLIT_MAX);
    const ny = clamp(dragging.startY + dyFrac, SPLIT_MIN, SPLIT_MAX);
    if (nx !== layout.split_x || ny !== layout.split_y) {
      layout.split_x = nx;
      layout.split_y = ny;
      paintAll();
      pushLayout(false);
    }
  } else if (dragging.kind === "swap") {
    const dx = e.clientX - dragging.originX;
    const dy = e.clientY - dragging.originY;
    if (dx * dx + dy * dy > 9) dragging.moved = true;
    // Visually nudge the dragging card toward the cursor for feedback,
    // without changing the underlying layout (which only commits on drop).
    const card = cardEls[dragging.id].card;
    card.style.transform = `translate(${dx}px, ${dy}px)`;
    highlightDropTarget(e);
  }
}

function highlightDropTarget(e) {
  const target = findCardAt(e.clientX, e.clientY, dragging.id);
  for (const id of CARDS) {
    if (!cardEls[id]) continue;
    cardEls[id].card.classList.toggle("drop-target", id === target);
  }
}

function findCardAt(clientX, clientY, excludeId) {
  if (!layout) return null;
  const rect = container.getBoundingClientRect();
  const xFrac = (clientX - rect.left) / rect.width;
  const yFrac = (clientY - rect.top) / rect.height;
  if (xFrac < 0 || xFrac > 1 || yFrac < 0 || yFrac > 1) return null;
  const q = (xFrac < layout.split_x ? 0 : 1) + (yFrac < layout.split_y ? 0 : 2);
  const id = layout.quadrants[q];
  return id === excludeId ? null : id;
}

function onUp(e) {
  if (dragging && e.pointerId !== dragging.pointerId) return;
  endDrag(e);
}

// pointercancel / window blur: finish the drag without swapping.
function onCancel(e) {
  if (dragging && e.pointerId !== dragging.pointerId) return;
  endDrag(null);
}

function endDrag(e) {
  detachWindowDrag();
  if (!dragging) return;
  const d = dragging;
  dragging = null;
  document.body.classList.remove("layout-dragging");
  if (centerHandle) centerHandle.classList.remove("dragging");

  if (d.kind === "swap") {
    const card = cardEls[d.id].card;
    card.classList.remove("dragging");
    card.style.transform = "";
    for (const id of CARDS) cardEls[id]?.card.classList.remove("drop-target");
    if (d.moved && e && layout) {
      const target = findCardAt(e.clientX, e.clientY, d.id);
      if (target) {
        // Swap quadrant assignments. The visual rectangles stay tiled.
        const i = layout.quadrants.indexOf(d.id);
        const j = layout.quadrants.indexOf(target);
        [layout.quadrants[i], layout.quadrants[j]] = [layout.quadrants[j], layout.quadrants[i]];
        paintAll();
        pushLayout(true);
        return;
      }
    }
    // No swap occurred — nothing to commit.
    return;
  }

  // split or center drag end
  pushLayout(true);
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
