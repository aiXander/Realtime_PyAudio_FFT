// Side-panel resize + collapse. The controls panel shares the main grid with
// the viz area, separated by a draggable splitter (pointer events, so mouse,
// touch and pen all work; arrow keys resize when it has focus). Below 10% of
// page width the drag snaps to fully collapsed; the floating expand button
// restores the previous width. Width and collapsed state persist in
// localStorage (UI-only convenience, never sent to the server).

const STORAGE_KEY_W = "controls.width.px";
const STORAGE_KEY_C = "controls.collapsed";
const SNAP_FRAC = 0.10;
const DEFAULT_W = 304;
const MIN_W = 0;
const MAX_FRAC = 0.8;
const KEY_STEP = 16;

function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

export function setupSidebar() {
  const grid = document.getElementById("main-grid");
  const resizer = document.getElementById("grid-resizer");
  const controls = grid?.querySelector(".controls");
  const collapseBtn = document.getElementById("controls-collapse");
  const expandBtn = document.getElementById("controls-expand");
  if (!grid || !resizer || !controls || !collapseBtn || !expandBtn) return;

  const stored = parseInt(lsGet(STORAGE_KEY_W), 10);
  let savedW = Number.isFinite(stored) ? clamp(stored, 120, 1600) : DEFAULT_W;

  function applyWidth(px) {
    grid.style.setProperty("--controls-w", `${Math.round(px)}px`);
    resizer.setAttribute("aria-valuenow", String(Math.round(px)));
  }
  function applyCollapsed(c) {
    grid.classList.toggle("controls-collapsed", c);
    lsSet(STORAGE_KEY_C, c ? "1" : "0");
    // Move focus to whichever toggle is now visible so keyboard users
    // aren't stranded on a hidden button.
    const active = document.activeElement;
    if (active === collapseBtn && c) expandBtn.focus();
    else if (active === expandBtn && !c) collapseBtn.focus();
  }

  applyWidth(savedW);
  applyCollapsed(lsGet(STORAGE_KEY_C) === "1");

  let drag = null;
  resizer.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    e.preventDefault();
    drag = {
      pointerId: e.pointerId,
      startX: e.clientX,
      startW: controls.getBoundingClientRect().width,
    };
    try { resizer.setPointerCapture(e.pointerId); } catch {}
    resizer.classList.add("dragging");
    document.body.classList.add("col-resizing");
  });

  resizer.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.pointerId) return;
    const dx = e.clientX - drag.startX;
    const next = clamp(drag.startW - dx, MIN_W, window.innerWidth * MAX_FRAC);
    applyWidth(next);
  });

  function endDrag() {
    if (!drag) return;
    const { pointerId } = drag;
    drag = null;
    try { if (resizer.hasPointerCapture(pointerId)) resizer.releasePointerCapture(pointerId); } catch {}
    resizer.classList.remove("dragging");
    document.body.classList.remove("col-resizing");
    commitWidth(controls.getBoundingClientRect().width);
  }
  resizer.addEventListener("pointerup", endDrag);
  resizer.addEventListener("pointercancel", endDrag);
  resizer.addEventListener("lostpointercapture", endDrag);
  window.addEventListener("blur", endDrag);

  function commitWidth(w) {
    if (w / window.innerWidth < SNAP_FRAC) {
      // Snap closed; preserve the previously committed width as the "restore"
      // target so the expand button returns the panel to a sensible size.
      applyWidth(savedW);
      applyCollapsed(true);
    } else {
      savedW = Math.round(w);
      lsSet(STORAGE_KEY_W, String(savedW));
      applyWidth(savedW);
      applyCollapsed(false);
    }
  }

  // Keyboard: ←/→ grow/shrink the panel (it's on the right), Home/End = min/max.
  resizer.addEventListener("keydown", (e) => {
    const cur = controls.getBoundingClientRect().width;
    let next = null;
    if (e.key === "ArrowLeft") next = cur + KEY_STEP;
    else if (e.key === "ArrowRight") next = cur - KEY_STEP;
    else if (e.key === "Home") next = 200;
    else if (e.key === "End") next = window.innerWidth * MAX_FRAC;
    if (next === null) return;
    e.preventDefault();
    commitWidth(clamp(next, 160, window.innerWidth * MAX_FRAC));
  });

  collapseBtn.addEventListener("click", () => {
    const w = controls.getBoundingClientRect().width;
    if (w >= 60) {
      savedW = Math.round(w);
      lsSet(STORAGE_KEY_W, String(savedW));
    }
    applyWidth(savedW);
    applyCollapsed(true);
  });

  expandBtn.addEventListener("click", () => {
    applyCollapsed(false);
    applyWidth(savedW);
  });
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
