// Custom tooltips with a short hover delay. Replaces the browser's native
// `title` tooltip (which has a ~500ms+ delay that can't be configured).
//
// Accessibility: `title` is moved to `data-tooltip`, and a plain-text copy
// is kept in a hidden description node wired up with aria-describedby on the
// relevant focusable control (the input inside a <label>, a button, …), so
// screen readers still get the help text. Tooltips also open on keyboard
// focus (:focus-visible), anchored to the focused element; Esc hides them.
//
// Tooltip content is rendered as a small subset of Markdown:
//   **bold**, *italic*, `code`, # / ## headings, - bullet lists,
//   blank-line-separated paragraphs.

const SHOW_DELAY_MS = 150;
const HIDE_DELAY_MS = 60;

let tipEl = null;
let showTimer = null;
let hideTimer = null;
let activeTarget = null;

function injectStyles() {
  if (document.getElementById("tooltip-styles")) return;
  const s = document.createElement("style");
  s.id = "tooltip-styles";
  s.textContent = `
    .tooltip { font: 12px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
    .tooltip p { margin: 0 0 6px 0; }
    .tooltip p:last-child { margin-bottom: 0; }
    .tooltip h4, .tooltip h5, .tooltip h6 {
      margin: 8px 0 3px 0; font-size: 12px; font-weight: 600; color: #fff;
    }
    .tooltip h4:first-child, .tooltip h5:first-child, .tooltip h6:first-child { margin-top: 0; }
    .tooltip ul { margin: 2px 0 6px 0; padding-left: 16px; }
    .tooltip ul:last-child { margin-bottom: 0; }
    .tooltip li { margin: 1px 0; }
    .tooltip code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      background: rgba(255,255,255,0.08);
      padding: 0 4px;
      border-radius: 3px;
      color: #f2cf6a;
    }
    .tooltip strong { color: #fff; font-weight: 600; }
    .tooltip em { color: #b8b8b8; font-style: italic; }
  `;
  document.head.appendChild(s);
}

function ensureEl() {
  if (tipEl) return tipEl;
  injectStyles();
  tipEl = document.createElement("div");
  tipEl.className = "tooltip";
  tipEl.style.cssText = [
    "position:fixed",
    "z-index:9999",
    "max-width:420px",
    "padding:8px 11px",
    "max-width:min(420px, calc(100vw - 16px))",
    "background:rgba(18,20,24,0.97)",
    "color:#d6d9dc",
    "border:1px solid #353a42",
    "border-radius:6px",
    "pointer-events:none",
    "opacity:0",
    "transition:opacity 90ms ease",
    "box-shadow:0 8px 24px rgba(0,0,0,0.5)",
  ].join(";");
  tipEl.setAttribute("aria-hidden", "true"); // content is exposed via aria-describedby
  document.body.appendChild(tipEl);
  return tipEl;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function inlineMd(s) {
  // Order matters: code first (so its contents aren't re-parsed), then bold,
  // then italic. We're operating on already-escaped HTML.
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
}

function renderMarkdown(src) {
  const escaped = escapeHtml(src.trim());
  const lines = escaped.split("\n");
  let html = "";
  let inList = false;
  let para = [];
  const flushPara = () => {
    if (para.length) {
      html += "<p>" + inlineMd(para.join(" ")) + "</p>";
      para = [];
    }
  };
  const closeList = () => {
    if (inList) { html += "</ul>"; inList = false; }
  };
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flushPara(); closeList(); continue; }
    let m;
    if ((m = line.match(/^(#{1,3})\s+(.+)$/))) {
      flushPara(); closeList();
      const lvl = Math.min(6, 3 + m[1].length); // # → h4, ## → h5, ### → h6
      html += `<h${lvl}>${inlineMd(m[2])}</h${lvl}>`;
      continue;
    }
    if ((m = line.match(/^[-*]\s+(.+)$/))) {
      flushPara();
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + inlineMd(m[1]) + "</li>";
      continue;
    }
    closeList();
    para.push(line);
  }
  flushPara(); closeList();
  return html;
}

function position(ev) {
  const el = ensureEl();
  const pad = 12;
  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  const r = el.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 4) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 4) y = ev.clientY - r.height - pad;
  el.style.left = Math.max(4, x) + "px";
  el.style.top = Math.max(4, y) + "px";
}

// Anchor below (or above) an element — used for keyboard focus.
function positionAt(anchor) {
  const el = ensureEl();
  const a = anchor.getBoundingClientRect();
  const r = el.getBoundingClientRect();
  let x = Math.min(a.left, window.innerWidth - r.width - 8);
  let y = a.bottom + 8;
  if (y + r.height > window.innerHeight - 4) y = a.top - r.height - 8;
  el.style.left = Math.max(4, x) + "px";
  el.style.top = Math.max(4, y) + "px";
}

let renderedFor = null;
function show(target, ev) {
  const txt = target.getAttribute("data-tooltip");
  if (!txt) return;
  const el = ensureEl();
  if (renderedFor !== txt) { el.innerHTML = renderMarkdown(txt); renderedFor = txt; }
  if (ev) position(ev); else positionAt(target);
  el.style.opacity = "1";
}

function hide() {
  if (tipEl) tipEl.style.opacity = "0";
  activeTarget = null;
}

function findTipTarget(node) {
  // Suppress tooltips while pointing at the slider itself — they get in the
  // way of dragging. The wrapping label still shows on its text/readout.
  if (node && node.tagName === "INPUT" && node.type === "range") return null;
  while (node && node.nodeType === 1) {
    if (node.hasAttribute && node.hasAttribute("data-tooltip")) return node;
    node = node.parentNode;
  }
  return null;
}

// Plain-text version of the markdown for screen readers.
function plainText(md) {
  return md
    .replace(/^#{1,3}\s+/gm, "")
    .replace(/^[-*]\s+/gm, "• ")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/(^|[\s(])\*([^*\s][^*]*?)\*/g, "$1$2")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

let descHost = null;
let descSeq = 0;
function describe(el, text) {
  // The focusable thing the help belongs to: the element itself, or the
  // form control(s) inside a <label>, or the control a <legend> heads.
  let targets = [];
  const tag = el.tagName;
  if (tag === "LABEL") targets = Array.from(el.querySelectorAll("input, select, button"));
  else if (el.matches("button, input, select, textarea, [tabindex]")) targets = [el];
  else {
    // Non-focusable (card titles, BPM readout …): make it focusable so
    // keyboard users can reach the help.
    el.tabIndex = 0;
    el.classList.add("has-tip");
    targets = [el];
  }
  if (!targets.length) return;
  if (!descHost) {
    descHost = document.createElement("div");
    descHost.id = "tooltip-descriptions";
    descHost.hidden = true;
    document.body.appendChild(descHost);
  }
  const d = document.createElement("div");
  d.id = `tipdesc-${++descSeq}`;
  d.textContent = plainText(text);
  descHost.appendChild(d);
  for (const t of targets) {
    const prev = t.getAttribute("aria-describedby");
    t.setAttribute("aria-describedby", prev ? `${prev} ${d.id}` : d.id);
  }
}

function migrateTitles(root) {
  const els = root.querySelectorAll("[title]");
  els.forEach((el) => {
    const t = el.getAttribute("title");
    if (!t) return;
    el.setAttribute("data-tooltip", t);
    el.removeAttribute("title");
    describe(el, t);
  });
}

// For keyboard focus, the tooltip source is the focused element or the
// nearest ancestor carrying data-tooltip (e.g. the <label> around a slider).
function focusTipTarget(node) {
  while (node && node.nodeType === 1) {
    if (node.hasAttribute("data-tooltip")) return node;
    node = node.parentNode;
  }
  return null;
}

export function setupTooltips() {
  migrateTitles(document);

  document.addEventListener("focusin", (ev) => {
    const el = ev.target;
    if (!(el instanceof Element) || !el.matches(":focus-visible")) return;
    const t = focusTipTarget(el);
    if (!t) return;
    activeTarget = t;
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    showTimer = setTimeout(() => show(t, null), SHOW_DELAY_MS * 2);
  });
  document.addEventListener("focusout", () => {
    clearTimeout(showTimer);
    hideTimer = setTimeout(hide, HIDE_DELAY_MS);
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { clearTimeout(showTimer); hide(); }
  });
  window.addEventListener("scroll", () => { clearTimeout(showTimer); hide(); }, true);

  document.addEventListener("mouseover", (ev) => {
    const t = findTipTarget(ev.target);
    if (!t || t === activeTarget) return;
    activeTarget = t;
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    showTimer = setTimeout(() => show(t, ev), SHOW_DELAY_MS);
  });

  document.addEventListener("mousemove", (ev) => {
    if (tipEl && tipEl.style.opacity === "1") position(ev);
  });

  document.addEventListener("mouseout", (ev) => {
    const t = findTipTarget(ev.target);
    if (!t) return;
    const related = findTipTarget(ev.relatedTarget);
    if (related === t) return;
    clearTimeout(showTimer);
    hideTimer = setTimeout(hide, HIDE_DELAY_MS);
  });

  document.addEventListener("mousedown", () => {
    clearTimeout(showTimer);
    hide();
  }, true);
}
