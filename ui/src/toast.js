// Transient toasts (bottom-right). `kind` is "ok" | "err" | "info".
// Announced to screen readers through an aria-live region. At most
// MAX_TOASTS stack; identical messages in quick succession are collapsed
// into one with a ×N counter (a drag that keeps hitting a validator error
// shouldn't bury the page).

const MAX_TOASTS = 4;
const TTL_MS = { ok: 2200, info: 2800, err: 5000 };

let host = null;

function ensureHost() {
  if (host) return host;
  host = document.getElementById("toasts");
  if (!host) {
    host = document.createElement("div");
    host.id = "toasts";
    document.body.appendChild(host);
  }
  host.className = "toasts";
  host.setAttribute("role", "status");
  host.setAttribute("aria-live", "polite");
  return host;
}

export function toast(message, kind = "info") {
  const h = ensureHost();
  const last = h.lastElementChild;
  if (last && last.dataset.msg === message && last.dataset.kind === kind && !last.classList.contains("leaving")) {
    const n = (parseInt(last.dataset.count, 10) || 1) + 1;
    last.dataset.count = String(n);
    last.querySelector(".toast-count").textContent = `×${n}`;
    clearTimeout(last._timer);
    last._timer = setTimeout(() => dismiss(last), TTL_MS[kind] || 3000);
    return;
  }
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.dataset.msg = message;
  el.dataset.kind = kind;
  const text = document.createElement("span");
  text.className = "toast-text";
  text.textContent = message;
  const count = document.createElement("span");
  count.className = "toast-count";
  el.append(text, count);
  el.addEventListener("click", () => dismiss(el));
  h.appendChild(el);
  while (h.children.length > MAX_TOASTS) h.firstElementChild.remove();
  el._timer = setTimeout(() => dismiss(el), TTL_MS[kind] || 3000);
}

function dismiss(el) {
  if (!el.isConnected || el.classList.contains("leaving")) return;
  clearTimeout(el._timer);
  el.classList.add("leaving");
  setTimeout(() => el.remove(), 180);
}
