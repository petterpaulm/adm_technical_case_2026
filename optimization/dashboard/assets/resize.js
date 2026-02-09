/**
 * Sidebar drag-to-resize handler.
 *
 * Uses document-level event delegation so it survives Dash/React
 * re-renders without needing to re-attach listeners.
 *
 * Constraints:  350 px  ≤  width  ≤  85 vw
 * Persists in sessionStorage.
 */
(function () {
  "use strict";

  var MIN_W = 350;
  var MAX_RATIO = 0.85;
  var KEY = "adm_sidebar_width";

  var dragging = false;
  var startX = 0;
  var startW = 0;

  function sidebar() { return document.getElementById("dash-sidebar"); }

  function isHandle(el) {
    while (el) {
      if (el.id === "sidebar-resize-handle") return true;
      el = el.parentElement;
    }
    return false;
  }

  function clamp(w) {
    return Math.min(Math.max(w, MIN_W), window.innerWidth * MAX_RATIO);
  }

  function applyWidth(sb, w) {
    sb.style.width = w + "px";
  }

  /* ---------- mousedown ---------- */
  document.addEventListener("mousedown", function (e) {
    if (e.button !== 0 || !isHandle(e.target)) return;
    var sb = sidebar();
    if (!sb || sb.classList.contains("collapsed")) return;

    e.preventDefault();
    e.stopPropagation();

    dragging = true;
    startX = e.clientX;
    startW = sb.getBoundingClientRect().width;

    sb.style.transition = "none";
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.body.style.webkitUserSelect = "none";

    var h = document.getElementById("sidebar-resize-handle");
    if (h) h.classList.add("active");
  });

  /* ---------- mousemove ---------- */
  document.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    var sb = sidebar();
    if (!sb) return;
    e.preventDefault();

    var newW = clamp(startW + (startX - e.clientX));
    applyWidth(sb, newW);
  });

  /* ---------- mouseup ---------- */
  document.addEventListener("mouseup", function () {
    if (!dragging) return;
    dragging = false;

    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    document.body.style.webkitUserSelect = "";

    var sb = sidebar();
    if (sb) {
      sb.style.transition = "";
      sessionStorage.setItem(KEY, Math.round(sb.getBoundingClientRect().width));
    }
    var h = document.getElementById("sidebar-resize-handle");
    if (h) h.classList.remove("active");

    window.dispatchEvent(new Event("resize"));
  });

  /* ---------- collapse / expand patch ---------- */
  function watchCollapse() {
    var sb = sidebar();
    if (!sb) { setTimeout(watchCollapse, 600); return; }

    new MutationObserver(function (muts) {
      if (dragging) return;                          // never interfere mid-drag
      muts.forEach(function (m) {
        if (m.attributeName !== "class") return;
        var s = sidebar();
        if (!s) return;
        if (s.classList.contains("collapsed")) {
          var w = s.getBoundingClientRect().width || s.offsetWidth;
          s.style.transform = "translateX(" + w + "px)";
        } else {
          s.style.transform = "translateX(0)";
          // Re-apply saved width when un-collapsing
          var saved = sessionStorage.getItem(KEY);
          if (saved) applyWidth(s, clamp(parseInt(saved, 10)));
        }
      });
    }).observe(sb, { attributes: true, attributeFilter: ["class"] });
  }

  /* ---------- restore on Dash navigation ---------- */
  function restoreIfNeeded() {
    if (dragging) return;                            // NEVER during drag
    var sb = sidebar();
    if (!sb) return;
    var saved = sessionStorage.getItem(KEY);
    if (!saved) return;
    var want = clamp(parseInt(saved, 10));
    var have = Math.round(sb.getBoundingClientRect().width);
    // Only touch if Dash reset the width (differs by > 5 px)
    if (Math.abs(have - want) > 5) {
      applyWidth(sb, want);
    }
  }

  /* Debounced observer — only fires 200 ms after the LAST mutation,
     so it won't fight the drag's rapid style changes. */
  var restoreTimer = null;
  function watchRenders() {
    var target = document.getElementById("_dash-app-content") || document.body;
    new MutationObserver(function () {
      clearTimeout(restoreTimer);
      restoreTimer = setTimeout(restoreIfNeeded, 200);
    }).observe(target, { childList: true, subtree: true });
  }

  /* ---------- boot ---------- */
  function boot() {
    restoreIfNeeded();
    watchCollapse();
    watchRenders();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
