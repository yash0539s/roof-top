/* Kanjikode Roof Classifier — frontend for the FastAPI service in api/main.py. */
"use strict";

const CLASSES = ["ASBESTOS", "METAL", "RCC", "TILED"];
const BASE_FIELDS = ["area_m2", "perimeter_m", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"];
const EXAMPLE = { uid: "IND_B12", area_m2: 451.66, perimeter_m: 102.05, mean_r: 85.06, mean_g: 105.91,
                  mean_b: 156.42, std_r: 12.03, std_g: 10.59, std_b: 9.26 };
const UTM43N = "+proj=utm +zone=43 +datum=WGS84 +units=m +no_defs";
const SOURCE_NAMES = { guesthouse: "Guest house", commercial: "Commercial", industrial: "Industrial", residential: "Residential" };

const $ = (s, el = document) => el.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
const title = (c) => (c === "RCC" ? "RCC" : c.charAt(0) + c.slice(1).toLowerCase());
const colour = (cls) => getComputedStyle(document.documentElement).getPropertyValue(`--${cls.toLowerCase()}`).trim() || "#888";
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const state = {
  config: null, info: null, features: [], layer: null, selected: null, drawLayer: null,
  filters: { source: "", classes: new Set(CLASSES), errorsOnly: false, testOnly: false, minConf: 0 },
};

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "content-type": "application/json" }, ...opts });
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) { const err = new Error(`HTTP ${res.status}`); err.status = res.status; err.body = body; throw err; }
  return body;
}

/* ---------- status ---------- */
async function loadStatus() {
  try {
    const [health, info, cfg] = await Promise.all([api("/health"), api("/model-info"), api("/api/config")]);
    state.info = info; state.config = cfg;
    $("#status-dot").className = `dot ${health.status === "ok" ? "ok" : "down"}`;
    $("#status-text").textContent = `${info.model_version} · test macro-F1 ${info.test_macro_f1?.toFixed(3)}`;
  } catch (e) {
    $("#status-dot").className = "dot down";
    $("#status-text").textContent = "API unreachable";
  }
}

/* ---------- map ---------- */
const map = L.map("map", { zoomControl: true, preferCanvas: false }).setView([10.792, 76.742], 15);
const imagery = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  // Esri has real imagery here only up to z18; deeper zooms upscale it instead of showing placeholders.
  maxZoom: 21, maxNativeZoom: 18, attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
}).addTo(map);
const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 21, maxNativeZoom: 19, attribution: "© OpenStreetMap contributors",
});
L.control.layers({ "Satellite": imagery, "Streets": osm }, null, { position: "topright" }).addTo(map);
L.control.scale({ imperial: false }).addTo(map);

function renderLegend() {
  $("#legend").innerHTML = CLASSES.map((c) => `<div><span class="sw" style="background:${colour(c)}"></span>${title(c)}</div>`).join("")
    + `<div><span class="err-sw"></span>Misclassified</div>`;
}

function styleFor(f) {
  const p = f.properties;
  const wrong = p.correct === 0;
  const sel = state.selected === p.UID;
  return {
    color: sel ? "#ffffff" : wrong ? cssVar("--danger") : "#1b1f24",
    weight: sel ? 3 : wrong ? 2.5 : 0.8,
    fillColor: colour(p.pred_class),
    fillOpacity: 0.35 + 0.4 * (p.confidence ?? 0.5),
    dashArray: p.split === "test" ? "4 3" : null,
  };
}

function passes(p) {
  const f = state.filters;
  return (!f.source || p.SOURCE === f.source) && f.classes.has(p.pred_class)
    && (!f.errorsOnly || p.correct === 0) && (!f.testOnly || p.split === "test")
    && (p.confidence ?? 1) >= f.minConf;
}

function renderBuildings(fit = false) {
  const shown = state.features.filter((f) => passes(f.properties));
  if (state.layer) state.layer.remove();
  state.layer = L.geoJSON({ type: "FeatureCollection", features: shown }, {
    style: styleFor,
    onEachFeature: (f, layer) => {
      const p = f.properties;
      layer.bindTooltip(`<b>${esc(p.UID)}</b> · ${title(p.pred_class)} ${pct(p.confidence)}`, { sticky: true });
      layer.on("click", () => selectBuilding(p.UID, false));
    },
  }).addTo(map);
  if (fit && shown.length) map.fitBounds(state.layer.getBounds(), { padding: [20, 20] });
  renderStats(shown);
}

function renderStats(shown) {
  const labelled = shown.filter((f) => f.properties.correct !== undefined && f.properties.correct !== null);
  const correct = labelled.filter((f) => f.properties.correct === 1).length;
  const test = labelled.filter((f) => f.properties.split === "test");
  const testOk = test.filter((f) => f.properties.correct === 1).length;
  $("#explore-stats").innerHTML = `
    <div class="stat"><b>${shown.length}</b><span>buildings shown</span></div>
    <div class="stat"><b>${labelled.length ? pct(correct / labelled.length) : "–"}</b><span>match field label</span></div>
    <div class="stat"><b>${test.length ? pct(testOk / test.length) : "–"}</b><span>hold-out accuracy</span></div>`;
}

async function loadBuildings() {
  if (!state.config?.buildings_available) {
    showMapMsg("No predictions yet. Run <code>python -m src.predict</code> to populate the map.");
    return;
  }
  const gj = await api("/api/buildings");
  state.features = gj.features;
  const sources = [...new Set(state.features.map((f) => f.properties.SOURCE))].sort();
  $("#f-source").insertAdjacentHTML("beforeend", sources.map((s) => `<option value="${esc(s)}">${esc(SOURCE_NAMES[s] || s)}</option>`).join(""));
  $("#uid-list").innerHTML = state.features.map((f) => `<option value="${esc(f.properties.UID)}">`).join("");
  renderBuildings(true);
}

function showMapMsg(html) { const m = $("#map-msg"); m.innerHTML = html; m.hidden = !html; }

function probBars(probs) {
  return `<div class="probs">${CLASSES.filter((c) => c in probs).sort((a, b) => probs[b] - probs[a]).map((c) => `
    <div class="prob"><span>${title(c)}</span><span class="bar"><i style="width:${(probs[c] * 100).toFixed(1)}%;background:${colour(c)}"></i></span><span class="v">${pct(probs[c], 1)}</span></div>`).join("")}</div>`;
}

function selectBuilding(uid, zoom = true) {
  const f = state.features.find((x) => x.properties.UID === uid);
  if (!f) return;
  state.selected = uid;
  state.layer?.setStyle(styleFor);
  state.layer?.eachLayer((l) => { if (l.feature.properties.UID === uid) l.bringToFront(); });
  if (zoom) map.fitBounds(L.geoJSON(f).getBounds(), { maxZoom: 19, padding: [60, 60], animate: false });
  const p = f.properties;
  const probs = Object.fromEntries(CLASSES.filter((c) => `prob_${c}` in p).map((c) => [c, p[`prob_${c}`]]));
  const hasLabel = p.ROOFTYPE !== undefined && p.ROOFTYPE !== null;
  const verdict = !hasLabel ? "" : p.correct === 1
    ? `<div class="verdict good">Matches the field label (${title(p.ROOFTYPE)}).</div>`
    : `<div class="verdict bad">Field label is <b>${title(p.ROOFTYPE)}</b>, the model says ${title(p.pred_class)}.</div>`;
  const rgb = p.mean_r ? `rgb(${p.mean_r},${p.mean_g},${p.mean_b})` : null;
  const media = [
    p.has_chip ? `<figure><img src="/api/buildings/${encodeURIComponent(uid)}/chip" alt="Orthophoto chip of ${esc(uid)}" loading="lazy"><figcaption>Orthophoto chip</figcaption></figure>` : "",
    p.has_photo ? `<figure><img src="/api/buildings/${encodeURIComponent(uid)}/photo" alt="Field photo of ${esc(uid)}" loading="lazy"><figcaption>Field photo</figcaption></figure>` : "",
  ].join("");
  const d = $("#detail");
  d.classList.remove("empty");
  d.innerHTML = `
    <div class="head"><span class="uid">${esc(uid)}</span><span class="badge" style="background:${colour(p.pred_class)}">${title(p.pred_class)} · ${pct(p.confidence)}</span></div>
    ${verdict}
    ${probBars(probs)}
    <dl class="kv">
      <dt>Area</dt><dd>${esc(SOURCE_NAMES[p.SOURCE] || p.SOURCE)}${p.split ? ` · ${esc(p.split)} set` : ""}</dd>
      ${p.area_m2 ? `<dt>Footprint</dt><dd>${p.area_m2.toFixed(1)} m² · perimeter ${p.perimeter_m.toFixed(1)} m</dd>` : ""}
      ${p.compactness ? `<dt>Shape</dt><dd>compactness ${p.compactness.toFixed(3)} · elongation ${p.elongation.toFixed(2)}</dd>` : ""}
      ${rgb ? `<dt>Mean colour</dt><dd><span class="sw" style="background:${rgb};vertical-align:-1px"></span> ${Math.round(p.mean_r)}, ${Math.round(p.mean_g)}, ${Math.round(p.mean_b)}</dd>` : ""}
      ${p.edge_coherence != null ? `<dt>Texture</dt><dd>edge coherence ${p.edge_coherence.toFixed(2)} · GLCM contrast ${p.glcm_contrast.toFixed(2)}</dd>` : `<dt>Texture</dt><dd>not available (no orthophoto)</dd>`}
      <dt>Model</dt><dd>${esc(p.model_version)}</dd>
    </dl>
    ${media ? `<div class="media">${media}</div>` : ""}
    <div class="actions" style="margin-top:10px"><button type="button" id="to-classify">Open in Classify</button></div>`;
  $("#to-classify").onclick = () => { fillForm(p); switchTab("classify"); };
  d.querySelectorAll(".media img").forEach((img) => {
    img.onerror = () => img.closest("figure").remove();
    img.onclick = () => lightbox(img.src);
  });
}

function lightbox(src) {
  const lb = document.createElement("div");
  lb.className = "lightbox";
  lb.innerHTML = `<img src="${esc(src)}" alt="">`;
  lb.onclick = () => lb.remove();
  document.addEventListener("keydown", function onKey(e) { if (e.key === "Escape") { lb.remove(); document.removeEventListener("keydown", onKey); } });
  document.body.appendChild(lb);
}

/* ---------- filters ---------- */
function setupFilters() {
  $("#f-classes").innerHTML = CLASSES.map((c) => `<span class="chip" data-c="${c}" role="checkbox" aria-checked="true" tabindex="0"><span class="sw" style="background:${colour(c)}"></span>${title(c)}</span>`).join("");
  $("#f-classes").querySelectorAll(".chip").forEach((chip) => {
    const toggle = () => {
      const c = chip.dataset.c, set = state.filters.classes;
      set.has(c) ? set.delete(c) : set.add(c);
      chip.classList.toggle("off", !set.has(c));
      chip.setAttribute("aria-checked", set.has(c));
      renderBuildings();
    };
    chip.onclick = toggle;
    chip.onkeydown = (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); } };
  });
  $("#f-source").onchange = (e) => { state.filters.source = e.target.value; renderBuildings(true); };
  $("#f-errors").onchange = (e) => { state.filters.errorsOnly = e.target.checked; renderBuildings(); };
  $("#f-test").onchange = (e) => { state.filters.testOnly = e.target.checked; renderBuildings(); };
  $("#f-conf").oninput = (e) => { state.filters.minConf = e.target.value / 100; $("#f-conf-out").textContent = `${e.target.value}%`; renderBuildings(); };
  $("#f-search").onchange = (e) => { const uid = e.target.value.trim().toUpperCase(); if (uid) selectBuilding(uid); };
}

/* ---------- tabs ---------- */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => { const on = t.dataset.tab === name; t.classList.toggle("active", on); t.setAttribute("aria-selected", on); });
  document.querySelectorAll(".tabpanel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  syncUploadView(name);
  if (name === "model") renderModel();
}
document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));

/* ---------- classify ---------- */
function setupForm() {
  const tex = state.config?.texture_features || [];
  $("#texture-fields").innerHTML = tex.map((k) => `<div class="field"><label for="${k}">${k.replace(/_/g, " ")}</label><input id="${k}" name="${k}" type="number" step="any"></div>`).join("");
  const form = $("#predict-form");
  form.addEventListener("input", updateSwatch);
  form.addEventListener("submit", async (e) => { e.preventDefault(); await classify(readForm()); });
  $("#fill-example").onclick = () => fillForm(EXAMPLE);
  $("#fill-random").onclick = () => {
    if (!state.features.length) return fillForm(EXAMPLE);
    const f = state.features[Math.floor(Math.random() * state.features.length)];
    fillForm(f.properties);
  };
  $("#draw-btn").onclick = startDraw;
  fillForm(EXAMPLE);
}

function fillForm(p) {
  const form = $("#predict-form");
  form.querySelectorAll("input").forEach((i) => { i.classList.remove("invalid"); });
  form.querySelectorAll(".err").forEach((e) => e.remove());
  $("#uid").value = p.uid ?? p.UID ?? "";
  for (const k of [...BASE_FIELDS, ...(state.config?.texture_features || [])]) {
    const el = form.elements[k];
    if (el) el.value = p[k] ?? "";
  }
  const hasTexture = (state.config?.texture_features || []).some((k) => p[k] != null);
  form.querySelector("details").open = hasTexture;
  updateSwatch();
}

function readForm() {
  const form = $("#predict-form");
  const body = {};
  const uid = $("#uid").value.trim();
  if (uid) body.uid = uid;
  for (const el of form.querySelectorAll("input[type=number]")) {
    if (el.value !== "") body[el.name] = Number(el.value);
  }
  return body;
}

function updateSwatch() {
  const r = +$("#mean_r").value, g = +$("#mean_g").value, b = +$("#mean_b").value;
  $("#swatch").style.background = r || g || b ? `rgb(${r},${g},${b})` : "transparent";
}

function showFieldErrors(detail) {
  const form = $("#predict-form");
  form.querySelectorAll(".err").forEach((e) => e.remove());
  form.querySelectorAll("input").forEach((i) => i.classList.remove("invalid"));
  const general = [];
  for (const d of Array.isArray(detail) ? detail : [{ field: "", message: String(detail) }]) {
    const el = d.field && form.elements[d.field];
    if (el) {
      el.classList.add("invalid");
      el.insertAdjacentHTML("afterend", `<span class="err">${esc(d.message)}</span>`);
    } else general.push(d.message);
  }
  return general;
}

async function classify(body, endpoint = "/predict") {
  const out = $("#predict-result");
  out.hidden = false;
  out.innerHTML = `<p class="meta">Classifying…</p>`;
  const t0 = performance.now();
  try {
    const r = await api(endpoint, { method: "POST", body: JSON.stringify(body) });
    showFieldErrors([]);
    const ms = performance.now() - t0;
    out.innerHTML = `
      <div class="meta">${esc(r.uid || "Unnamed building")}</div>
      <div class="big" style="color:${colour(r.roof_class)}">${title(r.roof_class)}</div>
      <div class="meta">confidence ${pct(r.confidence, 1)}${r.confidence < 0.6 ? " · low, check in the field" : ""}</div>
      ${probBars(r.probabilities)}
      <div class="meta">${esc(r.model_version)} · ${ms.toFixed(0)} ms round trip</div>`;
    return r;
  } catch (e) {
    const general = e.status === 422 ? showFieldErrors(e.body?.detail) : [e.body?.detail || e.message];
    out.innerHTML = `<p class="err">${general.length ? general.map(esc).join("<br>") : "Please fix the highlighted fields."}</p>`;
    return null;
  }
}

/* ---------- draw a roof ---------- */
function startDraw() {
  switchTab("classify");
  if (state.drawLayer) state.drawLayer.remove();
  const drawer = new L.Draw.Polygon(map, { shapeOptions: { color: cssVar("--accent"), weight: 2 }, showArea: true, allowIntersection: false });
  drawer.enable();
  showMapMsg("Click the roof corners, then click the first point to finish.");
}

map.on(L.Draw.Event.CREATED, async (e) => {
  showMapMsg("");
  state.drawLayer = e.layer.addTo(map);
  const ring = e.layer.getLatLngs()[0].map((ll) => proj4("EPSG:4326", UTM43N, [ll.lng, ll.lat]));
  ring.push(ring[0]);
  let area = 0, perim = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1, y1] = ring[i], [x2, y2] = ring[i + 1];
    area += x1 * y2 - x2 * y1;
    perim += Math.hypot(x2 - x1, y2 - y1);
  }
  area = Math.abs(area) / 2;
  $("#uid").value = "DRAWN_1";
  $("#area_m2").value = area.toFixed(2);
  $("#perimeter_m").value = perim.toFixed(2);
  if (state.config?.raster_lookup) {
    const props = { uid: "DRAWN_1" };
    const r = await classify({ type: "Feature", geometry: { type: "Polygon", coordinates: [ring] }, properties: props }, "/predict/geometry");
    if (r) return;
  }
  $("#predict-result").hidden = false;
  $("#predict-result").innerHTML = `<p class="meta">Area ${area.toFixed(1)} m² and perimeter ${perim.toFixed(1)} m filled in from your drawing.
    ${state.config?.raster_lookup ? "The polygon is outside the available orthophotos," : "The server has no orthophotos configured (RASTER_DIR),"}
    so enter the roof's colour statistics and press <b>Classify roof</b>.</p>`;
});
map.on(L.Draw.Event.DRAWSTOP, () => showMapMsg(""));

/* ---------- model tab ---------- */
let modelRendered = false;
async function renderModel() {
  if (modelRendered) return;
  const el = $("#model-body");
  const info = state.info || await api("/model-info").catch(() => null);
  let m = null;
  try { m = await api("/api/metrics"); } catch { /* optional */ }
  if (!info) { el.innerHTML = `<p class="err">Model information unavailable.</p>`; return; }
  modelRendered = true;
  const head = `
    <div class="stats">
      <div class="stat"><b>${info.test_macro_f1?.toFixed(3)}</b><span>test macro-F1</span></div>
      <div class="stat"><b>${pct(info.test_accuracy ?? 0, 1)}</b><span>test accuracy</span></div>
      <div class="stat"><b>${info.cv_macro_f1_mean?.toFixed(3)}</b><span>CV ± ${info.cv_macro_f1_std?.toFixed(3)}</span></div>
    </div>
    <dl class="kv">
      <dt>Version</dt><dd>${esc(info.model_version)}</dd>
      <dt>Algorithm</dt><dd>${esc(info.algorithm)} (probability-calibrated)</dd>
      <dt>Trained</dt><dd>${esc(new Date(info.trained_at).toLocaleString())}</dd>
      <dt>Data version</dt><dd><code>${esc((info.data_md5 || "").slice(0, 12))}</code></dd>
      <dt>Features</dt><dd>${info.features?.length ?? "–"}</dd>
    </dl>`;
  if (!m) { el.innerHTML = head + `<p class="hint">Detailed metrics are not available on this server.</p>`; return; }

  const labels = m.confusion_matrix.labels, mat = m.confusion_matrix.matrix;
  const maxv = Math.max(...mat.flat());
  const cm = `<div class="cm" style="grid-template-columns:70px repeat(${labels.length},1fr)">
      <div class="h">true ↓ / pred →</div>${labels.map((l) => `<div class="h">${title(l)}</div>`).join("")}
      ${mat.map((row, i) => `<div class="h" style="text-align:left">${title(labels[i])}</div>` + row.map((v, j) => {
        const a = maxv ? v / maxv : 0;
        const bg = i === j ? `color-mix(in srgb, ${cssVar("--ok")} ${Math.round(15 + a * 60)}%, transparent)` : v ? `color-mix(in srgb, ${cssVar("--danger")} ${Math.round(20 + a * 60)}%, transparent)` : "var(--surface-2)";
        return `<div style="background:${bg}">${v}</div>`;
      }).join("")).join("")}
    </div>`;
  const perClass = `<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>n</th></tr></thead><tbody>
      ${Object.entries(m.per_class).map(([c, r]) => `<tr><td><span class="sw" style="background:${colour(c)}"></span> ${title(c)}</td><td>${r.precision.toFixed(2)}</td><td>${r.recall.toFixed(2)}</td><td>${r["f1-score"].toFixed(2)}</td><td>${r.support}</td></tr>`).join("")}
    </tbody></table>`;
  const cands = `<table><thead><tr><th>Model (5×5 CV)</th><th>Macro-F1</th><th>± std</th></tr></thead><tbody>
      ${Object.entries(m.candidates).sort((a, b) => b[1].cv_mean - a[1].cv_mean).map(([k, v]) => `<tr${k === m.model ? ' style="font-weight:700"' : ""}><td>${esc(k.replace(/_/g, " "))}${k === m.model ? " ✓" : ""}</td><td>${v.cv_mean.toFixed(3)}</td><td>${v.cv_std.toFixed(3)}</td></tr>`).join("")}
    </tbody></table>`;
  const grouped = `<table><thead><tr><th>Held-out area</th><th>n</th><th>Macro-F1</th><th>Accuracy</th></tr></thead><tbody>
      ${Object.entries(m.grouped_cv_by_source).map(([s, r]) => `<tr><td>${esc(SOURCE_NAMES[s] || s)}${r.classes_unseen_in_training.length ? " *" : ""}</td><td>${r.n}</td><td>${r.macro_f1.toFixed(2)}</td><td>${pct(r.accuracy)}</td></tr>`).join("")}
    </tbody></table>`;
  el.innerHTML = head
    + `<h2>Hold-out test set, per class</h2>${perClass}`
    + `<h2>Confusion matrix</h2>${cm}`
    + `<h2>Model comparison</h2>${cands}`
    + `<h2>Transfer to an unseen area</h2>${grouped}
       <p class="note">Each row trains on the other three areas. Low scores mean the model has not yet learnt to
       generalise to new areas: asbestos (*) only occurs in the industrial area, and the guest-house layer has no
       orthophoto, so it has no texture features. More labels from other areas are the fix.</p>`;
}

/* ---------- upload an image: find and classify every roof ---------- */
const up = { file: null, info: null, viewer: null, overlay: null, roofLayer: null, toPx: 1, view: "image",
             markers: L.layerGroup().addTo(map), result: null, selected: null };
const UP_FIELDS = { latitude: "#up-lat", longitude: "#up-lon", uid: "#up-lat", outline: "#up-info", file: "#up-info" };

function setView(v) {
  up.view = v;
  $("#imgview").hidden = v !== "image";
  document.querySelectorAll("#viewswitch button").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  (v === "image" ? up.viewer : map)?.invalidateSize();
}

function syncUploadView(tab) {
  const show = tab === "upload" && !!up.info;
  $("#viewswitch").hidden = !show;
  $("#imgview").hidden = !(show && up.view === "image");
  if (show && up.view === "image") up.viewer?.invalidateSize();
}

function stepState(id, s) { const el = $(id); el.classList.toggle("active", s !== "todo"); el.classList.toggle("done", s === "done"); }

async function postForm(path, fd) {
  const res = await fetch(path, { method: "POST", body: fd });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) { const e = new Error(`HTTP ${res.status}`); e.status = res.status; e.body = body; throw e; }
  return body;
}

function clearUploadErrors() {
  document.querySelectorAll("#tab-upload .err").forEach((e) => e.remove());
  document.querySelectorAll("#tab-upload input").forEach((i) => i.classList.remove("invalid"));
}

function showUploadErrors(detail) {
  clearUploadErrors();
  const list = Array.isArray(detail) ? detail : [{ field: "file", message: String(detail || "Request failed") }];
  for (const d of list) {
    const el = $(UP_FIELDS[d.field] || "#up-info");
    if (el.tagName === "INPUT") el.classList.add("invalid");
    el.insertAdjacentHTML("afterend", `<span class="err">${esc(d.message)}</span>`);
  }
}

/* pixel [col,row] (original image) <-> viewer latlng (preview, CRS.Simple) */
const pxToLatLng = ([c, r]) => L.latLng(-r / up.toPx, c / up.toPx);
const latLngToPx = (ll) => [+(ll.lng * up.toPx).toFixed(2), +(-ll.lat * up.toPx).toFixed(2)];

async function loadImage(file) {
  if (!file) return;
  clearUploadErrors();
  up.file = file; up.result = null;
  $("#drop-text").innerHTML = `<b>${esc(file.name)}</b><br>${(file.size / 1048576).toFixed(1)} MB · reading…`;
  $("#up-result").hidden = true;
  $("#up-manual").hidden = true;
  const fd = new FormData();
  fd.append("file", file);
  let info;
  try { info = await postForm("/api/image/inspect", fd); }
  catch (e) {
    $("#drop-text").innerHTML = `<b>Choose another image</b><br>JPG, PNG, WebP or GeoTIFF`;
    showUploadErrors(e.body?.detail || e.message);
    return;
  }
  up.info = info;
  $("#drop-text").innerHTML = `<b>${esc(file.name)}</b><br>click to replace`;
  const geo = info.georeferenced;
  $("#up-info").innerHTML = `${info.width} × ${info.height} px · ${geo ? `GeoTIFF, ${info.m_per_px.toFixed(3)} m per pixel — location and size read from the file` : "no georeferencing — enter the location below"}`
    + (info.warnings.length ? `<div class="warn">${info.warnings.map(esc).join("<br>")}</div>` : "");

  const img = new Image();
  img.src = info.preview;
  await img.decode();
  const w = img.naturalWidth, h = img.naturalHeight;
  up.toPx = info.width / w;
  if (!up.viewer) {
    up.viewer = L.map("imgview", { crs: L.CRS.Simple, minZoom: -6, maxZoom: 6, zoomSnap: 0.25, attributionControl: false });
    up.viewer.on(L.Draw.Event.CREATED, onManualOutline);
  }
  $("#imgview").hidden = false;
  up.viewer.invalidateSize();
  up.overlay?.remove(); up.roofLayer?.remove();
  const bounds = [[-h, 0], [0, w]];
  up.overlay = L.imageOverlay(info.preview, bounds).addTo(up.viewer);
  up.roofLayer = L.featureGroup().addTo(up.viewer);
  up.viewer.setMaxBounds(L.latLngBounds(bounds).pad(0.5));
  up.viewer.fitBounds(bounds);

  $("#up-lat").value = geo ? info.center_lat : "";
  $("#up-lon").value = geo ? info.center_lon : "";
  $("#up-submit").disabled = false;
  stepState("#step-file", "done"); stepState("#step-details", "active");
  $("#viewswitch").hidden = false;
  setView("image");
  if (!geo) $("#up-lat").focus();
}

function locationForm() {
  const fd = new FormData();
  fd.append("file", up.file);
  const lat = $("#up-lat").value, lon = $("#up-lon").value;
  if (lat !== "") fd.append("latitude", lat);
  if (lon !== "") fd.append("longitude", lon);
  return fd;
}

async function submitUpload() {
  clearUploadErrors();
  if (!up.info.georeferenced && ($("#up-lat").value === "" || $("#up-lon").value === "")) {
    showUploadErrors([{ field: $("#up-lat").value === "" ? "latitude" : "longitude", message: "required" }]);
    return;
  }
  const out = $("#up-result");
  out.hidden = false;
  out.innerHTML = `<p class="meta">Finding roofs… this takes a few seconds (longer for large GeoTIFFs).</p>`;
  $("#up-submit").disabled = true;
  try {
    const r = await postForm("/predict/image/auto", locationForm());
    up.result = r;
    renderRoofs();
    stepState("#step-details", "done");
    $("#up-manual").hidden = false;
  } catch (e) {
    out.hidden = true;
    showUploadErrors(e.body?.detail || e.message);
  } finally {
    $("#up-submit").disabled = false;
  }
}

function roofTitle(p, i) { return p.uid || `Roof ${i + 1}`; }

function renderRoofs() {
  const r = up.result;
  up.roofLayer.clearLayers();
  up.markers.clearLayers();
  r.roofs.forEach((p, i) => {
    const c = colour(p.roof_class);
    const poly = L.polygon(p.polygon.map(pxToLatLng), { color: c, weight: 2.5, fillColor: c, fillOpacity: 0.28 })
      .bindTooltip(`${i + 1} · ${title(p.roof_class)} ${pct(p.confidence)}`, { permanent: true, direction: "center", className: "class-label", opacity: 1 })
      .on("click", () => selectRoof(i));
    poly.addTo(up.roofLayer);
    const el = poly.getTooltip().getElement();
    if (el) el.style.background = c;
    L.circleMarker([p.latitude, p.longitude], { radius: 7, weight: 2, color: "#fff", fillColor: c, fillOpacity: 0.95 })
      .bindPopup(`<img class="popup-img" src="${p.roof_image}" alt=""><b>${esc(roofTitle(p, i))}</b><br>${title(p.roof_class)} · ${pct(p.confidence)}`)
      .addTo(up.markers);
  });
  const total = r.roofs.length;
  const counts = Object.entries(r.counts).filter(([, n]) => n > 0);
  $("#up-result").innerHTML = `
    <div class="big">${total} roof${total === 1 ? "" : "s"} found</div>
    <div class="chips" style="margin:8px 0">${counts.map(([c, n]) => `<span class="chip"><span class="sw" style="background:${colour(c)}"></span>${title(c)} · ${n}</span>`).join("") || '<span class="meta">none</span>'}</div>
    <dl class="kv">
      <dt>Location</dt><dd>${r.latitude.toFixed(6)}, ${r.longitude.toFixed(6)}</dd>
      <dt>Roof size</dt><dd>${r.georeferenced ? `known (${r.m_per_px.toFixed(3)} m per pixel) · main model` : "unknown · size-free model"}</dd>
      <dt>Detector</dt><dd>${esc(r.detector)} · ${r.seconds.toFixed(1)} s</dd>
    </dl>
    ${r.warnings.length ? `<div class="warn"><b>Check before relying on this</b><ul>${r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>` : ""}
    <div class="actions" style="margin:10px 0">
      <a class="button primary" id="up-download" download="${esc((up.file.name || "image").replace(/\.[^.]+$/, ""))}_roofs.png" href="${r.labelled_image}">Download labelled image</a>
      <button type="button" id="up-csv">Download CSV</button>
      <button type="button" id="up-show">Show on map</button>
    </div>
    <div class="roof-list" id="roof-list">${r.roofs.map((p, i) => `
      <div class="roof-item" data-i="${i}">
        <img src="${p.roof_image}" alt="" style="border-color:${colour(p.roof_class)}">
        <div>
          <b>${i + 1}. ${title(p.roof_class)}</b> <span class="meta">${pct(p.confidence)}${p.confidence < 0.6 ? " · uncertain" : ""}</span>
          <div class="meta">${p.area_m2 != null ? `${p.area_m2.toFixed(0)} m² · ` : ""}${p.latitude.toFixed(5)}, ${p.longitude.toFixed(5)}</div>
        </div>
      </div>`).join("")}</div>`;
  $("#up-result").hidden = false;
  $("#roof-list").querySelectorAll(".roof-item").forEach((el) => (el.onclick = () => selectRoof(+el.dataset.i)));
  $("#up-show").onclick = () => { setView("map"); map.setView([r.latitude, r.longitude], 18, { animate: false }); };
  $("#up-csv").onclick = downloadCsv;
  if (total) up.viewer.fitBounds(up.roofLayer.getBounds().pad(0.1));
}

function selectRoof(i) {
  const p = up.result.roofs[i];
  setView("image");
  up.viewer.fitBounds(L.polygon(p.polygon.map(pxToLatLng)).getBounds().pad(0.6), { animate: false });
  document.querySelectorAll(".roof-item").forEach((el) => el.classList.toggle("sel", +el.dataset.i === i));
  const d = $("#roof-list").children[i];
  d?.scrollIntoView({ block: "nearest" });
  up.roofLayer.getLayers().forEach((l, j) => l.setStyle({ weight: j === i ? 5 : 2.5 }));
}

function downloadCsv() {
  const cols = ["n", "uid", "roof_class", "confidence", "prob_ASBESTOS", "prob_METAL", "prob_RCC", "prob_TILED",
                "latitude", "longitude", "area_m2", "size_known", "model_version"];
  const rows = up.result.roofs.map((p, i) => [i + 1, p.uid ?? "", p.roof_class, p.confidence,
    ...["ASBESTOS", "METAL", "RCC", "TILED"].map((c) => p.probabilities[c] ?? ""),
    p.latitude, p.longitude, p.area_m2 ?? "", p.size_known, p.model_version]);
  const csv = [cols, ...rows].map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = `${(up.file.name || "image").replace(/\.[^.]+$/, "")}_roofs.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function onManualOutline(e) {
  const fd = locationForm();
  fd.append("outline", JSON.stringify(e.layer.getLatLngs()[0].map(latLngToPx)));
  fd.append("uid", `ROOF_${(up.result?.roofs.length || 0) + 1}`);
  try {
    const p = await postForm("/predict/image", fd);
    up.result = up.result || { roofs: [], counts: {}, latitude: p.latitude, longitude: p.longitude,
      georeferenced: up.info.georeferenced, m_per_px: up.info.m_per_px, detector: "hand-drawn", seconds: 0,
      warnings: p.warnings, labelled_image: up.info.preview };
    up.result.roofs.push(p);
    up.result.counts[p.roof_class] = (up.result.counts[p.roof_class] || 0) + 1;
    renderRoofs();
    selectRoof(up.result.roofs.length - 1);
  } catch (err) {
    showUploadErrors(err.body?.detail || err.message);
  }
}

function setupUpload() {
  const input = $("#up-file"), drop = $("#drop");
  input.onchange = () => loadImage(input.files[0]);
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => loadImage(e.dataTransfer.files[0]));
  $("#up-draw").onclick = () => {
    setView("image");
    new L.Draw.Polygon(up.viewer, { shapeOptions: { color: cssVar("--accent"), weight: 2 }, allowIntersection: false }).enable();
  };
  $("#up-pick").onclick = () => {
    setView("map");
    map.getContainer().classList.add("pick-cursor");
    showMapMsg("Click the image's location on the map.");
    map.once("click", (e) => {
      $("#up-lat").value = e.latlng.lat.toFixed(6);
      $("#up-lon").value = e.latlng.lng.toFixed(6);
      map.getContainer().classList.remove("pick-cursor");
      showMapMsg("");
      if (up.info) setView("image");
    });
  };
  document.querySelectorAll("#viewswitch button").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
  $("#up-submit").onclick = submitUpload;
}

/* ---------- boot ---------- */
(async function init() {
  proj4.defs("EPSG:32643", UTM43N);
  renderLegend();
  setupFilters();
  await loadStatus();
  setupForm();
  setupUpload();
  try { await loadBuildings(); } catch (e) { showMapMsg("Could not load buildings."); }
  setInterval(loadStatus, 30000);
})();
