"use strict";
const $ = (s) => document.querySelector(s);
const video = $("#video"), serverImg = $("#server-img"), uploadImg = $("#upload-img");
const overlay = $("#overlay"), octx = overlay.getContext("2d");
const grab = document.createElement("canvas");
const MAX_SIDE = 1280, FRAME_GAP_MS = 120;

let source = "browser", stream = null, looping = false, lastEventId = null, lastShownId = null;

// ---------- helpers ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const plateHtml = (display, small = false) =>
  `<span class="ge-plate${small ? " small" : ""}"><span class="eu"><span class="flag"></span>GE</span><span class="num">${esc(display)}</span></span>`;
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString("en-GB");
const DECISION = { granted: "Access granted", denied: "Not in database", manual_open: "Opened manually", manual_close: "Closed manually" };

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

let audio;
function beep(ok) {
  try {
    audio = audio || new AudioContext();
    const tones = ok ? [880, 1320] : [300, 220];
    tones.forEach((f, i) => {
      const o = audio.createOscillator(), g = audio.createGain();
      o.frequency.value = f; o.connect(g); g.connect(audio.destination);
      const t = audio.currentTime + i * 0.14;
      g.gain.setValueAtTime(0.08, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.13);
      o.start(t); o.stop(t + 0.14);
    });
  } catch {}
}

// ---------- overlay ----------
const STATE = { reading: "confirming", granted: "granted", denied: "not in database" };

function drawReads(data, mediaW, mediaH) {
  overlay.width = mediaW; overlay.height = mediaH;
  octx.clearRect(0, 0, mediaW, mediaH);
  if (!data) return renderReads([]);
  const reads = data.reads || [];
  const kx = mediaW / data.w, ky = mediaH / data.h;
  for (const r of reads) {
    const [x1, y1, x2, y2] = r.box;
    const color = r.state === "granted" ? "#22c55e" : r.state === "denied" ? "#ef4444" : "#f59e0b";
    octx.lineWidth = Math.max(3, mediaW / 300);
    octx.strokeStyle = color;
    octx.strokeRect(x1 * kx, y1 * ky, (x2 - x1) * kx, (y2 - y1) * ky);
    const label = r.state === "reading"
      ? `${r.display || r.raw || "?"} · ${Math.min(r.agree, r.need)}/${r.need}`
      : r.display;
    octx.font = `bold ${Math.max(16, mediaW / 40)}px -apple-system, sans-serif`;
    const tw = octx.measureText(label).width + 12, th = Math.max(22, mediaW / 32);
    const ly = Math.max(0, y1 * ky - th - 4);
    octx.fillStyle = color; octx.fillRect(x1 * kx, ly, tw, th);
    octx.fillStyle = "#000"; octx.fillText(label, x1 * kx + 6, ly + th * 0.75);
  }
  renderReads(reads);
}

function renderReads(reads) {
  $("#reads").innerHTML = reads.length
    ? reads.map((r) => `<span class="read">frame <b>${esc(r.frame_plate || r.raw || "—")}</b> ${Math.round(r.frame_conf * 100)}%
        · car #${r.track}: <b>${esc(r.display || "—")}</b> ${Math.round(r.conf * 100)}%, ${r.agree} of ${r.votes} frames agree (need ${r.need})
        · ${STATE[r.state] || r.state}${r.region ? ` · ${esc(r.region)}` : ""}</span>`).join("")
    : `<span class="read">no plate in frame</span>`;
}

// ---------- browser camera ----------
async function listCameras() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const sel = $("#cam-select"), cur = sel.value;
  sel.innerHTML = devs.map((d, i) => `<option value="${d.deviceId}">${esc(d.label || "Camera " + (i + 1))}</option>`).join("");
  if (cur) sel.value = cur;
  sel.hidden = source !== "browser" || devs.length < 2;
}

async function startBrowserCamera(deviceId) {
  stopBrowserCamera();
  $("#stage-msg").textContent = "Allow camera access in the browser";
  $("#stage-msg").hidden = false;
  if (!navigator.mediaDevices?.getUserMedia) {
    $("#stage-msg").textContent = "The camera only works on http://localhost or over HTTPS";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1920 }, height: { ideal: 1080 }, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) },
      audio: false,
    });
  } catch (e) {
    $("#stage-msg").textContent = "No camera access: " + e.message;
    return;
  }
  video.srcObject = stream;
  await video.play();
  $("#stage-msg").hidden = true;
  listCameras();
  if (!looping) { looping = true; browserLoop(); }
}

function stopBrowserCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null; video.srcObject = null;
}

async function browserLoop() {
  while (source === "browser" && stream) {
    const t0 = performance.now();
    try {
      const vw = video.videoWidth, vh = video.videoHeight;
      if (!vw) { await sleep(200); continue; }
      const k = Math.min(1, MAX_SIDE / Math.max(vw, vh));
      grab.width = Math.round(vw * k); grab.height = Math.round(vh * k);
      grab.getContext("2d").drawImage(video, 0, 0, grab.width, grab.height);
      const blob = await new Promise((res) => grab.toBlob(res, "image/jpeg", 0.88));
      const data = await api("/api/recognize?source=browser-camera", { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
      if (source !== "browser") break;
      drawReads(data, vw, vh);
      onRecognized(data, performance.now() - t0);
    } catch (e) {
      setConn(false, e.message);
      await sleep(1000);
    }
    await sleep(FRAME_GAP_MS);
  }
  looping = false;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function onRecognized(data, rtt) {
  setConn(true);
  $("#perf").textContent = `recognition ${data.ms} ms · loop ${Math.round(rtt)} ms`;
  applyGate(data.gate);
  if (data.events?.length) refreshEvents();
}

// ---------- upload ----------
async function recognizeFile(file) {
  if (!file) return;
  uploadImg.src = URL.createObjectURL(file);
  await uploadImg.decode().catch(() => {});
  uploadImg.hidden = false;
  $("#upload-drop").classList.add("has-img");
  $("#upload-drop span").textContent = "Another photo";
  const t0 = performance.now();
  try {
    const data = await api("/api/recognize?source=upload", { method: "POST", body: file });
    drawReads(data, uploadImg.naturalWidth, uploadImg.naturalHeight);
    onRecognized(data, performance.now() - t0);
  } catch (e) {
    renderReads([]); $("#reads").innerHTML = `<span class="read">Error: ${esc(e.message)}</span>`;
  }
}

// ---------- source switching ----------
function setSource(src) {
  source = src;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.src === src));
  video.hidden = src !== "browser";
  serverImg.hidden = src !== "server";
  uploadImg.hidden = src !== "upload" || !uploadImg.src;
  $("#upload-drop").hidden = src !== "upload";
  $("#stage-msg").hidden = true;
  $("#cam-select").hidden = true;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  renderReads([]);
  if (src === "browser") startBrowserCamera($("#cam-select").value || undefined);
  else stopBrowserCamera();
  if (src === "server") serverImg.src = "/api/camera/stream?t=" + Date.now();
  else serverImg.removeAttribute("src");
}

// ---------- gate / state ----------
function applyGate(g) {
  if (!g) return;
  $("#arm").classList.toggle("open", g.open);
  $("#tl-green").classList.toggle("on", g.open);
  $("#tl-red").classList.toggle("on", !g.open);
  const last = g.last;
  const banner = $("#banner");
  if (!last) return;
  if (last.id !== lastShownId) {
    lastShownId = last.id;
    if (last.decision === "granted" || last.decision === "denied") beep(last.decision === "granted");
  }
  const recent = Date.now() / 1000 - last.ts < 15 || g.open;
  if (!recent) {
    banner.className = "banner idle";
    $("#banner-title").textContent = "Waiting for a car";
    $("#banner-plate").innerHTML = ""; $("#banner-sub").textContent = ""; $("#banner-add").hidden = true;
    return;
  }
  const kind = last.decision.startsWith("manual") ? "manual" : last.decision;
  banner.className = "banner " + kind;
  $("#banner-title").textContent = DECISION[last.decision] || last.decision;
  $("#banner-plate").innerHTML = last.display ? plateHtml(last.display) : "";
  const parts = [];
  if (last.owner) parts.push(last.owner);
  if (g.open) parts.push(`closes in ${Math.ceil(g.closes_in)} s`);
  $("#banner-sub").textContent = parts.join(" · ");
  const add = $("#banner-add");
  add.hidden = last.decision !== "denied";
  add.onclick = () => { const f = $("#plate-form"); f.plate.value = last.display; f.owner.focus(); };
}

function setConn(ok, msg) {
  const el = $("#conn");
  el.className = "pill " + (ok ? "ok" : "bad");
  el.textContent = ok ? "server online" : "offline" + (msg ? `: ${msg}` : "");
}

async function pollState() {
  try {
    const s = await api("/api/state");
    setConn(true);
    applyGate(s.gate);
    $("#tab-server").hidden = !s.server_camera.enabled;
    if (source === "server") {
      const last = s.server_camera.last;
      if (serverImg.naturalWidth) drawReads(last, serverImg.naturalWidth, serverImg.naturalHeight);
      $("#perf").textContent = "server camera: " + s.server_camera.status;
    }
    if (s.gate.last && s.gate.last.id !== lastEventId) refreshEvents();
  } catch (e) {
    setConn(false, e.message);
  }
}

// ---------- events ----------
async function refreshEvents() {
  const evs = await api("/api/events?limit=100");
  lastEventId = evs[0]?.id ?? null;
  $("#events").innerHTML = evs.map((e) => `
    <tr>
      <td>${fmtTime(e.ts)}</td>
      <td>${e.plate && e.plate !== "—" ? plateHtml(e.display, true) : "—"}</td>
      <td>${e.has_crop ? `<img src="/api/events/${e.id}/crop.jpg" alt="">` : ""}</td>
      <td><span class="tag ${e.decision}">${DECISION[e.decision] || e.decision}</span></td>
      <td>${esc(e.owner)}</td>
      <td>${e.decision === "denied" ? `<button class="btn link" data-add="${esc(e.display)}" title="Add to database">＋</button>` : ""}</td>
    </tr>`).join("") || `<tr><td colspan="6" class="muted">No events yet</td></tr>`;
}

$("#events").addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-add]");
  if (!b) return;
  const f = $("#plate-form"); f.plate.value = b.dataset.add; f.owner.focus();
});

// ---------- plates ----------
async function refreshPlates() {
  const list = await api("/api/plates");
  $("#plates-count").textContent = `(${list.length})`;
  $("#plates").innerHTML = list.map((p) => `
    <li>
      ${plateHtml(p.display, true)}
      <span class="who">${esc(p.owner) || "<small>no owner</small>"} ${p.note ? `<small>· ${esc(p.note)}</small>` : ""}</span>
      <button class="btn link" data-del="${esc(p.plate)}" title="Remove">✕</button>
    </li>`).join("") || `<li class="muted">Database is empty</li>`;
}

$("#plates").addEventListener("click", async (ev) => {
  const b = ev.target.closest("[data-del]");
  if (!b || !confirm(`Remove ${b.dataset.del} from the database?`)) return;
  await api(`/api/plates/${encodeURIComponent(b.dataset.del)}`, { method: "DELETE" });
  refreshPlates();
});

$("#plate-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target;
  $("#form-err").textContent = "";
  try {
    await api("/api/plates", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plate: f.plate.value, owner: f.owner.value, note: f.note.value }),
    });
    f.reset();
    refreshPlates();
  } catch (e) {
    $("#form-err").textContent = e.message;
  }
});

// ---------- wiring ----------
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.src)));
$("#cam-select").addEventListener("change", (e) => startBrowserCamera(e.target.value));
$("#file").addEventListener("change", (e) => recognizeFile(e.target.files[0]));
const stage = $("#stage");
stage.addEventListener("dragover", (e) => { if (source === "upload") e.preventDefault(); });
stage.addEventListener("drop", (e) => { if (source !== "upload") return; e.preventDefault(); recognizeFile(e.dataTransfer.files[0]); });
$("#btn-open").addEventListener("click", async () => applyGate(await api("/api/gate/open", { method: "POST" })));
$("#btn-close").addEventListener("click", async () => applyGate(await api("/api/gate/close", { method: "POST" })));

refreshPlates();
refreshEvents();
pollState();
setInterval(pollState, 700);
setSource("browser");
