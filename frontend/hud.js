"use strict";

// Task 16: DriveSense HUD. Vanilla JS, no build step. Connects to /ws/stream
// (Task 10's relay) and renders whatever it broadcasts:
//   {"type":"frame","data":"<base64 JPEG>","timestamp":float}
//   {"type":"event","alert":bool,"reason":str|null,"ego_stationary":bool,"timestamp":float}
//   {"type":"audio","reason":str,"data":"<base64 MP3>"}   -- may never arrive (no ElevenLabs key)

const REASON_LABEL = {
  light_green: "LIGHT IS GREEN — GO",
  lead_accelerating: "CAR AHEAD IS MOVING",
};

const BANNER_HOLD_MS = 2500;
const STALL_MS = 2000;
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 8000;

// One silent WAV sample, used only to unlock autoplay inside the start
// button's click handler -- the one moment the browser counts as a user
// gesture. Real alert audio replaces this element's src later.
const SILENT_WAV =
  "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA";

const els = {
  connDot: document.getElementById("conn-dot"),
  connLabel: document.getElementById("conn-label"),
  fps: document.getElementById("fps"),
  ego: document.getElementById("ego"),
  sound: document.getElementById("sound"),
  canvas: document.getElementById("canvas"),
  banner: document.getElementById("banner"),
  readout: document.getElementById("readout"),
  startOverlay: document.getElementById("start-overlay"),
  startBtn: document.getElementById("start-btn"),
};

const ctx = els.canvas.getContext("2d");

let ws = null;
let reconnectDelay = RECONNECT_MIN_MS;
let reconnectTimer = null;
let lastFrameAt = 0;
let stallTimer = null;

let pendingFrame = null;
let decoding = false;

let frameCount = 0;
let fpsTimer = null;

let bannerHideTimer = null;

let audioEl = null;

let egoStationary = true;

function setConn(state) {
  els.connDot.className = state;
  els.connLabel.textContent = state.toUpperCase();
}

function b64ToBlob(b64, mime) {
  const bytes = atob(b64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

function drawBitmap(bmp) {
  if (els.canvas.width !== bmp.width || els.canvas.height !== bmp.height) {
    els.canvas.width = bmp.width;
    els.canvas.height = bmp.height;
  }
  ctx.drawImage(bmp, 0, 0);
  frameCount++;
}

async function pumpFrame() {
  if (decoding || pendingFrame === null) return;
  decoding = true;
  const b64 = pendingFrame;
  pendingFrame = null;
  try {
    const blob = b64ToBlob(b64, "image/jpeg");
    const bmp = await createImageBitmap(blob);
    drawBitmap(bmp);
    bmp.close();
  } catch (err) {
    // Corrupt/partial frame -- skip it, keep the stream alive.
    console.warn("frame decode failed", err);
  } finally {
    decoding = false;
    if (pendingFrame !== null) pumpFrame();
  }
}

function onFrame(msg) {
  const now = performance.now();
  const wasStalled = now - lastFrameAt > STALL_MS;
  lastFrameAt = now;
  if (wasStalled && ws && ws.readyState === WebSocket.OPEN) setConn("live");
  clearTimeout(stallTimer);
  stallTimer = setTimeout(() => {
    if (ws && ws.readyState === WebSocket.OPEN) setConn("stalled");
  }, STALL_MS);

  pendingFrame = msg.data;
  pumpFrame();
}

function setEgo(stationary) {
  if (stationary === egoStationary) return;
  egoStationary = stationary;
  els.ego.textContent = stationary ? "STOPPED" : "MOVING";
  els.ego.className = stationary ? "" : "moving";
}

function showBanner(reason) {
  const label = REASON_LABEL[reason] || "GO";
  els.banner.textContent = label;
  els.banner.classList.add("show");
  clearTimeout(bannerHideTimer);
  bannerHideTimer = setTimeout(() => {
    els.banner.classList.remove("show");
  }, BANNER_HOLD_MS);

  const t = new Date((msgTimestampOrNow()) * 1000);
  els.readout.textContent = `LAST: ${reason ? reason.replace(/_/g, " ") : "alert"} · ${t.toLocaleTimeString()}`;
}

let _lastEventTimestamp = null;
function msgTimestampOrNow() {
  return _lastEventTimestamp != null ? _lastEventTimestamp : Date.now() / 1000;
}

function onEvent(msg) {
  _lastEventTimestamp = msg.timestamp;
  setEgo(!!msg.ego_stationary);
  if (msg.alert) showBanner(msg.reason);
}

function onAudio(msg) {
  if (!audioEl) return; // not yet unlocked (shouldn't happen post-start-click)
  audioEl.src = "data:audio/mpeg;base64," + msg.data;
  audioEl.currentTime = 0;
  audioEl.play().catch((err) => console.warn("audio play failed", err));
}

function tickFps() {
  els.fps.textContent = `${frameCount} fps`;
  frameCount = 0;
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/stream`;
}

function connect() {
  if (ws && ws.readyState <= WebSocket.OPEN) return; // already connecting/open

  setConn("reconnecting");
  ws = new WebSocket(wsUrl());

  ws.onopen = () => {
    reconnectDelay = RECONNECT_MIN_MS;
    setConn("stalled"); // live once the first frame actually arrives
    lastFrameAt = 0;
  };

  ws.onmessage = (evt) => {
    let msg;
    try {
      msg = JSON.parse(evt.data);
    } catch (err) {
      return;
    }
    if (msg.type === "frame") onFrame(msg);
    else if (msg.type === "event") onEvent(msg);
    else if (msg.type === "audio") onAudio(msg);
  };

  ws.onclose = scheduleReconnect;
  ws.onerror = () => ws.close();
}

function scheduleReconnect() {
  setConn("reconnecting");
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    reconnectDelay = Math.min(reconnectDelay * 1.7, RECONNECT_MAX_MS);
    connect();
  }, reconnectDelay);
}

function start() {
  els.startOverlay.classList.add("hidden");

  audioEl = new Audio();
  audioEl.src = SILENT_WAV;
  audioEl.play().catch(() => {});
  els.sound.classList.add("armed");

  fpsTimer = setInterval(tickFps, 1000);
  connect();
}

els.startBtn.addEventListener("click", start, { once: true });
