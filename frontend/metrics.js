"use strict";

// Task 17: DriveSense metrics dashboard. Polls GET /api/metrics/summary
// (Task 17's endpoint, server/app.py + db/metrics.py) every 5s and renders
// two Chart.js charts + a table view of each under it.

const POLL_MS = 5000;

// Fixed mapping, matches the CSS custom properties in metrics.css -- see
// that file's comment for why these colors (not the HUD's green/amber) were
// chosen: validated with the dataviz skill's validate_palette.js.
const REASON_COLOR = {
  light_green: getVar("--series-light-green"),
  lead_accelerating: getVar("--series-lead-accel"),
};
const REASON_LABEL = {
  light_green: "Light Green",
  lead_accelerating: "Lead Accelerating",
};

function getVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function colorFor(reason) {
  return REASON_COLOR[reason] || "#8a93a1";
}

function labelFor(reason) {
  return REASON_LABEL[reason] || reason || "unknown";
}

const els = {
  banner: document.getElementById("db-banner"),
  total: document.getElementById("tile-total"),
  avgReaction: document.getElementById("tile-avg-reaction"),
  timelineTableBody: document.querySelector("#table-timeline tbody"),
  reactionTableBody: document.querySelector("#table-reaction tbody"),
};

let timelineChart = null;
let reactionChart = null;

function buildTimelineChart() {
  const ctx = document.getElementById("chart-timeline").getContext("2d");
  timelineChart = new Chart(ctx, {
    type: "bar",
    data: { labels: [], datasets: [
      { label: labelFor("light_green"), backgroundColor: colorFor("light_green"), data: [], stack: "alerts" },
      { label: labelFor("lead_accelerating"), backgroundColor: colorFor("lead_accelerating"), data: [], stack: "alerts" },
    ] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { stacked: true, ticks: { color: "#8a93a1" }, grid: { color: "#262c36" } },
        y: { stacked: true, beginAtZero: true, ticks: { color: "#8a93a1", precision: 0 }, grid: { color: "#262c36" } },
      },
    },
  });
}

function buildReactionChart() {
  const ctx = document.getElementById("chart-reaction").getContext("2d");
  reactionChart = new Chart(ctx, {
    type: "bar",
    data: { labels: [], datasets: [
      { label: "Reaction time (ms)", data: [], backgroundColor: [] },
    ] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel(ctx) {
              const iso = ctx.dataset.timestamps?.[ctx.dataIndex];
              return iso ? new Date(iso).toLocaleTimeString() : "";
            },
          },
        },
      },
      scales: {
        x: { ticks: { color: "#8a93a1" }, grid: { color: "#262c36" } },
        y: { beginAtZero: true, ticks: { color: "#8a93a1" }, grid: { color: "#262c36" } },
      },
    },
  });
}

function updateTiles(data) {
  els.total.textContent = data.total_alerts;
  els.avgReaction.textContent =
    data.avg_reaction_ms == null ? "--" : `${Math.round(data.avg_reaction_ms)} ms`;
}

function updateTimeline(data) {
  const buckets = [...new Set(data.events_timeline.map((r) => r.bucket))].sort();
  const byReasonBucket = { light_green: {}, lead_accelerating: {} };
  for (const row of data.events_timeline) {
    if (!byReasonBucket[row.reason]) byReasonBucket[row.reason] = {};
    byReasonBucket[row.reason][row.bucket] = row.count;
  }

  timelineChart.data.labels = buckets.map((b) => new Date(b).toLocaleTimeString());
  timelineChart.data.datasets[0].data = buckets.map((b) => byReasonBucket.light_green[b] || 0);
  timelineChart.data.datasets[1].data = buckets.map((b) => byReasonBucket.lead_accelerating[b] || 0);
  timelineChart.update();

  els.timelineTableBody.innerHTML = data.events_timeline
    .map(
      (r) => `<tr>
        <td>${new Date(r.bucket).toLocaleTimeString()}</td>
        <td><span class="swatch" style="background:${colorFor(r.reason)}"></span>${labelFor(r.reason)}</td>
        <td class="num">${r.count}</td>
      </tr>`
    )
    .join("");
}

function updateReaction(data) {
  const rows = data.reaction_timeline;
  reactionChart.data.labels = rows.map((_, i) => `#${i + 1}`);
  reactionChart.data.datasets[0].data = rows.map((r) => r.delta_ms);
  reactionChart.data.datasets[0].backgroundColor = rows.map((r) => colorFor(r.reason));
  reactionChart.data.datasets[0].timestamps = rows.map((r) => r.alert_timestamp);
  reactionChart.update();

  els.reactionTableBody.innerHTML = rows
    .map(
      (r, i) => `<tr>
        <td>${i + 1}</td>
        <td>${new Date(r.alert_timestamp).toLocaleTimeString()}</td>
        <td><span class="swatch" style="background:${colorFor(r.reason)}"></span>${labelFor(r.reason)}</td>
        <td class="num">${r.delta_ms}</td>
      </tr>`
    )
    .join("");
}

async function poll() {
  let data;
  try {
    const res = await fetch("/api/metrics/summary");
    data = await res.json();
  } catch (err) {
    console.warn("metrics fetch failed", err);
    return;
  }

  els.banner.classList.toggle("show", data.db === "unavailable");
  updateTiles(data);
  updateTimeline(data);
  updateReaction(data);
}

buildTimelineChart();
buildReactionChart();
poll();
setInterval(poll, POLL_MS);
