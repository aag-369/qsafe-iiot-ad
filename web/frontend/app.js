// Q-Safe IIoT-AD — frontend logic. Talks to the FastAPI backend in
// web/backend/app.py, which wraps the real project modules. No fake data
// is hard-coded here beyond the hero copy (which mirrors PROJECT_INFO and
// is overwritten by the API response on load).

const API = ""; // same-origin (FastAPI serves this file too)

// ---------------------------------------------------------------------------
// Background "quantum network" particle animation
// ---------------------------------------------------------------------------
(function particleBackground() {
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const canvas = document.getElementById("bg-canvas");
  if (reduceMotion) {
    // The user has opted out of non-essential motion — hide the canvas
    // entirely rather than drawing a single static frame that would still
    // cost a layout/paint for no benefit.
    canvas.style.display = "none";
    return;
  }
  const ctx = canvas.getContext("2d");
  let w, h, particles;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resize);
  resize();

  const N = Math.min(70, Math.floor((window.innerWidth * window.innerHeight) / 22000));
  particles = Array.from({ length: N }, () => ({
    x: Math.random() * w,
    y: Math.random() * h,
    vx: (Math.random() - 0.5) * 0.25,
    vy: (Math.random() - 0.5) * 0.25,
    r: Math.random() * 1.6 + 0.6,
  }));

  function step() {
    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i], b = particles[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 140) {
          ctx.strokeStyle = `rgba(34, 211, 238, ${0.12 * (1 - d / 140)})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(168, 85, 247, 0.65)";
      ctx.fill();
    }
    requestAnimationFrame(step);
  }
  step();
})();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
async function apiGet(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}
async function apiPost(path, body) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}
function fmt(n, d = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(d);
}
function pct(n, d = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(d) + "%";
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------
async function loadHealth() {
  const dot = document.getElementById("health-dot");
  const label = document.getElementById("health-label");
  const banner = document.getElementById("connectivity-banner");
  try {
    const h = await apiGet("/api/health");
    dot.className = "status-dot " + (h.liboqs_available ? "status-ok" : "status-bad");
    label.textContent = h.liboqs_available
      ? `liboqs online · ${h.kem_backend}`
      : `simulated KEM fallback`;
    banner.hidden = true;
    return true;
  } catch (e) {
    dot.className = "status-dot status-bad";
    label.textContent = "backend unreachable";
    // This is the single most common failure mode: the page was opened as
    // a local file:// path instead of via the running server, so every
    // relative /api/... fetch fails. Say so explicitly instead of leaving
    // a silently broken page.
    banner.hidden = false;
    return false;
  }
}

// ---------------------------------------------------------------------------
// Project info (hero stats, keywords, pipeline)
// ---------------------------------------------------------------------------
// One minimal single-color line icon per pipeline stage — no filled shapes,
// no multi-color, matching the understated "instrument panel" iconography
// called for in the design brief. Indexed by stage order (Physical Layer,
// AI Detection, Crypto-Agility, Orchestration).
const PIPELINE_ICONS = [
  // Physical layer: atom / orbit
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="1.8" fill="currentColor" stroke="none"/><ellipse cx="12" cy="12" rx="9" ry="3.6"/><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(120 12 12)"/></svg>`,
  // AI detection: node graph
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><circle cx="19" cy="12" r="2.4"/><path d="M7 6.8 17 11 M7 17.2 17 13"/></svg>`,
  // Crypto-agility: key / shield
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2 4 5v6c0 5 3.4 8.4 8 9 4.6-.6 8-4 8-9V5l-8-3Z"/><path d="M9.5 12a2.5 2.5 0 1 1 2.4 1.7L15 17" stroke-linecap="round"/></svg>`,
  // Orchestration: flow / pipeline
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M3 6h6M3 12h6M3 18h6"/><path d="M9 6h4a3 3 0 0 1 3 3v0M9 18h4a3 3 0 0 0 3-3v0"/><path d="M16 9h5M16 15h5"/></svg>`,
];

async function loadProjectInfo() {
  try {
    const info = await apiGet("/api/project-info");
    document.getElementById("abstract-text").textContent = info.abstract;

    const chips = document.getElementById("keyword-chips");
    chips.innerHTML = "";
    info.keywords.forEach((k) => {
      const s = document.createElement("span");
      s.className = "chip";
      s.textContent = k;
      chips.appendChild(s);
    });

    const pipeline = document.getElementById("pipeline");
    pipeline.innerHTML = "";
    info.pipeline.forEach((stage, i) => {
      const card = document.createElement("div");
      card.className = "pipeline-card";
      card.innerHTML = `
        <div class="pipeline-icon">${PIPELINE_ICONS[i] || ""}</div>
        <div class="pipeline-index">STAGE ${String(i + 1).padStart(2, "0")}</div>
        <h4>${stage.stage}</h4>
        <span class="pipeline-module">${stage.module}</span>
        <p class="pipeline-desc">${stage.description}</p>
      `;
      pipeline.appendChild(card);
    });
  } catch (e) {
    console.error("project-info failed", e);
  }
}

// ---------------------------------------------------------------------------
// Results summary (hero stats + benchmark cards + charts)
// ---------------------------------------------------------------------------
let latencyChart, sizeChart, qberChart;

async function loadResultsSummary() {
  try {
    const summary = await apiGet("/api/results/summary");
    const tm = summary.train_metrics;
    const bm = summary.benchmark;
    const qz = summary.quantization;

    document.querySelector('[data-stat="f1"]').textContent = fmt(tm.f1, 3);
    document.querySelector('[data-stat="cpu"]').textContent = pct(bm.cpu_latency_reduction_pct, 1);

    const cards = document.getElementById("metric-cards");
    cards.innerHTML = "";
    const metrics = [
      ["Detector F1", fmt(tm.f1, 3)],
      ["Precision", fmt(tm.precision, 3)],
      ["Recall", fmt(tm.recall, 3)],
      ["ROC-AUC", fmt(tm.roc_auc, 3)],
      ["Operational F1", fmt(bm.operational_f1, 3)],
      ["CPU/Latency Saved", pct(bm.cpu_latency_reduction_pct, 1)],
    ];
    metrics.forEach(([label, value]) => {
      const el = document.createElement("div");
      el.className = "metric-card";
      el.innerHTML = `<div class="metric-value">${value}</div><div class="metric-label">${label}</div>`;
      cards.appendChild(el);
    });

    renderLatencyChart(bm.adaptive_total_kem_latency_ms, bm.static_hqc128_total_kem_latency_ms);
    renderSizeChart(qz.tflite_fp32_bytes, qz.tflite_int8_bytes);
  } catch (e) {
    console.error("results-summary failed", e);
  }
}

function chartTheme() {
  return {
    color: "#93a1c2",
    grid: "rgba(255,255,255,0.06)",
  };
}

function renderLatencyChart(adaptiveMs, staticMs) {
  const t = chartTheme();
  const ctx = document.getElementById("latency-chart");
  if (latencyChart) latencyChart.destroy();
  latencyChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["AI-gated adaptive", "Static always-on HQC-128"],
      datasets: [{
        data: [adaptiveMs, staticMs],
        backgroundColor: ["#22d3ee", "#e879f9"],
        borderRadius: 6,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: t.color }, grid: { display: false } },
        y: { ticks: { color: t.color }, grid: { color: t.grid }, title: { display: true, text: "ms total KEM latency", color: t.color } },
      },
    },
  });
}

function renderSizeChart(fp32, int8) {
  const t = chartTheme();
  const ctx = document.getElementById("size-chart");
  if (sizeChart) sizeChart.destroy();
  sizeChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Float32 TFLite", "INT8 TFLite"],
      datasets: [{
        data: [fp32 / 1024, int8 / 1024],
        backgroundColor: ["#a855f7", "#22d3ee"],
        borderRadius: 6,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: t.color }, grid: { display: false } },
        y: { ticks: { color: t.color }, grid: { color: t.grid }, title: { display: true, text: "KB", color: t.color } },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Live simulation
// ---------------------------------------------------------------------------
function setupSliders() {
  const bind = (id, valId, fmtFn) => {
    const input = document.getElementById(id);
    const out = document.getElementById(valId);
    const update = () => (out.textContent = fmtFn ? fmtFn(input.value) : input.value);
    input.addEventListener("input", update);
    update();
  };
  bind("rounds", "rounds-val");
  bind("qubits", "qubits-val");
  bind("intensity", "intensity-val", (v) => Number(v).toFixed(2));
  bind("probe-eve", "probe-eve-val", (v) => Number(v).toFixed(2));
  bind("bench-rounds", "bench-rounds-val");
  bind("fleet-devices", "fleet-devices-val");
  bind("fleet-rounds", "fleet-rounds-val");
  bind("fleet-min-devices", "fleet-min-devices-val");
}

function showInlineError(id, message) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = message;
  el.hidden = false;
}
function hideInlineError(id) {
  const el = document.getElementById(id);
  if (el) el.hidden = true;
}

async function runLiveSimulation() {
  const btn = document.getElementById("run-live-btn");
  const spinner = btn.querySelector(".btn-spinner");
  const label = btn.querySelector(".btn-label");
  btn.disabled = true;
  spinner.hidden = false;
  label.textContent = "Simulating…";
  hideInlineError("demo-error");

  try {
    const body = {
      n_rounds: Number(document.getElementById("rounds").value),
      n_qubits_per_round: Number(document.getElementById("qubits").value),
      inject_attack: document.getElementById("inject-attack").checked,
      attack_intensity: Number(document.getElementById("intensity").value),
    };
    const t0 = performance.now();
    const result = await apiPost("/api/simulate/live", body);
    const elapsedS = ((performance.now() - t0) / 1000).toFixed(1);
    renderLiveResult(result, body, elapsedS);
  } catch (e) {
    console.error(e);
    showInlineError(
      "demo-error",
      "Live simulation failed — the backend may be unreachable or still starting up. " + e.message
    );
    await loadHealth();
  } finally {
    btn.disabled = false;
    spinner.hidden = true;
    label.textContent = "Run Live Simulation";
  }
}

function renderLiveResult(result, requestBody, elapsedS) {
  document.getElementById("demo-empty-state").style.display = "none";
  const points = result.points;

  const labels = points.map((p) => p.t);
  const qberData = points.map((p) => p.qber);
  const confData = points.map((p) => p.confidence);
  const attackShade = points.map((p) => (p.label ? p.qber : null));
  const thresholdLine = points.map(() => result.threshold);

  const t = chartTheme();
  const ctx = document.getElementById("qber-chart");
  if (qberChart) qberChart.destroy();
  qberChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "QBER",
          data: qberData,
          borderColor: "#22d3ee",
          backgroundColor: "rgba(34,211,238,0.08)",
          pointRadius: 0,
          borderWidth: 2,
          tension: 0.25,
          fill: true,
          yAxisID: "y",
        },
        {
          label: "Ground-truth attack (QBER)",
          data: attackShade,
          borderColor: "transparent",
          backgroundColor: "rgba(239,68,68,0.35)",
          pointRadius: 0,
          borderWidth: 0,
          fill: true,
          yAxisID: "y",
        },
        {
          label: "Detector confidence",
          data: confData,
          borderColor: "#a855f7",
          borderDash: [4, 3],
          pointRadius: 0,
          borderWidth: 2,
          tension: 0.25,
          yAxisID: "y1",
        },
        {
          label: "Escalation threshold",
          data: thresholdLine,
          borderColor: "rgba(245,158,11,0.55)",
          borderDash: [2, 4],
          pointRadius: 0,
          borderWidth: 1,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: t.color, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: t.color, maxTicksLimit: 10 }, grid: { display: false }, title: { display: true, text: "round", color: t.color } },
        y: { position: "left", min: 0, ticks: { color: t.color }, grid: { color: t.grid }, title: { display: true, text: "QBER", color: t.color } },
        y1: { position: "right", min: 0, max: 1, ticks: { color: t.color }, grid: { display: false }, title: { display: true, text: "confidence", color: t.color } },
      },
    },
  });

  // Profile timeline strip
  const timeline = document.getElementById("profile-timeline");
  timeline.innerHTML = "";
  points.forEach((p) => {
    const tick = document.createElement("div");
    tick.className = "tick";
    tick.style.background = p.profile === "HQC-128" ? "#e879f9" : "#22d3ee";
    tick.title = `t=${p.t} · ${p.profile} · qber=${p.qber.toFixed(4)} · conf=${p.confidence.toFixed(2)}`;
    timeline.appendChild(tick);
  });

  // Status cards
  const escalations = points.filter((p) => p.escalated).length;
  const meanQber = qberData.reduce((a, b) => a + b, 0) / qberData.length;
  const meanConf = confData.reduce((a, b) => a + b, 0) / confData.length;
  const current = points[points.length - 1];
  const isEscalated = current.profile === "HQC-128";

  const profileEl = document.getElementById("stat-profile");
  profileEl.textContent = current.profile;
  profileEl.classList.toggle("is-escalated", isEscalated);
  const profileCard = profileEl.closest(".status-card");
  profileCard.classList.toggle("is-escalated", isEscalated);
  profileCard.classList.toggle("pulse-active", isEscalated);
  document.getElementById("stat-escalations").textContent = escalations;
  document.getElementById("stat-qber").textContent = fmt(meanQber, 4);
  document.getElementById("stat-conf").textContent = fmt(meanConf, 3);

  // Session metadata — the "instrument panel" readout of exactly what was
  // simulated and when, reinforcing that this run is real and specific.
  const meta = document.getElementById("session-meta");
  meta.classList.add("live");
  const now = new Date();
  meta.innerHTML =
    `<span class="session-meta-dot"></span> seed=${result.seed} · ` +
    `${result.n_rounds} rounds · ${result.n_qubits_per_round} qubits/round · ` +
    `attack=${requestBody.inject_attack ? "on" : "off"} · ` +
    `computed in ${elapsedS}s · run at ${now.toLocaleTimeString()}`;
}

async function runProbe() {
  const btn = document.getElementById("probe-btn");
  const out = document.getElementById("probe-result");
  btn.disabled = true;
  out.textContent = "Running one real BB84 circuit on Qiskit Aer…";
  try {
    const eve = Number(document.getElementById("probe-eve").value);
    const result = await apiPost("/api/simulate/probe", {
      n_qubits: 64,
      channel_error_prob: 0.02,
      eve_intercept_prob: eve,
    });
    out.textContent =
      `QBER = ${result.qber.toFixed(4)}  |  sifted key = ${result.sifted_key_length} bits  |  ` +
      `intercepted = ${result.n_intercepted}/${result.n_qubits}  |  ${result.simulation_time_ms.toFixed(1)} ms`;
  } catch (e) {
    out.textContent = "Probe failed: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Live benchmark
// ---------------------------------------------------------------------------
async function runLiveBenchmark() {
  const btn = document.getElementById("run-bench-btn");
  const spinner = btn.querySelector(".btn-spinner");
  const label = btn.querySelector(".btn-label");
  const out = document.getElementById("bench-result");
  btn.disabled = true;
  spinner.hidden = false;
  label.textContent = "Running real liboqs KEM ops…";
  out.innerHTML = "";

  try {
    const n_rounds = Number(document.getElementById("bench-rounds").value);
    const r = await apiPost("/api/benchmark/run", { n_rounds, n_qubits_per_round: 48 });
    out.innerHTML = `
      <table>
        <tr><td>KEM backend</td><td>${r.kem_backend}${r.liboqs_available ? " (real liboqs)" : " (simulated)"}</td></tr>
        <tr><td>Rounds simulated</td><td>${r.n_rounds} (${r.n_attack_rounds} attack rounds)</td></tr>
        <tr><td>Operational F1 / Precision / Recall</td><td>${fmt(r.operational_f1)} / ${fmt(r.operational_precision)} / ${fmt(r.operational_recall)}</td></tr>
        <tr><td>Adaptive total KEM latency</td><td>${fmt(r.adaptive_total_kem_latency_ms, 1)} ms</td></tr>
        <tr><td>Static HQC-128 total KEM latency</td><td>${fmt(r.static_hqc128_total_kem_latency_ms, 1)} ms</td></tr>
        <tr><td>CPU / latency reduction</td><td>${pct(r.cpu_latency_reduction_pct, 1)}</td></tr>
        <tr><td>Rounds on BIKE-L1 / HQC-128</td><td>${r.rounds_on_bike_l1} / ${r.rounds_on_hqc128}</td></tr>
      </table>
    `;
  } catch (e) {
    out.textContent = "Benchmark failed: " + e.message;
  } finally {
    btn.disabled = false;
    spinner.hidden = true;
    label.textContent = "Run Live Benchmark (real liboqs)";
  }
}

// ---------------------------------------------------------------------------
// Fleet View
// ---------------------------------------------------------------------------
const TYPE_COLORS = {
  benign: "#6b7280",
  eavesdrop: "#a855f7",
  jamming: "#f59e0b",
  pns: "#22d3ee",
};

function setupFleetScenarioToggle() {
  const scenarioSelect = document.getElementById("fleet-scenario");
  const campaignField = document.getElementById("fleet-campaign-type-field");
  const sync = () => {
    campaignField.style.display = scenarioSelect.value === "coordinated_campaign" ? "" : "none";
  };
  scenarioSelect.addEventListener("change", sync);
  sync();
}

async function runFleetSimulation() {
  const btn = document.getElementById("run-fleet-btn");
  const spinner = btn.querySelector(".btn-spinner");
  const label = btn.querySelector(".btn-label");
  btn.disabled = true;
  spinner.hidden = false;
  label.textContent = "Simulating fleet…";
  hideInlineError("fleet-error");

  try {
    const body = {
      n_devices: Number(document.getElementById("fleet-devices").value),
      n_rounds: Number(document.getElementById("fleet-rounds").value),
      n_qubits_per_round: 32,
      scenario: document.getElementById("fleet-scenario").value,
      campaign_attack_type: document.getElementById("fleet-campaign-type").value,
      campaign_fraction: 0.5,
      min_devices_for_alert: Number(document.getElementById("fleet-min-devices").value),
    };
    const result = await apiPost("/api/simulate/fleet", body);
    renderFleetResult(result);
  } catch (e) {
    console.error(e);
    showInlineError(
      "fleet-error",
      "Fleet simulation failed — the backend may be unreachable or still starting up. " + e.message
    );
    await loadHealth();
  } finally {
    btn.disabled = false;
    spinner.hidden = true;
    label.textContent = "Run Fleet Simulation";
  }
}

function renderFleetResult(result) {
  document.getElementById("fleet-empty-state").style.display = "none";

  const banner = document.getElementById("fleet-alert-banner");
  const alerts = result.fleet_alerts || [];
  banner.hidden = false;
  if (alerts.length === 0) {
    banner.className = "fleet-alert-banner is-nominal";
    banner.innerHTML = `<strong>Fleet nominal.</strong> No coordinated campaign detected across ${result.n_devices} devices
      (requires ${result.min_devices_for_alert}+ devices escalating together on the same attack type).`;
  } else {
    banner.className = "fleet-alert-banner";
    banner.innerHTML = alerts
      .map(
        (a) => `<div class="fleet-alert-item">
          <strong>Coordinated ${a.dominant_attack_type} campaign detected</strong>
          — rounds ${a.t_start}–${a.t_end}, ${a.peak_device_count} device(s) involved
          (type agreement ${(a.type_agreement * 100).toFixed(0)}%): ${a.device_ids.join(", ")}
        </div>`
      )
      .join("");
  }

  const flaggedDevices = new Set(alerts.flatMap((a) => a.device_ids));

  const grid = document.getElementById("fleet-device-grid");
  grid.innerHTML = "";
  result.devices.forEach((dev) => {
    const isEscalated = dev.final_profile === "HQC-128";
    const isFlagged = flaggedDevices.has(dev.device_id);

    const card = document.createElement("div");
    card.className = "fleet-device-card";
    card.classList.toggle("is-escalated", isEscalated);
    card.classList.toggle("is-flagged", isFlagged);

    const targetBadge = dev.is_campaign_target
      ? `<span class="fleet-device-badge is-target">campaign target</span>`
      : "";

    card.innerHTML = `
      <div class="fleet-device-head">
        <span class="fleet-device-id">${dev.device_id}</span>
        ${targetBadge}
      </div>
      <div class="fleet-device-stats">
        <span>profile <b>${dev.final_profile}</b></span>
        <span>escalations <b>${dev.n_escalations}</b></span>
        <span>mean QBER <b>${dev.mean_qber.toFixed(4)}</b></span>
      </div>
      <div class="fleet-device-timeline"></div>
    `;

    const timeline = card.querySelector(".fleet-device-timeline");
    dev.points.forEach((p) => {
      const tick = document.createElement("div");
      tick.className = "tick";
      tick.style.background = p.profile === "HQC-128" ? "#e879f9" : TYPE_COLORS[p.predicted_type] || "#22d3ee";
      tick.title = `t=${p.t} · ${p.profile} · predicted: ${p.predicted_type} · qber=${p.qber.toFixed(4)}`;
      timeline.appendChild(tick);
    });

    grid.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function setupMobileNav() {
  const toggle = document.getElementById("nav-toggle");
  const links = document.getElementById("nav-links");
  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  links.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => {
      links.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    })
  );
}

document.addEventListener("DOMContentLoaded", () => {
  setupSliders();
  setupMobileNav();
  loadHealth();
  loadProjectInfo();
  loadResultsSummary();

  document.getElementById("run-live-btn").addEventListener("click", runLiveSimulation);
  document.getElementById("probe-btn").addEventListener("click", runProbe);
  document.getElementById("run-bench-btn").addEventListener("click", runLiveBenchmark);
  setupFleetScenarioToggle();
  document.getElementById("run-fleet-btn").addEventListener("click", runFleetSimulation);
  document.getElementById("retry-connection-btn").addEventListener("click", () => {
    loadHealth();
    loadProjectInfo();
    loadResultsSummary();
  });

  const ghLink = document.getElementById("github-link");
  ghLink.href = "https://github.com/aag-369/qsafe-iiot-ad";
});
