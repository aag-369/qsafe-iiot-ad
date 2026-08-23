/* Q-Safe Field Link — shared client helpers.
   Kept dependency-free on purpose: the demo has to work on a laptop hotspot
   with no internet, so nothing may be fetched from a CDN at page load. */

export const PALETTE = {
  qber: '#3987e5',
  conf: '#d95926',
  good: '#0ca30c',
  warning: '#fab219',
  critical: '#d03b3b',
  grid: '#2c2c2a',
  axis: '#383835',
  ink2: '#c3c2b7',
  ink3: '#898781',
  surface1: '#1a1a19',
};

export const PROFILE_COLOR = {
  'BIKE-L1': PALETTE.good,
  'HQC-128': PALETTE.warning,
};

/* --- URL prefix --------------------------------------------------------
   The gateway can be served standalone (prefix "") or mounted beside the
   research dashboard (prefix "/link"). Each page is templated with its own
   prefix at request time and sets window.QSAFE_BASE before importing this
   module, so every URL below is built rather than hard-coded. */
export const BASE = (typeof window !== 'undefined' && window.QSAFE_BASE) || '';

/* --- websocket with automatic reconnection ----------------------------- */
export function connect(path, { onMessage, onOpen, onClose } = {}) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}${BASE}${path}`;
  let ws = null;
  let closed = false;
  let retry = 500;

  function open() {
    if (closed) return;
    ws = new WebSocket(url);
    ws.onopen = () => { retry = 500; onOpen && onOpen(); };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      onMessage && onMessage(msg);
    };
    ws.onclose = () => {
      onClose && onClose();
      if (closed) return;
      // Back off gently: a phone that walks out of range should reconnect
      // without hammering the gateway when it comes back.
      setTimeout(open, retry);
      retry = Math.min(retry * 1.6, 5000);
    };
    ws.onerror = () => { try { ws.close(); } catch {} };
  }
  open();

  return {
    send(obj) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(obj));
        return true;
      }
      return false;
    },
    close() { closed = true; if (ws) ws.close(); },
    get ready() { return ws && ws.readyState === WebSocket.OPEN; },
  };
}

export async function api(path, method = 'GET', body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) throw new Error(`${method} ${path} -> ${res.status}`);
  return res.status === 204 ? null : res.json();
}

/* --- formatting -------------------------------------------------------- */
export const fmt = {
  pct: (v, d = 1) => (v == null ? '—' : `${v.toFixed(d)}%`),
  ms: (v, d = 2) => (v == null ? '—' : `${v.toFixed(d)} ms`),
  num: (v, d = 3) => (v == null ? '—' : v.toFixed(d)),
  bytes(v) {
    if (v == null) return '—';
    if (v < 1024) return `${v} B`;
    if (v < 1024 * 1024) return `${(v / 1024).toFixed(1)} KB`;
    return `${(v / 1048576).toFixed(2)} MB`;
  },
  clock(ts) {
    const d = ts ? new Date(ts * 1000) : new Date();
    return d.toLocaleTimeString([], { hour12: false });
  },
};

/* --- canvas plumbing --------------------------------------------------- */
function setupCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height || parseInt(canvas.dataset.h || '120', 10)));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/**
 * Streaming time-series line chart.
 *
 * One measure per chart, always — QBER and detector confidence live on
 * different scales and get their own panels rather than sharing a canvas
 * with two y-axes.
 */
export function drawSeries(canvas, points, opts = {}) {
  const {
    color = PALETTE.qber,
    yMin = 0,
    yMax = 1,
    threshold = null,
    thresholdLabel = '',
    fill = true,
    yTicks = 3,
    yFormat = (v) => v.toFixed(2),
    bands = [],
    markers = [],
  } = opts;

  const { ctx, w, h } = setupCanvas(canvas);
  const padL = 46, padR = 10, padT = 8, padB = 16;
  const plotW = Math.max(1, w - padL - padR);
  const plotH = Math.max(1, h - padT - padB);

  const yPos = (v) => padT + plotH * (1 - (v - yMin) / (yMax - yMin || 1));
  const xPos = (i, n) => padL + (n <= 1 ? plotW : (plotW * i) / (n - 1));

  // Profile bands (state-over-time context behind the line).
  for (const b of bands) {
    ctx.fillStyle = b.color;
    ctx.globalAlpha = 0.10;
    const x0 = xPos(b.from, points.length);
    const x1 = xPos(b.to, points.length);
    ctx.fillRect(x0, padT, Math.max(1, x1 - x0), plotH);
    ctx.globalAlpha = 1;
  }

  // Recessive grid + axis labels.
  ctx.strokeStyle = PALETTE.grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = PALETTE.ink3;
  ctx.font = '10px system-ui, -apple-system, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + ((yMax - yMin) * i) / yTicks;
    const y = Math.round(yPos(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(yFormat(v), padL - 8, y);
  }

  // Threshold: an annotation, not a series.
  if (threshold != null && threshold >= yMin && threshold <= yMax) {
    const y = Math.round(yPos(threshold)) + 0.5;
    ctx.save();
    ctx.strokeStyle = PALETTE.ink3;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.restore();
    if (thresholdLabel) {
      ctx.fillStyle = PALETTE.ink3;
      ctx.textAlign = 'left';
      ctx.fillText(thresholdLabel, padL + 6, y - 8);
    }
  }

  if (!points.length) return { padL, padR, padT, padB, plotW, plotH, xPos, yPos, w, h };

  if (fill) {
    const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
    grad.addColorStop(0, `${color}44`);
    grad.addColorStop(1, `${color}00`);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(xPos(0, points.length), padT + plotH);
    points.forEach((v, i) => ctx.lineTo(xPos(i, points.length), yPos(v)));
    ctx.lineTo(xPos(points.length - 1, points.length), padT + plotH);
    ctx.closePath();
    ctx.fill();
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  points.forEach((v, i) => {
    const x = xPos(i, points.length), y = yPos(v);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Rekey markers: a 2px surface ring keeps them legible over the line.
  for (const m of markers) {
    if (m.index < 0 || m.index >= points.length) continue;
    const x = xPos(m.index, points.length);
    ctx.strokeStyle = m.color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Leading value dot.
  const lastX = xPos(points.length - 1, points.length);
  const lastY = yPos(points[points.length - 1]);
  ctx.fillStyle = PALETTE.surface1;
  ctx.beginPath(); ctx.arc(lastX, lastY, 5.5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(lastX, lastY, 3.5, 0, Math.PI * 2); ctx.fill();

  return { padL, padR, padT, padB, plotW, plotH, xPos, yPos, w, h };
}

/** Profile state over time — a categorical band, not a value plot. */
export function drawProfileStrip(canvas, profiles) {
  const { ctx, w, h } = setupCanvas(canvas);
  const padL = 46, padR = 10;
  const plotW = Math.max(1, w - padL - padR);
  if (!profiles.length) return;
  const step = plotW / profiles.length;
  profiles.forEach((p, i) => {
    ctx.fillStyle = PROFILE_COLOR[p] || PALETTE.ink3;
    ctx.globalAlpha = 0.85;
    // 2px surface gap keeps adjacent segments from merging into one block.
    ctx.fillRect(padL + i * step, 4, Math.max(1, step - 0.5), h - 8);
  });
  ctx.globalAlpha = 1;
}

/** Horizontal comparison bars — used for the adaptive-vs-static cost panel. */
export function drawCompareBars(canvas, rows) {
  const { ctx, w, h } = setupCanvas(canvas);
  const padL = 4, padR = 60;
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  const barH = 18;
  const gap = (h - rows.length * barH) / (rows.length + 1);
  ctx.font = '11px system-ui, -apple-system, sans-serif';
  ctx.textBaseline = 'middle';
  rows.forEach((r, i) => {
    const y = gap + i * (barH + gap);
    const bw = Math.max(2, ((w - padL - padR) * r.value) / max);
    ctx.fillStyle = r.color;
    // 4px rounded data-end, anchored flat against the baseline.
    if (ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(padL, y, bw, barH, [0, 4, 4, 0]);
      ctx.fill();
    } else {
      ctx.fillRect(padL, y, bw, barH);
    }
    ctx.fillStyle = PALETTE.ink2;
    ctx.textAlign = 'left';
    ctx.fillText(r.label, padL + 8, y + barH / 2);
    ctx.fillStyle = PALETTE.ink2;
    ctx.textAlign = 'right';
    ctx.fillText(r.display, w - 6, y + barH / 2);
  });
}

/* --- crosshair tooltip ------------------------------------------------- */
export function attachCrosshair(canvas, tooltipEl, getRows) {
  let geom = null;
  canvas.__setGeom = (g) => { geom = g; };

  function move(ev) {
    if (!geom) return;
    const rect = canvas.getBoundingClientRect();
    const touch = ev.touches && ev.touches[0];
    const x = (touch ? touch.clientX : ev.clientX) - rect.left;
    const rows = getRows(x, geom);
    if (!rows) { tooltipEl.classList.remove('on'); return; }
    tooltipEl.innerHTML = rows;
    tooltipEl.classList.add('on');
    const tw = tooltipEl.offsetWidth;
    let left = x + 14;
    if (left + tw > rect.width) left = x - tw - 14;
    tooltipEl.style.left = `${Math.max(0, left)}px`;
    tooltipEl.style.top = `8px`;
  }
  canvas.addEventListener('mousemove', move);
  canvas.addEventListener('touchmove', move, { passive: true });
  canvas.addEventListener('mouseleave', () => tooltipEl.classList.remove('on'));
  canvas.addEventListener('touchend', () => tooltipEl.classList.remove('on'));
}

/* --- offline banner ---------------------------------------------------- */
export function offlineBanner(text = 'Disconnected from the gateway — retrying…') {
  let el = document.querySelector('.offline');
  return {
    show() {
      if (!el) {
        el = document.createElement('div');
        el.className = 'offline';
        el.textContent = text;
        document.body.appendChild(el);
      }
      el.style.display = 'block';
    },
    hide() { if (el) el.style.display = 'none'; },
  };
}
