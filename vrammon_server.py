#!/usr/bin/env python3
"""vrammon-server — self-contained web dashboard for GPU VRAM / utilization.

Stdlib-only Python + embedded HTML/JS/canvas (no dependencies, no root).
Sampling code is shared with the CLI via vrammon_core.py (sysfs, AMD amdgpu):
  /sys/class/drm/card*/device/mem_info_vram_{used,total}
  /sys/class/drm/card*/device/gpu_busy_percent
  /proc/meminfo, /proc/stat

A background thread samples at a configurable interval (default 1 s) into an
in-memory ring buffer. The history endpoint aggregates the buffer into
min/avg/max buckets over the requested period, so you can view e.g.
30 min of history at coarse buckets or the last minute at 1 s resolution.

Usage:
  vrammon_server.py [--host 127.0.0.1] [--port 8787] [--interval 1.0]
"""

import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from vrammon_core import find_gpus, read_proc_stat, sample_system, gib


# ---------------------------------------------------------------------------
# Sampler + history
# ---------------------------------------------------------------------------

class Sampler:
    """Background thread sampling GPUs/CPU/RAM into a ring buffer."""

    def __init__(self, interval=1.0):
        self.interval = float(interval)
        self.lock = threading.Lock()
        self.current = None
        self.samples = deque(maxlen=20000)  # ~5.5 h at 1 Hz, ~3.4 h at 0.2 Hz
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def set_interval(self, interval):
        self.interval = max(0.05, float(interval))

    def _loop(self):
        prev_stat = read_proc_stat()
        prev_t = time.monotonic()
        while not self._stop.is_set():
            gpus = find_gpus()
            prev2 = read_proc_stat()
            sysinfo = sample_system(
                True, (prev_stat, prev_t) if prev_stat and prev2 else None
            )
            if prev2:
                prev_stat, prev_t = prev2, time.monotonic()
            rec = {
                "t": time.time(),
                "gpus": {
                    n: {"used": v["used"], "total": v["total"], "busy": v["busy"]}
                    for n, v in gpus.items()
                },
            }
            if "cpu_percent" in sysinfo:
                rec["cpu"] = sysinfo["cpu_percent"]
            if "ram_used_kb" in sysinfo:
                rec["ram_gib"] = sysinfo["ram_used_kb"] / 1024 / 1024
                rec["ram_total_gib"] = sysinfo.get("ram_total_kb", 0) / 1024 / 1024
            with self.lock:
                self.current = rec
                self.samples.append(rec)
                cutoff = time.time() - 7200
                while self.samples and self.samples[0]["t"] < cutoff:
                    self.samples.popleft()
            self._stop.wait(self.interval)


STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300]


def choose_step(period):
    """Bucket size so period/step is roughly <= 240 points."""
    for s in STEPS:
        if period / s <= 240:
            return s
    return STEPS[-1]


def _bucket(vals):
    if not vals:
        return None
    return {
        "min": round(min(vals), 3),
        "avg": round(sum(vals) / len(vals), 3),
        "max": round(max(vals), 3),
    }


def history(sampler, period):
    """Aggregate raw samples in [now-period, now] into min/avg/max buckets."""
    with sampler.lock:
        now = time.time()
        t0 = now - period
        recs = [r for r in sampler.samples if r["t"] >= t0]
        current = sampler.current
    if not recs and not current:
        return None
    step = choose_step(period)
    nb = int((now - t0) // step) + 1
    # bucket index relative to t0 so buckets are stable between polls
    bufs = {n: {"used": [], "busy": []} for n in current["gpus"]} if current else {}
    cpu, ram = [], []
    for r in recs:
        bi = int((r["t"] - t0) // step)
        for n, g in r["gpus"].items():
            if n not in bufs:
                bufs[n] = {"used": [], "busy": []}
            while len(bufs[n]["used"]) <= bi:
                bufs[n]["used"].append([])
                bufs[n]["busy"].append([])
            bufs[n]["used"][bi].append(gib(g["used"]))
            bufs[n]["busy"][bi].append(float(g["busy"]))
        if "cpu" in r:
            while len(cpu) <= bi:
                cpu.append([])
            cpu[bi].append(r["cpu"])
        if "ram_gib" in r:
            while len(ram) <= bi:
                ram.append([])
            ram[bi].append(r["ram_gib"])

    gpus = {}
    if current:
        for n, g in current["gpus"].items():
            u = bufs[n]["used"]
            gpus[n] = {
                "total_gib": round(gib(g["total"]), 2),
                "used_min": [m["min"] if m else None for m in map(_bucket, u)],
                "used_avg": [m["avg"] if m else None for m in map(_bucket, u)],
                "used_max": [m["max"] if m else None for m in map(_bucket, u)],
                "busy_avg": [round(sum(v) / len(v), 1) if v else None
                             for v in bufs[n]["busy"]],
                "busy_max": [max(v) if v else None for v in bufs[n]["busy"]],
            }
    return {
        "now": now,
        "period": period,
        "step": step,
        "interval": sampler.interval,
        "current": current,
        "gpus": gpus,
        "cpu": [round(sum(v) / len(v), 1) if v else None for v in cpu],
        "ram": [round(sum(v) / len(v), 2) if v else None for v in ram],
    }


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    sampler = None

    def log_message(self, fmt, *args):
        pass  # quiet; uncomment to debug

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _page(self):
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._page()
        elif u.path == "/api/history":
            q = parse_qs(u.query)
            try:
                period = float((q.get("period") or [1800])[0])
                period = min(max(period, 10), 7200)
            except ValueError:
                period = 1800
            doc = history(self.sampler, period)
            if doc is None:
                self._json({"error": "no samples yet"}, 503)
            else:
                self._json(doc)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/config":
            try:
                n = int(self.headers.get("Content-Length", 0))
                doc = json.loads(self.rfile.read(n) or b"{}")
                interval = doc.get("interval")
                if interval is not None:
                    self.sampler.set_interval(float(interval))
                self._json({"interval": self.sampler.interval})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        else:
            self._json({"error": "not found"}, 404)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vrammon</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #2d333b;
    --text: #e6edf3; --dim: #8b949e; --accent: #4fc3f7;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.4 ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 16px; }
  header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
           margin-bottom: 14px; }
  h1 { font-size: 18px; margin: 0; }
  h1 .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            background: #3fb950; margin-right: 6px; }
  h1 .dot.off { background: #f85149; }
  .sub { color: var(--dim); font-size: 12px; }
  .controls { margin-left: auto; display: flex; gap: 10px; align-items: center; }
  select { background: var(--panel); color: var(--text); border: 1px solid var(--border);
           border-radius: 6px; padding: 4px 8px; font: inherit; font-size: 12px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
           gap: 10px; margin-bottom: 14px; }
  .card { background: var(--panel); border: 1px solid var(--border);
          border-radius: 8px; padding: 10px 12px; }
  .card .name { font-size: 12px; color: var(--dim); margin-bottom: 6px; }
  .big { font-size: 20px; }
  .bar { height: 8px; background: #21262d; border-radius: 4px; overflow: hidden;
         margin: 6px 0; }
  .bar i { display: block; height: 100%; background: var(--accent); }
  .kv { display: flex; justify-content: space-between; color: var(--dim);
        font-size: 12px; }
  .kv b { color: var(--text); font-weight: normal; }
  .chartpanel { background: var(--panel); border: 1px solid var(--border);
                border-radius: 8px; padding: 10px 12px; margin-bottom: 14px; }
  .chartpanel h2 { font-size: 13px; margin: 0 0 8px; color: var(--dim);
                   font-weight: normal; }
  canvas { width: 100%; display: block; }
  .err { color: #f85149; }
  footer { color: var(--dim); font-size: 11px; text-align: center; margin-top: 8px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><span class="dot" id="livedot"></span>vrammon</h1>
    <span class="sub" id="sub">connecting…</span>
    <div class="controls">
      <label>history
        <select id="period">
          <option value="60">1 min</option>
          <option value="300">5 min</option>
          <option value="900">15 min</option>
          <option value="1800" selected>30 min</option>
          <option value="3600">1 hour</option>
        </select>
      </label>
      <label>sample every
        <select id="interval">
          <option value="0.2">0.2 s</option>
          <option value="0.5">0.5 s</option>
          <option value="1" selected>1 s</option>
          <option value="5">5 s</option>
          <option value="15">15 s</option>
          <option value="60">60 s</option>
        </select>
      </label>
    </div>
  </header>
  <div class="cards" id="cards"></div>
  <div class="chartpanel"><h2 id="vramtitle">VRAM usage (GiB)</h2>
    <canvas id="vram" height="220"></canvas></div>
  <div class="chartpanel"><h2 id="busytitle">utilization (%)</h2>
    <canvas id="busy" height="180"></canvas></div>
  <footer id="foot"></footer>
</div>
<script>
"use strict";
const COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ba68c8", "#ffd54f"];
let doc = null;
const periodEl = document.getElementById("period");
const intervalEl = document.getElementById("interval");

async function tick() {
  try {
    const r = await fetch("/api/history?period=" + periodEl.value);
    if (!r.ok) throw new Error(r.status);
    doc = await r.json();
    render();
    const dot = document.getElementById("livedot");
    dot.classList.remove("off");
  } catch (e) {
    document.getElementById("sub").textContent = "disconnected: " + e;
    document.getElementById("livedot").classList.add("off");
  }
}

intervalEl.addEventListener("change", async () => {
  await fetch("/api/config", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ interval: parseFloat(intervalEl.value) }) });
});

function render() {
  const c = doc.current;
  const sub = document.getElementById("sub");
  sub.textContent = "sample every " + doc.interval + "s · " +
    new Date(c.t * 1000).toLocaleTimeString();
  const names = Object.keys(doc.gpus);
  const cards = [];
  names.forEach((n, i) => {
    const g = doc.gpus[n];
    const cur = c.gpus[n] || {};
    const used = gib(cur.used), total = g.total_gib;
    const pct = total ? (100 * used / total) : 0;
    const col = COLORS[i % COLORS.length];
    cards.push(`<div class="card">
      <div class="name">GPU ${esc(n)}</div>
      <div class="big">${used.toFixed(2)} <span style="font-size:12px;color:var(--dim)">/ ${total.toFixed(1)} GiB</span></div>
      <div class="bar"><i style="width:${Math.min(100, pct).toFixed(1)}%;background:${col}"></i></div>
      <div class="kv"><span>busy</span><b>${cur.busy ?? 0}%</b></div>
    </div>`);
    cards.push(`<div class="card">
      <div class="name">GPU ${esc(n)} utilization</div>
      <div class="big">${cur.busy ?? 0}%</div>
      <div class="bar"><i style="width:${Math.min(100, cur.busy ?? 0)}%;background:${col}"></i></div>
    </div>`);
  });
  if (typeof c.cpu !== "undefined") {
    cards.push(`<div class="card">
      <div class="name">system CPU</div>
      <div class="big">${c.cpu}%</div>
      <div class="bar"><i style="width:${Math.min(100, c.cpu)}%"></i></div>
    </div>`);
  }
  if (typeof c.ram_gib !== "undefined") {
    const rt = c.ram_total_gib || 0, rp = rt ? 100 * c.ram_gib / rt : 0;
    cards.push(`<div class="card">
      <div class="name">system RAM</div>
      <div class="big">${c.ram_gib.toFixed(2)} <span style="font-size:12px;color:var(--dim)">/ ${rt.toFixed(1)} GiB</span></div>
      <div class="bar"><i style="width:${Math.min(100, rp).toFixed(1)}%"></i></div>
    </div>`);
  }
  document.getElementById("cards").innerHTML = cards.join("");
  drawVram();
  drawBusy();
  document.getElementById("foot").textContent =
    `history: ${fmtPeriod(doc.period)} at ${fmtStep(doc.step)} buckets · ${new Date(doc.now * 1000).toLocaleTimeString()}`;
}

function esc(s) { return String(s).replace(/[&<>"]/g,
  m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[m])); }
function gib(b) { return b / 1073741824; }
function fmtPeriod(s) {
  if (s < 60) return s + "s";
  if (s < 3600) return Math.round(s / 60) + "m";
  return (s / 3600).toFixed(1).replace(".0", "") + "h";
}
function fmtStep(s) { return s < 60 ? s + "s" : Math.round(s / 60) + "m"; }

function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = parseInt(cv.getAttribute("height"), 10);
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function grid(ctx, w, h, pad, ymax, unit, nlabels = 4) {
  ctx.strokeStyle = "#21262d"; ctx.fillStyle = "#8b949e";
  ctx.font = "10px ui-monospace, monospace"; ctx.lineWidth = 1;
  for (let i = 0; i <= nlabels; i++) {
    const v = ymax * i / nlabels, y = h - pad - (h - 2 * pad) * i / nlabels;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - 4, y); ctx.stroke();
    ctx.fillText(v.toFixed(ymax < 10 ? 1 : 0), 2, y + 3);
  }
  const now = doc.now, t0 = now - doc.period;
  for (let i = 0; i <= 4; i++) {
    const t = t0 + doc.period * i / 4;
    const x = pad + (w - pad - 4) * i / 4;
    const d = new Date(t * 1000);
    const lbl = String(d.getMinutes()).padStart(2, "0") + ":" +
                String(d.getSeconds()).padStart(2, "0");
    ctx.fillText(lbl, x - 12, h - 2);
  }
}

function line(ctx, xs, ys, pad, w, h, ymax, color, width, dash) {
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (y == null) { started = false; continue; }
    const x = pad + (w - pad - 4) * xs[i];
    const yy = h - pad - (h - 2 * pad) * Math.min(1, y / ymax);
    if (!started) { ctx.moveTo(x, yy); started = true; }
    else ctx.lineTo(x, yy);
  }
  ctx.stroke(); ctx.setLineDash([]);
}

function drawVram() {
  const cv = document.getElementById("vram");
  const { ctx, w, h } = setupCanvas(cv);
  const pad = 34;
  ctx.clearRect(0, 0, w, h);
  const names = Object.keys(doc.gpus);
  let ymax = 1;
  names.forEach(n => {
    ymax = Math.max(ymax, (doc.gpus[n].used_max || []).filter(v => v != null).pop() || 0,
                     doc.gpus[n].total_gib * 0.95);
    (doc.gpus[n].used_max || []).forEach(v => { if (v != null) ymax = Math.max(ymax, v); });
  });
  ymax *= 1.05;
  grid(ctx, w, h, pad, ymax, "GiB");
  const t0 = doc.now - doc.period;
  const n = names.length
    ? (doc.gpus[names[0]].used_max || []).length : 0;
  const xs = [];
  for (let i = 0; i < n; i++) xs.push((i * doc.step) / doc.period);
  names.forEach((name, i) => {
    const g = doc.gpus[name];
    const col = COLORS[i % COLORS.length];
    line(ctx, xs, g.used_max, pad, w, h, ymax, col, 1, [3, 3]);
    line(ctx, xs, g.used_avg, pad, w, h, ymax, col, 2);
    line(ctx, xs, xs.map(() => g.total_gib), pad, w, h, ymax,
         "#484f58", 1, [5, 4]);
  });
  document.getElementById("vramtitle").textContent =
    "VRAM usage (GiB) — solid = avg, dashed = max, dotted = total · " +
    names.join(", ");
}

function drawBusy() {
  const cv = document.getElementById("busy");
  const { ctx, w, h } = setupCanvas(cv);
  const pad = 34;
  ctx.clearRect(0, 0, w, h);
  grid(ctx, w, h, pad, 100, "%");
  const names = Object.keys(doc.gpus);
  const t0 = doc.now - doc.period;
  const n = (doc.cpu || []).length;
  const xs = [];
  for (let i = 0; i < n; i++) xs.push((i * doc.step) / doc.period);
  names.forEach((name, i) => {
    const col = COLORS[i % COLORS.length];
    line(ctx, xs, doc.gpus[name].busy_avg, pad, w, h, 100, col, 2);
  });
  line(ctx, xs, doc.cpu, pad, w, h, 100, "#c9d1d9", 1, [6, 3]);
  document.getElementById("busytitle").textContent =
    "utilization (%) — GPU busy avg per card, dashed = system CPU";
}

window.addEventListener("resize", () => { if (doc) { drawVram(); drawBusy(); } });
tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="Self-contained web dashboard for GPU VRAM / utilization.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--interval", type=float, default=1.0,
                    help="initial sampling interval in seconds")
    args = ap.parse_args()

    sampler = Sampler(args.interval)
    Handler.sampler = sampler
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"vrammon-server: http://{args.host}:{args.port} "
          f"(sampling every {sampler.interval}s; change in UI)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
