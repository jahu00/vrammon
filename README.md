# vrammon

On-demand GPU VRAM / utilization monitor for AI workloads. Stdlib-only Python,
no root. Two entry points sharing the same sampling code (`vrammon_core.py`):

- `vrammon.py` — CLI. Built for the agent workflow: spin it up around a job
  and get the **peak** VRAM, not just before/after readings.
- `vrammon_server.py` — self-contained web dashboard (embedded HTML/JS/canvas,
  no dependencies) with live charts and an in-memory history buffer.

Hardware: AMD RX 7900 XTX (amdgpu). Data comes straight from sysfs, so it sees
driver-level truth (Vulkan allocations included) at ~0 overhead:

- `/sys/class/drm/card*/device/mem_info_vram_used` / `mem_info_vram_total`
- `/sys/class/drm/card*/device/gpu_busy_percent`
- `/proc/meminfo`, `/proc/stat` (system RAM / CPU, with `--cpu-ram`)

`card1` and `renderD128` are deduped by PCI device, so one physical GPU shows
once.

## Modes

```bash
# One-shot current reading (before/after check)
vrammon.py snapshot                 # human
vrammon.py snapshot --json --cpu-ram

# Continuous monitor (default). Ctrl+C or --duration ends it;
# prints min/avg/peak summary.
vrammon.py watch --interval 0.1 --duration 60
vrammon.py watch --log run.jsonl --json      # JSONL sample log + JSON summary

# Measure a workload end-to-end: spawns the command, monitors while it runs,
# exits with the child's exit code.
vrammon.py run --interval 0.1 --cpu-ram -- <cmd and its args>
```

Flags (watch/run): `--interval` seconds (default 0.1 — 10 Hz is plenty to
catch VRAM peaks), `--log FILE` (one JSON line per sample, appended),
`--cpu-ram`, `--json` (summary as JSON on stdout), `--quiet`, `--duration`.

## Summary output (JSON)

```json
{
  "gpus": {"card1": {
      "vram_total_gib": 23.98,
      "vram_used_min_gib": 0.807,
      "vram_used_avg_gib": 15.335,
      "vram_used_peak_gib": 21.484,
      "vram_peak_pct": 89.6,
      "gpu_busy_avg_pct": 22.1,
      "gpu_busy_peak_pct": 100}},
  "cpu": {"avg_pct": 12.3, "peak_pct": 95.0},
  "ram": {"used_min_gib": 0.3, "used_avg_gib": 0.4, "used_peak_gib": 0.5},
  "samples": 2321, "duration_s": 120.0
}
```

## Validated against Krea2 (2026-10-29)

Krea2 Q8_0 t2i on this build of sd-server (be48649), 1024² image:
peak VRAM **21.48 GiB (89.6% of 24 GB)**, busy peak 100%, ~34 s for 8 steps.
Note: the krea2 skill documents ~17.3 GB peak from the previous build —
this build holds ~4 GB more (see the krea2 skill note).

## Caveats

- `gpu_busy_percent` is the amdgpu scheduler's busy fraction; fine for
  "is it crunching" but not a throughput number.
- `--log` appends; remove/rotate old files if you reuse names.
- CPU% is whole-machine across all cores (0 = idle, 100 = all cores busy).

## Web dashboard

```bash
./vrammon_server.py                       # http://127.0.0.1:8787
./vrammon_server.py --host 0.0.0.0 --port 9000 --interval 0.5
```

- A background thread samples sysfs at a **configurable interval**
  (default **1 s**; changeable live from the UI via `POST /api/config`,
  or `--interval` at startup). Samples stay in an in-memory ring buffer
  (last ~1–5 h depending on interval).
- `GET /api/history?period=<seconds>` returns the last `period` of data
  **aggregated into min/avg/max buckets**. Bucket size is chosen so the
  window has ≤ 240 points, e.g.:

  | period | bucket | points |
  |--------|--------|--------|
  | 1 min  | 1 s    | 60     |
  | 15 min | 5 s    | 180    |
  | 30 min | 10 s   | 180    |
  | 1 h    | 15 s   | 240    |

  Per GPU: `used_min/used_avg/used_max` (GiB), `busy_avg`, `busy_max`,
  `total_gib`. Plus per-bucket system `cpu` avg % and `ram` avg GiB, and
  `current` (latest raw sample).
- The page shows current-stat cards (VRAM bar, busy, CPU, RAM) and two
  charts: VRAM usage (GiB; solid = avg, dashed = max, dotted = total)
  and utilization (%; per-GPU busy + system CPU). It polls every second.

API:

```
GET  /                        # dashboard
GET  /api/history?period=1800 # aggregated history (10..7200 s)
POST /api/config              # {"interval": 1.0}
```

### systemd (user) service

`vrammon.service` ships in this folder:

```bash
mkdir -p ~/.config/systemd/user
cp vrammon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now vrammon
```

(Adjust the paths in the unit if the toolbox lives elsewhere.)
