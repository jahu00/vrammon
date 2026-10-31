#!/usr/bin/env python3
"""vrammon — on-demand GPU VRAM / utilization monitor for AI workloads.

Stdlib-only. Designed to be spun up by an agent around a workload to capture
PEAK VRAM (not just before/after).

Data sources (AMD amdgpu, no root needed) — see vrammon_core.py:
  /sys/class/drm/card*/device/mem_info_vram_{used,total}
  /sys/class/drm/card*/device/gpu_busy_percent
  /proc/meminfo, /proc/stat   (system RAM / CPU, optional)

Modes:
  snapshot   one-shot current reading (human or --json)
  watch      (default) continuous monitor; Ctrl+C or --duration ends it and
             prints a summary (min/avg/peak per GPU + optional CPU/RAM)
  run        spawn a command, monitor while it runs, print summary with the
             child's exit code — the "measure one workload" mode

Examples:
  vrammon.py snapshot --json
  vrammon.py watch --interval 0.1 --duration 30
  vrammon.py watch --log run.jsonl --json
  vrammon.py run -- --interval 0.1 python3 gen_image.py
  vrammon.py run --log krea2.jsonl --json -- <some command>
"""

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from vrammon_core import (
    find_gpus,
    read_proc_stat,
    sample_system,
    gib,
    fmt_gib,
    now_iso,
)


def bar_str(pct, width=20):
    filled = int(width * pct / 100)
    return "[" + "#" * filled + "." * (width - filled) + "]"


class Stats:
    def __init__(self):
        self.gpu = {}  # name -> stats
        self.cpu = {"min": None, "sum": 0.0, "peak": None, "n": 0}
        self.ram = {"min": None, "sum": 0, "peak": None, "n": 0}
        self.samples = 0
        self.t0 = time.monotonic()
        self.t0_wall = time.time()
        self.t_end = None

    def add(self, gpus, sysinfo):
        self.samples += 1
        for name, g in gpus.items():
            s = self.gpu.setdefault(
                name,
                {"min": g["used"], "sum": 0, "peak": g["used"],
                 "busy_sum": 0, "busy_peak": 0, "n": 0, "total": g["total"]},
            )
            s["n"] += 1
            s["min"] = min(s["min"], g["used"])
            s["sum"] += g["used"]
            s["peak"] = max(s["peak"], g["used"])
            s["busy_sum"] += g["busy"]
            s["busy_peak"] = max(s["busy_peak"], g["busy"])
        c = sysinfo.get("cpu_percent")
        if c is not None:
            self.cpu["n"] += 1
            self.cpu["min"] = c if self.cpu["min"] is None else min(self.cpu["min"], c)
            self.cpu["sum"] += c
            self.cpu["peak"] = c if self.cpu["peak"] is None else max(self.cpu["peak"], c)
        r = sysinfo.get("ram_used_kb")
        if r is not None:
            self.ram["n"] += 1
            self.ram["min"] = r if self.ram["min"] is None else min(self.ram["min"], r)
            self.ram["sum"] += r
            self.ram["peak"] = r if self.ram["peak"] is None else max(self.ram["peak"], r)

    def summary(self, command=None, exit_code=None):
        self.t_end = time.monotonic()
        dur = (self.t_end - self.t0) if self.samples else 0.0
        out = {
            "tool": "vrammon",
            "started": datetime.fromtimestamp(self.t0_wall, timezone.utc).isoformat(timespec="milliseconds"),
            "duration_s": round(dur, 2),
            "samples": self.samples,
            "gpus": {},
        }
        for name, s in self.gpu.items():
            out["gpus"][name] = {
                "vram_total_gib": round(gib(s["total"]), 2),
                "vram_used_min_gib": round(gib(s["min"]), 3),
                "vram_used_avg_gib": round(gib(s["sum"] / s["n"]), 3) if s["n"] else None,
                "vram_used_peak_gib": round(gib(s["peak"]), 3),
                "vram_peak_pct": round(100.0 * s["peak"] / s["total"], 1) if s["total"] else None,
                "gpu_busy_avg_pct": round(s["busy_sum"] / s["n"], 1) if s["n"] else None,
                "gpu_busy_peak_pct": s["busy_peak"],
            }
        if self.cpu["n"]:
            out["cpu"] = {
                "avg_pct": round(self.cpu["sum"] / self.cpu["n"], 1),
                "peak_pct": round(self.cpu["peak"], 1),
            }
        if self.ram["n"]:
            out["ram"] = {
                "used_min_gib": round(self.ram["min"] / 1024 / 1024, 2),
                "used_avg_gib": round(self.ram["sum"] / self.ram["n"] / 1024 / 1024, 2),
                "used_peak_gib": round(self.ram["peak"] / 1024 / 1024, 2),
            }
        if command:
            out["command"] = " ".join(command)
            out["exit_code"] = exit_code
        return out


def print_summary(s):
    print()
    print(f"=== vrammon summary — {s['samples']} samples over {s['duration_s']:.1f}s ===")
    if s.get("command"):
        print(f"command: {s['command']}  (exit {s.get('exit_code')})")
    for name, g in s["gpus"].items():
        print(f"GPU {name}:")
        print(f"  VRAM  min {g['vram_used_min_gib']:7.2f}  avg {g['vram_used_avg_gib']:7.2f}  "
              f"peak {g['vram_used_peak_gib']:7.2f} GiB  ({g['vram_peak_pct']:.1f}% of {g['vram_total_gib']} GiB)")
        print(f"  busy  avg {g['gpu_busy_avg_pct']:5.1f}%  peak {g['gpu_busy_peak_pct']:3d}%")
    if "cpu" in s:
        print(f"CPU   avg {s['cpu']['avg_pct']:5.1f}%  peak {s['cpu']['peak_pct']:5.1f}%")
    if "ram" in s:
        print(f"RAM   min {s['ram']['used_min_gib']:7.2f}  avg {s['ram']['used_avg_gib']:7.2f}  "
              f"peak {s['ram']['used_peak_gib']:7.2f} GiB")


def do_snapshot(args):
    gpus = find_gpus()
    sysinfo = sample_system(args.cpu_ram, None)
    if args.json:
        g = {n: {"vram_used_gib": round(gib(v["used"]), 3),
                 "vram_total_gib": round(gib(v["total"]), 2),
                 "gpu_busy_pct": v["busy"]} for n, v in gpus.items()}
        doc = {"time": now_iso(), "gpus": g}
        if "cpu_percent" in sysinfo:
            doc["cpu_percent"] = sysinfo["cpu_percent"]
        if "ram_used_kb" in sysinfo:
            doc["ram_used_gib"] = round(sysinfo["ram_used_kb"] / 1024 / 1024, 2)
        print(json.dumps(doc, indent=2))
        return 0
    if not gpus:
        print("vrammon: no GPU with mem_info_vram_* found under /sys/class/drm")
        return 1
    for name, v in gpus.items():
        print(f"{name}: {fmt_gib(v['used'])} / {fmt_gib(v['total'])} VRAM  "
              f"GPU {v['busy']:3d}% {bar_str(v['busy'])}")
    if "ram_used_kb" in sysinfo:
        print(f"system RAM: {sysinfo['ram_used_kb'] / 1024 / 1024:.2f} GiB used / "
              f"{sysinfo['ram_total_kb'] / 1024 / 1024:.2f} GiB")
    return 0


def _monitor(args, child=None, cmd=None, stop_event=None, on_stop=None):
    """Core sampling loop. child: Popen to follow (watch mode) or None."""
    gpus = find_gpus()
    if not gpus:
        print("vrammon: no GPU with mem_info_vram_* found under /sys/class/drm", file=sys.stderr)
        return 1
    stats = Stats()
    logf = open(args.log, "a") if args.log else None
    stop = {"flag": False}

    def on_sig(signum, frame):
        stop["flag"] = True
        if on_stop:
            on_stop()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    prev_stat = read_proc_stat()
    prev_t = time.monotonic()
    last_draw = 0.0
    t_end = time.monotonic() + args.duration if args.duration else None
    child_rc = None
    try:
        while not stop["flag"] and not (child is not None and child.poll() is not None):
            gpus = find_gpus()
            prev_stat2 = read_proc_stat()
            sysinfo = sample_system(args.cpu_ram,
                                    (prev_stat, prev_t) if prev_stat and prev_stat2 else None)
            if prev_stat2:
                prev_stat, prev_t = prev_stat2, time.monotonic()
            stats.add(gpus, sysinfo)
            if logf:
                rec = {"time": now_iso(),
                       "gpus": {n: {"vram_used": v["used"], "gpu_busy_pct": v["busy"]}
                                for n, v in gpus.items()}}
                if sysinfo:
                    rec["system"] = sysinfo
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

            if not args.json and sys.stderr.isatty() and not args.quiet:
                if time.monotonic() - last_draw >= 0.25:
                    last_draw = time.monotonic()
                    line = "\r" + " | ".join(
                        f"{n}: {gib(v['used']):.2f}/{gib(v['total']):.0f} GiB {bar_str(v['busy'], 10)} {v['busy']:3d}%"
                        for n, v in gpus.items())
                    if "cpu_percent" in sysinfo:
                        line += f"  CPU {sysinfo['cpu_percent']:.1f}%"
                    if "ram_used_kb" in sysinfo:
                        line += f"  RAM {sysinfo['ram_used_kb'] / 1024 / 1024:.2f} GiB"
                    sys.stderr.write(line + "\n")
                    sys.stderr.flush()

            if child is not None:
                if stop_event is not None and stop_event.is_set():
                    child.terminate()
                    break
                time.sleep(0.02)
            else:
                if t_end is not None:
                    remain = t_end - time.monotonic()
                    if remain <= 0:
                        break
                    time.sleep(min(args.interval, remain))
                else:
                    time.sleep(args.interval)
    finally:
        if logf:
            logf.close()
    if child is not None:
        child_rc = child.wait()

    if stats.samples == 0:
        stats.add(find_gpus(), sample_system(args.cpu_ram, None))

    summary = stats.summary(command=cmd, exit_code=child_rc)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary)
    if child_rc is not None and child_rc != 0 and not args.quiet:
        print(f"vrammon: command exited with code {child_rc}", file=sys.stderr)
    return child_rc if child is not None else 0


def do_watch(args):
    return _monitor(args)


def do_run(args, cmd):
    stop_event = threading.Event()
    child = subprocess.Popen(cmd)
    if args.duration:
        def kill_later():
            if stop_event.wait(args.duration):
                return
            child.kill()
            stop_event.set()
        threading.Thread(target=kill_later, daemon=True).start()
    return _monitor(args, child=child, cmd=cmd, stop_event=stop_event,
                    on_stop=lambda: child.terminate())


def main():
    p = argparse.ArgumentParser(
        prog="vrammon",
        description="Monitor GPU VRAM (peak!), GPU busy%, optionally CPU/RAM. "
                    "Stdlib-only, AMD amdgpu sysfs.",
    )
    sub = p.add_subparsers(dest="mode")

    def add_common(sp):
        sp.add_argument("--interval", type=float, default=0.1,
                        help="sampling interval in seconds (default 0.1)")
        sp.add_argument("--log", metavar="FILE",
                        help="append one JSON line per sample to FILE")
        sp.add_argument("--cpu-ram", action="store_true",
                        help="also sample system CPU%% and RAM")
        sp.add_argument("--json", action="store_true",
                        help="machine-readable output")
        sp.add_argument("--quiet", action="store_true",
                        help="no live progress output")

    sp = sub.add_parser("snapshot", help="one-shot current reading")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--cpu-ram", action="store_true")

    sp = sub.add_parser("watch", help="monitor until Ctrl+C / --duration")
    sp.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds")
    add_common(sp)

    sp = sub.add_parser("run", help="monitor while running a command (args after --)")
    sp.add_argument("--duration", type=float, default=None,
                    help="safety cap in seconds (child is killed)")
    sp.add_argument("cmd", nargs=argparse.REMAINDER)
    add_common(sp)

    args = p.parse_args()
    mode = args.mode or "watch"
    if mode == "snapshot":
        sys.exit(do_snapshot(args))
    if mode == "run":
        cmd = list(args.cmd)
        while cmd and cmd[0] == "--":
            cmd.pop(0)
        if not cmd:
            p.error("run: no command given (put it after --)")
        sys.exit(do_run(args, cmd))
    if mode == "watch":
        sys.exit(do_watch(args))


if __name__ == "__main__":
    main()
