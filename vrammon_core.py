#!/usr/bin/env python3
"""Shared sampling code for vrammon (CLI) and vrammon_server (web dashboard).

Stdlib-only. Reads AMD amdgpu sysfs (no root needed):
  /sys/class/drm/card*/device/mem_info_vram_{used,total}
  /sys/class/drm/card*/device/gpu_busy_percent
  /proc/meminfo, /proc/stat   (system RAM / CPU)
"""

import os
import time
from datetime import datetime, timezone

DRM = "/sys/class/drm"


def find_gpus():
    """Return {card: {"total": bytes, "used": bytes, "busy": pct}}, deduped
    by PCI device so cardN and renderDN (same chip) count once."""
    gpus = {}
    seen = set()
    try:
        entries = sorted(os.listdir(DRM))
    except OSError:
        return gpus
    for e in entries:
        dev = os.path.join(DRM, e, "device")
        tot = _read_int(os.path.join(dev, "mem_info_vram_total"))
        if tot is None:
            continue
        key = os.path.realpath(dev)
        if key in seen:
            continue
        seen.add(key)
        used = _read_int(os.path.join(dev, "mem_info_vram_used")) or 0
        busy = _read_int(os.path.join(dev, "gpu_busy_percent")) or 0
        gpus[e] = {"total": tot, "used": used, "busy": busy}
    return gpus


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def read_proc_stat():
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        return [int(x) for x in parts[1:8]]
    except (OSError, ValueError, IndexError):
        return None


def sample_system(with_cpu_ram, cpu_delta):
    """Read system RAM; compute CPU% if cpu_delta is (prev_stat, prev_t).

    cpu_delta is a pair captured just before the GPU sample so the CPU%
    covers the sampling gap: ((user, nice, system, idle, iowait, irq,
    softirq) from /proc/stat, monotonic timestamp).
    """
    out = {}
    if not with_cpu_ram:
        return out
    mi = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                mi[k] = int(v.strip().split()[0])
        out["ram_total_kb"] = mi.get("MemTotal")
        out["ram_used_kb"] = mi.get("MemTotal", 0) - mi.get("MemAvailable", 0)
    except (OSError, ValueError):
        pass
    if cpu_delta:
        try:
            prev, _ = cpu_delta
            cur = read_proc_stat()
            if cur:
                d_all = sum(cur) - sum(prev)
                d_idle = (cur[3] - prev[3]) + (cur[4] - prev[4])  # idle + iowait
                out["cpu_percent"] = (
                    round(100.0 * (d_all - d_idle) / d_all, 1) if d_all > 0 else 0.0
                )
        except (OSError, ValueError, IndexError):
            pass
    return out


def gib(n):
    return n / (1024 ** 3)


def fmt_gib(nbytes):
    return f"{gib(nbytes):.2f} GiB"


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
