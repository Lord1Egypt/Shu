"""Sample the RAM, CPU, VRAM and temperature a running model actually uses."""

import glob
import os
import threading
import time

from . import hardware

_TICKS = os.sysconf("SC_CLK_TCK")
RUNNER_NAMES = ("llama-server", "ollama")


def _status(pid):
    fields = {}
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem") and parts:
                    fields[key] = int(parts[0]) * 1024
    except (OSError, ValueError):
        return None
    return fields if "VmRSS" in fields else None


def _cpu_seconds(pid):
    try:
        with open(f"/proc/{pid}/stat") as handle:
            rest = handle.read().rsplit(")", 1)[1].split()
        return (int(rest[11]) + int(rest[12])) / _TICKS
    except (OSError, ValueError, IndexError):
        return None


def find_runner(model_path):
    """PID of the inference process that has ``model_path`` on its command line."""
    newest = None
    for entry in glob.glob("/proc/[0-9]*"):
        try:
            with open(entry + "/cmdline", "rb") as handle:
                arguments = handle.read().split(b"\0")
        except OSError:
            continue
        if not arguments or os.path.basename(arguments[0].decode("utf-8", "replace")) not in RUNNER_NAMES:
            continue
        if model_path.encode() in arguments:
            pid = int(os.path.basename(entry))
            newest = pid if newest is None else max(newest, pid)
    return newest


class Sampler(threading.Thread):
    """Background sampler; ``pid`` may be a PID or a callable that finds one later."""

    def __init__(self, pid, cgroup=None, interval=0.5):
        super().__init__(daemon=True)
        self._pid = pid
        self.cgroup = cgroup
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()

    def run(self):
        last_gpu, gpu = 0.0, None
        while not self._stop_event.is_set():
            now = time.monotonic()
            pid = self._pid() if callable(self._pid) else self._pid
            if callable(self._pid) and pid:
                self._pid = pid
            if now - last_gpu >= 1.0:
                cards = hardware.gpus()
                gpu = cards[0] if cards else None
                last_gpu = now
            sample = {"time": now, "gpu": gpu, "cpu_temp": hardware.cpu_temperature()}
            if pid:
                status = _status(pid)
                if status:
                    sample.update(status)
                    sample["cpu_seconds"] = _cpu_seconds(pid)
            if self.cgroup:
                sample["cgroup"] = _read_int(os.path.join(self.cgroup, "memory.current"))
            self.samples.append(sample)
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=10)

    def summary(self, baseline_vram, since=None):
        """Peaks over the whole run; CPU usage only for samples after ``since``."""
        process = [s for s in self.samples if "VmRSS" in s]
        gpu = [s["gpu"] for s in self.samples if s.get("gpu")]
        top = max(process, key=lambda s: s["VmRSS"], default={})
        result = {
            "ram_peak": top.get("VmRSS", 0),
            "ram_private": top.get("RssAnon", 0),
            "ram_model_pages": top.get("RssFile", 0) + top.get("RssShmem", 0),
            "vram_peak": max(0, max((g["used"] for g in gpu), default=0) - baseline_vram) if gpu else 0,
            "gpu_util_avg": round(sum(g["utilization"] for g in gpu) / len(gpu), 1) if gpu else None,
            "gpu_temp_max": max((g["temperature"] for g in gpu), default=None),
            "cpu_temp_max": max((s["cpu_temp"] for s in self.samples if s.get("cpu_temp")), default=None),
            "cpu_cores_avg": None,
            "cpu_cores_peak": None,
        }
        if self.cgroup:
            peak = _read_int(os.path.join(self.cgroup, "memory.peak"))
            seen = max((s.get("cgroup") or 0 for s in self.samples), default=0)
            result["cgroup_peak"] = peak or seen
        timed = [s for s in process if s.get("cpu_seconds") is not None
                 and (since is None or s["time"] >= since)]
        if len(timed) >= 2:
            span = timed[-1]["time"] - timed[0]["time"]
            if span > 0:
                result["cpu_cores_avg"] = round(
                    (timed[-1]["cpu_seconds"] - timed[0]["cpu_seconds"]) / span, 2)
            rates = [
                (b["cpu_seconds"] - a["cpu_seconds"]) / (b["time"] - a["time"])
                for a, b in zip(timed, timed[1:]) if b["time"] > a["time"]
            ]
            if rates:
                result["cpu_cores_peak"] = round(max(rates), 2)
        return result


def _read_int(path):
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def settle_vram(seconds=20.0):
    """Wait for VRAM use to stop falling after models unload; return the baseline in bytes."""
    deadline = time.monotonic() + seconds
    previous = None
    while True:
        cards = hardware.gpus()
        if not cards:
            return 0
        used = cards[0]["used"]
        if previous is not None and used >= previous:
            return used
        if time.monotonic() >= deadline:
            return used
        previous = used
        time.sleep(1.5)


def judge(measured, budget):
    """Compare a measurement with the budget; return a list of (label, used, limit, ok)."""
    ram = max(measured.get("cgroup_peak") or 0, measured["ram_peak"])
    rows = [
        ("RAM", ram, budget["ram"], ram <= budget["ram"]),
        ("VRAM", measured["vram_peak"], budget["vram"], measured["vram_peak"] <= budget["vram"]),
    ]
    if measured.get("cpu_cores_avg") is not None:
        limit = budget["cpu_quota"] / 100
        rows.append(("CPU cores", measured["cpu_cores_avg"], limit,
                     measured["cpu_cores_avg"] <= limit * 1.05))
    if budget.get("max_temp") and measured.get("gpu_temp_max") is not None:
        rows.append(("GPU temp", measured["gpu_temp_max"], budget["max_temp"],
                     measured["gpu_temp_max"] <= budget["max_temp"]))
    return rows
