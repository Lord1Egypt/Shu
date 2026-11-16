"""Sample the RAM, CPU, VRAM and temperature a running model actually uses."""

import collections
import glob
import os
import threading
import time

from . import hardware

_TICKS = os.sysconf("SC_CLK_TCK")
RUNNER_NAMES = ("llama-server", "ollama")
CHARGE_SLACK = 4 * 1024 * 1024
MAX_SAMPLES = 7200           # samples kept for CPU averages; peaks are tracked separately


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


def _disk_read_bytes(pid):
    """Bytes the process has really fetched from storage (readable for our own processes only)."""
    try:
        with open(f"/proc/{pid}/io") as handle:
            for line in handle:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
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
    """Background sampler; ``pid`` may be a PID or a callable that finds one later.

    Peaks are folded into running totals, and only the newest ``MAX_SAMPLES`` samples are
    kept, so a server that runs for days does not grow this object without bound.
    """

    def __init__(self, pid, cgroup=None, interval=0.5):
        super().__init__(daemon=True)
        self._pid = pid
        self.cgroup = cgroup
        self.interval = interval
        self.samples = collections.deque(maxlen=MAX_SAMPLES)
        self._top = {}
        self._gpu_used = self._gpu_temp = self._cpu_temp = self._cgroup_seen = None
        self._gpu_util_sum = 0.0
        self.gpu_samples = self.process_samples = 0
        self._stop_event = threading.Event()

    def record(self, sample):
        """Fold one sample into the peaks and the bounded history."""
        gpu = sample.get("gpu")
        if gpu:
            self.gpu_samples += 1
            self._gpu_util_sum += gpu["utilization"]
            self._gpu_used = max(self._gpu_used or 0, gpu["used"])
            self._gpu_temp = max(self._gpu_temp or 0, gpu["temperature"])
        if sample.get("cpu_temp"):
            self._cpu_temp = max(self._cpu_temp or 0, sample["cpu_temp"])
        if "VmRSS" in sample:
            self.process_samples += 1
            if sample["VmRSS"] >= self._top.get("VmRSS", 0):
                self._top = sample
        if sample.get("cgroup") is not None:
            self._cgroup_seen = max(self._cgroup_seen or 0, sample["cgroup"])
        self.samples.append(sample)

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
                    sample["disk_read"] = _disk_read_bytes(pid)
            if self.cgroup:
                sample["cgroup"] = _read_int(os.path.join(self.cgroup, "memory.current"))
            self.record(sample)
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=10)

    def summary(self, baseline_vram, since=None):
        """Peaks over the whole run; CPU and disk figures only for samples after ``since``.

        A value that could not be measured is None, never zero: a missing reading must not
        look like a model that used nothing.
        """
        top = self._top
        result = {
            "ram_peak": top.get("VmRSS"),
            "ram_private": top.get("RssAnon"),
            "ram_model_pages": (top.get("RssFile", 0) + top.get("RssShmem", 0)) if top else None,
            "vram_peak": max(0, self._gpu_used - baseline_vram) if self._gpu_used is not None else None,
            "gpu_util_avg": round(self._gpu_util_sum / self.gpu_samples, 1) if self.gpu_samples else None,
            "gpu_temp_max": self._gpu_temp,
            "cpu_temp_max": self._cpu_temp,
            "cpu_cores_avg": None,
            "cpu_cores_peak": None,
            "disk_read_bytes": None,
            "process_samples": self.process_samples,
            "gpu_samples": self.gpu_samples,
        }
        if self.cgroup:
            result["cgroup_peak"] = (_read_int(os.path.join(self.cgroup, "memory.peak"))
                                     or self._cgroup_seen)
        timed = [s for s in self.samples if s.get("cpu_seconds") is not None
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
            if timed[0].get("disk_read") is not None and timed[-1].get("disk_read") is not None:
                result["disk_read_bytes"] = timed[-1]["disk_read"] - timed[0]["disk_read"]
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


def judge(measured, budget, gpu_expected=True, caged=False):
    """Compare a measurement with the budget.

    Returns rows of (label, used, limit, ok). ``ok`` is None when the value could not be
    measured: that is reported as unverified, never as a pass.

    RAM is what the kernel charged to the cage when there is one: that is the number the
    limit applies to, and the model file is dropped from the page cache before a caged run
    so all of its pages are charged there. The process's own resident size also counts
    shared system libraries that sit in memory regardless. Without a cage it is all we have.
    """
    ram = measured.get("ram_peak")
    if caged and ram is not None and measured.get("cgroup_peak") is not None:
        ram = measured["cgroup_peak"]
    # The kernel charges a cage in per-CPU batches, so its peak counter can read a few pages
    # above the limit it enforces. That is accounting slack, not memory the engine obtained.
    rows = [("RAM", ram, budget["ram"], None if ram is None else ram <= budget["ram"] + CHARGE_SLACK)]
    if gpu_expected or budget["vram"]:
        vram = measured.get("vram_peak")
        rows.append(("VRAM", vram, budget["vram"], None if vram is None else vram <= budget["vram"]))
    cores, limit = measured.get("cpu_cores_avg"), budget["cpu_quota"] / 100
    rows.append(("CPU cores", cores, limit, None if cores is None else cores <= limit * 1.05))
    if budget.get("max_temp"):
        temps = [value for value in (measured.get("gpu_temp_max"), measured.get("cpu_temp_max"))
                 if value is not None]
        hottest = max(temps) if temps else None
        rows.append(("Temperature", hottest, budget["max_temp"],
                     None if hottest is None else hottest <= budget["max_temp"]))
    return rows


def verdict(rows):
    """0 when everything was measured inside the budget, 2 when something is over,
    3 when nothing is over but something could not be measured."""
    states = [row[3] for row in rows]
    if any(state is False for state in states):
        return 2
    return 3 if any(state is None for state in states) else 0
