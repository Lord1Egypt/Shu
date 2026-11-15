"""Detect the CPU, RAM, GPU and thermal sensors of the current machine."""

import glob
import os
import shutil
import subprocess

MIB = 1024 * 1024
GIB = 1024 * MIB

# Ollama ships llama.cpp's server and quantizer next to its own libraries.
ENGINE_DIRS = ("/usr/local/lib/ollama", "/usr/lib/ollama", "/opt/ollama/lib/ollama")


def _meminfo():
    values = {}
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    values[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return values


def _physical_cores():
    cores = set()
    physical = core = None
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("physical id"):
                    physical = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip():
                    if core is not None:
                        cores.add((physical, core))
                    physical = core = None
    except OSError:
        pass
    return len(cores)


def _cpu_name():
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def gpus():
    """Return NVIDIA GPUs as dicts; an empty list means CPU-only planning."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    if result.returncode != 0:
        return found
    for line in result.stdout.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 5:
            continue
        try:
            found.append({
                "name": fields[0],
                "total": int(float(fields[1])) * MIB,
                "used": int(float(fields[2])) * MIB,
                "temperature": float(fields[3]),
                "utilization": float(fields[4]),
            })
        except ValueError:
            continue
    return found


def cpu_temperature():
    """Hottest CPU sensor in Celsius, or None (WSL exposes no CPU sensors)."""
    readings = []
    for pattern in ("/sys/class/thermal/thermal_zone*/temp", "/sys/class/hwmon/hwmon*/temp*_input"):
        for path in glob.glob(pattern):
            try:
                with open(path) as handle:
                    value = int(handle.read().strip()) / 1000
            except (OSError, ValueError):
                continue
            if 0 < value < 150:
                readings.append(value)
    return max(readings) if readings else None


def temperatures():
    cards = gpus()
    return {
        "gpu": max((card["temperature"] for card in cards), default=None),
        "cpu": cpu_temperature(),
    }


def can_cage():
    """True when unprivileged systemd scopes can limit memory and CPU."""
    if not shutil.which("systemd-run"):
        return False
    path = f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/cgroup.controllers"
    try:
        with open(path) as handle:
            controllers = handle.read().split()
    except OSError:
        return False
    return "memory" in controllers and "cpu" in controllers


def find_engine(name):
    """Locate a llama.cpp binary on PATH or bundled with Ollama; return (path, lib_dir)."""
    override = os.environ.get("PTAH_ENGINE_DIR")
    for directory in ([override] if override else []) + list(ENGINE_DIRS):
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate, directory
    on_path = shutil.which(name)
    if on_path:
        return on_path, None
    return None, None


def detect():
    memory = _meminfo()
    logical = os.cpu_count() or 1
    physical = _physical_cores() or logical
    cards = gpus()
    try:
        with open("/proc/version") as handle:
            wsl = "microsoft" in handle.read().lower()
    except OSError:
        wsl = False
    return {
        "cpu_name": _cpu_name(),
        "logical_cpus": logical,
        "physical_cores": physical,
        "ram_total": memory.get("MemTotal", 0),
        "ram_available": memory.get("MemAvailable", 0),
        "swap_total": memory.get("SwapTotal", 0),
        "swap_free": memory.get("SwapFree", 0),
        "gpus": cards,
        "vram_total": cards[0]["total"] if cards else 0,
        "vram_used": cards[0]["used"] if cards else 0,
        "cpu_temperature": cpu_temperature(),
        "wsl": wsl,
        "can_cage": can_cage(),
        "ollama": shutil.which("ollama"),
        "llama_server": find_engine("llama-server")[0],
        "llama_quantize": find_engine("llama-quantize")[0],
    }


def fmt_bytes(value):
    if value >= GIB:
        return f"{value / GIB:.2f} GiB"
    return f"{value / MIB:.0f} MiB"
