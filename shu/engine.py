"""Run llama.cpp's server inside a cgroup cage with a thermal guard.

The cage is a transient systemd user scope: the kernel enforces the RAM and CPU
limits, so a model larger than the RAM budget is paged from disk instead of
growing past it. No root is needed.
"""

import ctypes
import glob
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from . import hardware

RESUME_MARGIN = 5.0          # degrees below the limit at which a paused server resumes
SENSOR_PATIENCE = 5          # consecutive failed readings before the guard pauses the engine
PR_SET_PDEATHSIG = 1


class EngineError(RuntimeError):
    pass


def engine_environment(lib_dir):
    """Environment that lets an Ollama-bundled llama.cpp binary find its libraries and GPU backend."""
    env = dict(os.environ)
    if not lib_dir:
        return env
    paths = [lib_dir]
    backends = sorted(glob.glob(os.path.join(lib_dir, "cuda_v*", "libggml-cuda.so")), reverse=True)
    if backends and hardware.gpus():
        paths.append(os.path.dirname(backends[0]))
        env["GGML_BACKEND_PATH"] = backends[0]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(paths)
    return env


def free_port():
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


# Options that only newer llama.cpp servers know: (flag as listed by --help, arguments to pass).
OPTIONAL_FLAGS = (
    ("--load-mode", ["--load-mode", "mmap"]),      # older builds memory-map by default
    ("--fit", ["--fit", "off"]),                   # keep the planned layer count exactly
    ("--cache-ram", ["--cache-ram", "0"]),         # no prompt cache growing in RAM
    ("--cors-origins", ["--cors-origins", "localhost"]),   # web pages elsewhere may not call the API
    ("--no-webui", ["--no-webui"]),
    ("--offline", ["--offline"]),
    ("--no-mmproj-offload", ["--no-mmproj-offload"]),
)
_flag_cache = {}


def supported_flags(binary, environment=None):
    """The optional flags this llama-server build accepts, read from its own --help."""
    if binary not in _flag_cache:
        try:
            listing = subprocess.run([binary, "--help"], env=environment, capture_output=True,
                                     text=True, timeout=60).stdout
        except (OSError, subprocess.TimeoutExpired):
            listing = ""
        # An unreadable --help means an unknown build: pass only what every version accepts.
        _flag_cache[binary] = {flag for flag, _ in OPTIONAL_FLAGS if flag in listing}
    return _flag_cache[binary]


def server_command(binary, model_path, settings, port, projector=None, host="127.0.0.1", flags=None):
    """Command line for the server; ``flags`` limits the optional ones to what the build supports."""
    command = [
        binary, "--model", model_path, "--host", host, "--port", str(port),
        "-c", str(settings["num_ctx"]),
        "-b", str(settings["num_batch"]), "-ub", str(settings["num_batch"]),
        "-ngl", str(settings["num_gpu"]),
        "-t", str(settings["num_thread"]), "-tb", str(settings["num_thread"]),
        "-np", "1",
    ]
    for flag, arguments in OPTIONAL_FLAGS:
        if flag == "--no-mmproj-offload":
            continue
        if flags is None or flag in flags:
            command += arguments
    if projector:
        command += ["--mmproj", projector]
        if flags is None or "--no-mmproj-offload" in flags:
            command += ["--no-mmproj-offload"]
    else:
        command += ["--no-mmproj"]
    return command


def cage_prefix(budget, unit):
    """systemd-run arguments that enforce the RAM and CPU share of ``budget``."""
    return [
        "systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", unit,
        # MemoryMax alone: at the ceiling the kernel reclaims model pages and carries on.
        # MemoryHigh would add deliberate stalls on top (measured: several times slower).
        "-p", f"MemoryMax={budget['ram']}",
        "-p", "MemorySwapMax=0",
        "-p", f"CPUQuota={budget['cpu_quota']}%",
        "--",
    ]


_LIBC = ctypes.CDLL(None)     # resolved once here, so nothing is loaded between fork and exec


def _die_with_parent():
    """Runs in the child before exec: the engine is terminated if Shu itself dies."""
    _LIBC.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def _read(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


class Server:
    """A llama.cpp server process, caged when the system allows it."""

    def __init__(self, model_path, settings, budget, projector=None, port=None, log_path=None,
                 report=None):
        binary, lib_dir = hardware.find_engine("llama-server")
        if not binary:
            raise EngineError(
                "llama-server was not found. Install Ollama (it bundles one) or llama.cpp, "
                "or set SHU_ENGINE_DIR to the folder that contains it."
            )
        self.port = port or free_port()
        self.budget = budget
        self.caged = hardware.can_cage()
        self.unit = f"shu-{os.getpid()}-{self.port}"
        self.environment = engine_environment(lib_dir)
        self.command = server_command(binary, model_path, settings, self.port, projector,
                                      flags=supported_flags(binary, self.environment))
        if self.caged:
            self.command = cage_prefix(budget, self.unit) + self.command
        self.log_path = log_path
        self.report = report or (lambda message: None)
        self.process = None
        self.pauses = 0
        self.sensor_failures = 0
        self.paused = False
        self._cgroup = None
        self._guard = None
        self._guard_stop = threading.Event()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    @property
    def cgroup(self):
        """Filesystem path of the cage, or None when running uncaged."""
        if self._cgroup and os.path.isdir(self._cgroup):
            return self._cgroup
        if not self.caged or not self.process:
            return None
        relative = (_read(f"/proc/{self.process.pid}/cgroup") or "").partition("::")[2]
        path = "/sys/fs/cgroup" + relative
        if relative and self.unit in path and os.path.isdir(path):
            self._cgroup = path
            return path
        return None

    def enforcement(self):
        """What the kernel is really holding the engine to, read back from the cage."""
        cage = self.cgroup
        limits = {"caged": bool(cage), "ram_hard": False, "cpu_hard": False}
        if not cage:
            return limits
        memory_max = _read(os.path.join(cage, "memory.max"))
        cpu_max = (_read(os.path.join(cage, "cpu.max")) or "").split()
        limits["ram_hard"] = bool(memory_max and memory_max.isdigit()
                                  and int(memory_max) <= self.budget["ram"])
        if len(cpu_max) == 2 and cpu_max[0].isdigit() and cpu_max[1].isdigit():
            limits["cpu_hard"] = int(cpu_max[0]) / int(cpu_max[1]) <= self.budget["cpu_quota"] / 100 + 0.01
        return limits

    def oom_kills(self):
        """Times the kernel killed something in the cage for exceeding the RAM limit."""
        cage = self.cgroup
        for line in (_read(os.path.join(cage, "memory.events")) or "").splitlines() if cage else ():
            key, _, value = line.partition(" ")
            if key == "oom_kill" and value.isdigit():
                return int(value)
        return 0

    def _failure(self, what):
        if self.oom_kills():
            return EngineError(
                f"{what}: the kernel stopped the engine because its working memory "
                f"(not the model pages) passed the RAM budget. Use a smaller --ctx or a larger --ram."
            )
        return EngineError(what + (f"; see {self.log_path}" if self.log_path
                                   else "; rerun with --log FILE for details"))

    def start(self, ready_timeout=1800):
        """Launch the engine and wait until it can serve."""
        self.launch()
        self.wait_ready(ready_timeout)

    def launch(self):
        """Start the engine process; the thermal guard runs from this moment, through loading.

        Call this before starting other threads: the child is prepared between fork and exec.
        """
        max_temp = self.budget.get("max_temp")
        if max_temp and not any(value is not None for value in hardware.temperatures().values()):
            raise EngineError(
                "--max-temp was given but this machine exposes no temperature sensor, "
                "so the thermal guard cannot work. Remove --max-temp to run without it."
            )
        log = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        try:
            self.process = subprocess.Popen(self.command, env=self.environment, stdout=log,
                                            stderr=subprocess.STDOUT, preexec_fn=_die_with_parent)
        finally:
            if self.log_path:
                log.close()
        if max_temp:
            self._start_guard(max_temp)

    def wait_ready(self, ready_timeout=1800):
        """Block until the engine answers; stop it if it dies, times out or we are interrupted."""
        try:
            deadline = time.monotonic() + ready_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise self._failure(
                        f"llama-server exited with code {self.process.returncode} while loading")
                try:
                    with urllib.request.urlopen(self.url + "/health", timeout=5) as response:
                        if response.status == 200:
                            return
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(1)
            raise EngineError("llama-server did not become ready in time")
        except BaseException:
            self.stop()
            raise

    def complete(self, prompt, tokens, timeout=7200):
        body = json.dumps({"prompt": prompt, "n_predict": tokens, "cache_prompt": False,
                           "ignore_eos": True}).encode()
        request = urllib.request.Request(self.url + "/completion", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise self._failure(f"completion request failed ({error})") from error

    def _pause(self):
        """Freeze every process in the cage; without a cage, stop the engine process itself."""
        cage = self.cgroup
        if cage:
            try:
                with open(os.path.join(cage, "cgroup.freeze"), "w") as handle:
                    handle.write("1")
                self.paused = True
                return
            except OSError:
                pass
        os.kill(self.process.pid, signal.SIGSTOP)
        self.paused = True

    def _resume(self):
        cage = self.cgroup
        if cage:
            try:
                with open(os.path.join(cage, "cgroup.freeze"), "w") as handle:
                    handle.write("0")
            except OSError:
                pass
        if self.process.poll() is None:
            os.kill(self.process.pid, signal.SIGCONT)
        self.paused = False

    def _start_guard(self, max_temp, interval=2.0):
        """Pause the engine while a sensor is at or above ``max_temp``, or while no sensor answers."""
        def loop():
            blind = 0
            while not self._guard_stop.wait(interval):
                if self.process.poll() is not None:
                    return
                readings = [value for value in hardware.temperatures().values() if value is not None]
                if not readings:
                    blind += 1
                    self.sensor_failures += 1
                    if blind == SENSOR_PATIENCE and not self.paused:
                        self._pause()
                        self.pauses += 1
                        self.report("temperature sensors stopped answering: paused until they return")
                    continue
                blind = 0
                hottest = max(readings)
                if not self.paused and hottest >= max_temp:
                    self._pause()
                    self.pauses += 1
                    self.report(f"{hottest:.0f} C reached the {max_temp:.0f} C limit: paused")
                elif self.paused and hottest <= max_temp - RESUME_MARGIN:
                    self._resume()
                    self.report(f"cooled to {hottest:.0f} C: resumed")

        self._guard = threading.Thread(target=loop, daemon=True)
        self._guard.start()

    def stop(self):
        """Stop the guard and the engine, then make sure nothing is left alive in the cage."""
        self._guard_stop.set()
        if self._guard:
            self._guard.join(timeout=10)
        cage = self.cgroup
        if self.process and self.process.poll() is None:
            self._resume()
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if cage and os.path.isdir(cage) and _read(os.path.join(cage, "cgroup.procs")):
            try:
                with open(os.path.join(cage, "cgroup.kill"), "w") as handle:
                    handle.write("1")
            except OSError:
                pass
