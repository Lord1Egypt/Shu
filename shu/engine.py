"""Run llama.cpp's server inside a cgroup cage with a thermal guard.

The cage is a transient systemd user scope: the kernel enforces the RAM and CPU
limits, so a model larger than the RAM budget is paged from disk instead of
growing past it. No root is needed.
"""

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


def server_command(binary, model_path, settings, port, projector=None, host="127.0.0.1"):
    command = [
        binary, "--model", model_path, "--host", host, "--port", str(port),
        "-c", str(settings["num_ctx"]),
        "-b", str(settings["num_batch"]), "-ub", str(settings["num_batch"]),
        "-ngl", str(settings["num_gpu"]),
        "-t", str(settings["num_thread"]), "-tb", str(settings["num_thread"]),
        "--load-mode", "mmap", "--fit", "off", "--cache-ram", "0",
        "-np", "1", "--no-webui", "--offline",
    ]
    if projector:
        command += ["--mmproj", projector, "--no-mmproj-offload"]
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


class Server:
    """A llama.cpp server process, caged when the system allows it."""

    def __init__(self, model_path, settings, budget, projector=None, port=None, log_path=None):
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
        self.command = server_command(binary, model_path, settings, self.port, projector)
        if self.caged:
            self.command = cage_prefix(budget, self.unit) + self.command
        self.environment = engine_environment(lib_dir)
        self.log_path = log_path
        self.process = None
        self.pauses = 0
        self._guard = None
        self._guard_stop = threading.Event()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    @property
    def cgroup(self):
        """Filesystem path of the cage, or None when running uncaged."""
        if not self.caged or not self.process:
            return None
        try:
            with open(f"/proc/{self.process.pid}/cgroup") as handle:
                relative = handle.read().strip().split("::", 1)[1]
        except (OSError, IndexError):
            return None
        path = "/sys/fs/cgroup" + relative
        return path if self.unit in path and os.path.isdir(path) else None

    def start(self, ready_timeout=1800):
        log = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        try:
            self.process = subprocess.Popen(self.command, env=self.environment,
                                            stdout=log, stderr=subprocess.STDOUT)
        finally:
            if self.log_path:
                log.close()
        try:
            deadline = time.monotonic() + ready_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise EngineError(
                        f"llama-server exited with code {self.process.returncode} while loading"
                        + (f"; see {self.log_path}" if self.log_path else "; rerun with --log FILE for details")
                    )
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
        except (urllib.error.URLError, OSError) as error:
            raise EngineError(f"completion request failed: {error}") from error

    def start_guard(self, max_temp, interval=2.0, report=None):
        """Pause the server while a sensor is at or above ``max_temp`` degrees Celsius."""
        def loop():
            paused = False
            while not self._guard_stop.wait(interval):
                if self.process.poll() is not None:
                    return
                readings = [value for value in hardware.temperatures().values() if value is not None]
                if not readings:
                    continue
                hottest = max(readings)
                if not paused and hottest >= max_temp:
                    os.kill(self.process.pid, signal.SIGSTOP)
                    paused = True
                    self.pauses += 1
                    if report:
                        report(f"{hottest:.0f} C reached the {max_temp:.0f} C limit: paused")
                elif paused and hottest <= max_temp - RESUME_MARGIN:
                    os.kill(self.process.pid, signal.SIGCONT)
                    paused = False
                    if report:
                        report(f"cooled to {hottest:.0f} C: resumed")
            if paused and self.process.poll() is None:
                os.kill(self.process.pid, signal.SIGCONT)

        self._guard = threading.Thread(target=loop, daemon=True)
        self._guard.start()

    def stop(self):
        self._guard_stop.set()
        if self._guard:
            self._guard.join(timeout=10)
        if self.process and self.process.poll() is None:
            os.kill(self.process.pid, signal.SIGCONT)
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
