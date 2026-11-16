import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from shu import engine, hardware, hfconvert


def state(pid):
    with open(f"/proc/{pid}/stat") as handle:
        return handle.read().rsplit(")", 1)[1].split()[0]


def wait_for(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def bare_server(process=None, budget=None, cage=None):
    """A Server around a stand-in process, without looking for a real engine."""
    server = engine.Server.__new__(engine.Server)
    server.process = process
    server.budget = budget or {"ram": 4 << 30, "cpu_quota": 360}
    server.caged = bool(cage)
    server.unit = "shu-test"
    server.log_path = None
    server.report = lambda message: None
    server.pauses = server.sensor_failures = 0
    server.paused = False
    server._cgroup = cage
    server._guard = None
    server._guard_stop = threading.Event()
    return server


class ThermalGuardTest(unittest.TestCase):
    def setUp(self):
        self.server = bare_server(subprocess.Popen(["sleep", "60"]))
        self.addCleanup(self.server.stop)
        self.messages = []
        self.server.report = self.messages.append
        self.pid = self.server.process.pid

    def test_pauses_when_hot_and_resumes_when_cool(self):
        reading = {"gpu": 60.0, "cpu": None}
        with mock.patch.object(hardware, "temperatures", lambda: dict(reading)):
            self.server._start_guard(80.0, interval=0.02)
            time.sleep(0.2)
            self.assertNotEqual(state(self.pid), "T")
            reading["gpu"] = 81.0
            self.assertTrue(wait_for(lambda: state(self.pid) == "T"))
            reading["gpu"] = 77.0   # cooler, but not yet below the resume margin
            time.sleep(0.2)
            self.assertEqual(state(self.pid), "T")
            reading["gpu"] = 74.0
            self.assertTrue(wait_for(lambda: state(self.pid) != "T"))
            reading.update(gpu=None, cpu=90.0)   # the CPU sensor counts too
            self.assertTrue(wait_for(lambda: state(self.pid) == "T"))
            self.server.stop()
        self.assertEqual(self.server.pauses, 2)
        self.assertEqual(len(self.messages), 3)
        self.assertIsNotNone(self.server.process.returncode)

    def test_lost_sensors_pause_the_engine_until_they_return(self):
        reading = {"gpu": 60.0, "cpu": None}
        with mock.patch.object(hardware, "temperatures", lambda: dict(reading)):
            self.server._start_guard(80.0, interval=0.02)
            reading["gpu"] = None
            self.assertTrue(wait_for(lambda: state(self.pid) == "T"))
            self.assertGreaterEqual(self.server.sensor_failures, engine.SENSOR_PATIENCE)
            self.assertIn("sensors stopped answering", self.messages[0])
            reading["gpu"] = 60.0
            self.assertTrue(wait_for(lambda: state(self.pid) != "T"))
            self.server.stop()

    def test_a_short_sensor_glitch_does_not_pause(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            return {"gpu": None if calls["n"] % 3 == 0 else 60.0, "cpu": None}

        with mock.patch.object(hardware, "temperatures", flaky):
            self.server._start_guard(80.0, interval=0.01)
            time.sleep(0.4)
            self.assertNotEqual(state(self.pid), "T")
            self.assertEqual(self.server.pauses, 0)
            self.assertGreater(self.server.sensor_failures, 0)
            self.server.stop()

    def test_refuses_to_guard_without_any_sensor(self):
        server = bare_server(budget={"ram": 1 << 30, "cpu_quota": 100, "max_temp": 80})
        server.command = ["sleep", "60"]
        server.environment = dict(os.environ)
        with mock.patch.object(hardware, "temperatures", lambda: {"gpu": None, "cpu": None}):
            with self.assertRaises(engine.EngineError) as caught:
                server.start()
        self.assertIn("no temperature sensor", str(caught.exception))
        self.assertIsNone(server.process)

    def test_guard_runs_while_the_model_is_still_loading(self):
        """The guard must be live before the engine is ready, not only once it serves."""
        server = bare_server(budget={"ram": 1 << 30, "cpu_quota": 100, "max_temp": 80})
        server.command = ["sleep", "60"]      # never becomes ready, like a long load
        server.environment = dict(os.environ)
        server.port = engine.free_port()
        original = engine.Server._start_guard
        seen = {}

        def fast_guard(self, max_temp, interval=2.0):
            original(self, max_temp, interval=0.02)

        def watch():
            if wait_for(lambda: server.process is not None and state(server.process.pid) == "T"):
                seen["paused_during_load"] = True
            server._guard_stop.set()
            if server.process:
                server.process.kill()

        watcher = threading.Thread(target=watch)
        with mock.patch.object(hardware, "temperatures", lambda: {"gpu": 95.0, "cpu": None}), \
                mock.patch.object(engine.Server, "_start_guard", fast_guard):
            watcher.start()
            with self.assertRaises(engine.EngineError):
                server.start(ready_timeout=20)
            watcher.join()
        self.assertTrue(seen.get("paused_during_load"))


class CageTest(unittest.TestCase):
    def cage(self, **files):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        for name, text in files.items():
            with open(os.path.join(folder.name, name), "w") as handle:
                handle.write(text)
        return folder.name

    def test_pause_freezes_the_whole_cage(self):
        cage = self.cage(**{"cgroup.freeze": "0"})
        server = bare_server(subprocess.Popen(["sleep", "60"]), cage=cage)
        self.addCleanup(server.process.wait)
        self.addCleanup(server.process.kill)
        server._pause()
        with open(os.path.join(cage, "cgroup.freeze")) as handle:
            self.assertEqual(handle.read(), "1")
        self.assertNotEqual(state(server.process.pid), "T")   # frozen by the cage, not by a signal
        server._resume()
        with open(os.path.join(cage, "cgroup.freeze")) as handle:
            self.assertEqual(handle.read(), "0")

    def test_enforcement_is_read_back_not_assumed(self):
        budget = {"ram": 4 << 30, "cpu_quota": 360}
        held = bare_server(budget=budget, cage=self.cage(**{"memory.max": str(4 << 30), "cpu.max": "360000 100000"}))
        self.assertEqual(held.enforcement(), {"caged": True, "ram_hard": True, "cpu_hard": True})
        loose = bare_server(budget=budget, cage=self.cage(**{"memory.max": "max", "cpu.max": "max 100000"}))
        self.assertEqual(loose.enforcement(), {"caged": True, "ram_hard": False, "cpu_hard": False})
        wider = bare_server(budget=budget, cage=self.cage(**{"memory.max": str(8 << 30), "cpu.max": "800000 100000"}))
        self.assertEqual(wider.enforcement(), {"caged": True, "ram_hard": False, "cpu_hard": False})
        self.assertEqual(bare_server(budget=budget).enforcement(),
                         {"caged": False, "ram_hard": False, "cpu_hard": False})

    def test_an_oom_kill_is_reported_as_such(self):
        cage = self.cage(**{"memory.events": "low 0\nhigh 0\nmax 12\noom 1\noom_kill 1\n"})
        server = bare_server(cage=cage)
        self.assertEqual(server.oom_kills(), 1)
        self.assertIn("working memory", str(server._failure("llama-server exited")))
        calm = bare_server(cage=self.cage(**{"memory.events": "oom 0\noom_kill 0\n"}))
        self.assertEqual(calm.oom_kills(), 0)
        self.assertIn("--log", str(calm._failure("llama-server exited")))

    def test_engine_dies_with_its_parent(self):
        code = ("import subprocess, sys, time\n"
                "sys.path.insert(0, %r)\n"
                "from shu import engine\n"
                "p = subprocess.Popen(['sleep', '120'], preexec_fn=engine._die_with_parent)\n"
                "print(p.pid, flush=True); time.sleep(120)\n") % os.getcwd()
        parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        child = int(parent.stdout.readline())
        parent.kill()
        parent.wait()
        parent.stdout.close()
        self.assertTrue(wait_for(lambda: not os.path.exists(f"/proc/{child}")))


class EnvironmentTest(unittest.TestCase):
    def test_bundled_engine_gets_library_and_backend_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            for version in ("cuda_v12", "cuda_v13"):
                os.mkdir(os.path.join(folder, version))
                open(os.path.join(folder, version, "libggml-cuda.so"), "w").close()
            with mock.patch.object(hardware, "gpus", lambda: [{"name": "x"}]):
                env = engine.engine_environment(folder)
            self.assertEqual(env["GGML_BACKEND_PATH"], os.path.join(folder, "cuda_v13", "libggml-cuda.so"))
            self.assertTrue(env["LD_LIBRARY_PATH"].startswith(folder + ":" + os.path.join(folder, "cuda_v13")))
            with mock.patch.object(hardware, "gpus", lambda: []):
                self.assertNotIn("GGML_BACKEND_PATH", engine.engine_environment(folder))

    def test_engine_on_path_keeps_the_environment(self):
        self.assertEqual(engine.engine_environment(None), dict(os.environ))

    def test_find_engine_honours_override(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = os.path.join(folder, "llama-server")
            with open(binary, "w") as handle:
                handle.write("#!/bin/sh\n")
            os.chmod(binary, 0o755)
            with mock.patch.dict(os.environ, {"SHU_ENGINE_DIR": folder}):
                self.assertEqual(hardware.find_engine("llama-server"), (binary, folder))


class DiskProbeTest(unittest.TestCase):
    def test_measures_a_real_file_and_declines_a_small_one(self):
        with tempfile.TemporaryDirectory() as folder:
            small = os.path.join(folder, "small.bin")
            with open(small, "wb") as handle:
                handle.write(b"x" * 4096)
            self.assertIsNone(hardware.read_speed(small))
            big = os.path.join(folder, "big.bin")
            with open(big, "wb") as handle:
                handle.truncate(64 * hardware.MIB)
            speed = hardware.read_speed(big, extent=2 * hardware.MIB, readers=4)
            self.assertGreater(speed, 0)
            self.assertIsNone(hardware.read_speed(os.path.join(folder, "missing.bin")))


class ConverterTest(unittest.TestCase):
    def test_not_ready_in_an_empty_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.dict(os.environ, {"SHU_HOME": folder}):
                self.assertFalse(hfconvert.ready())
                with self.assertRaises(hfconvert.ConverterError):
                    hfconvert.convert(folder, os.path.join(folder, "o.gguf"))


if __name__ == "__main__":
    unittest.main()
