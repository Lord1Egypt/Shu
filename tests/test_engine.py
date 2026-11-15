import os
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

from ptah import engine, hardware, hfconvert


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


class ThermalGuardTest(unittest.TestCase):
    def test_pauses_when_hot_and_resumes_when_cool(self):
        server = engine.Server.__new__(engine.Server)
        server.process = subprocess.Popen(["sleep", "60"])
        server.pauses = 0
        server._guard = None
        server._guard_stop = threading.Event()
        self.addCleanup(server.stop)
        reading = {"gpu": 60.0, "cpu": None}
        messages = []
        with mock.patch.object(hardware, "temperatures", lambda: dict(reading)):
            server.start_guard(80.0, interval=0.02, report=messages.append)
            time.sleep(0.2)
            self.assertNotEqual(state(server.process.pid), "T")
            reading["gpu"] = 81.0
            self.assertTrue(wait_for(lambda: state(server.process.pid) == "T"))
            reading["gpu"] = 77.0   # cooler, but not yet below the resume margin
            time.sleep(0.2)
            self.assertEqual(state(server.process.pid), "T")
            reading["gpu"] = 74.0
            self.assertTrue(wait_for(lambda: state(server.process.pid) != "T"))
            reading.update(gpu=None, cpu=90.0)   # the CPU sensor counts too
            self.assertTrue(wait_for(lambda: state(server.process.pid) == "T"))
            server.stop()
        self.assertEqual(server.pauses, 2)
        self.assertEqual(len(messages), 3)
        self.assertIsNotNone(server.process.returncode)


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
            with mock.patch.dict(os.environ, {"PTAH_ENGINE_DIR": folder}):
                self.assertEqual(hardware.find_engine("llama-server"), (binary, folder))


class ConverterTest(unittest.TestCase):
    def test_not_ready_in_an_empty_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.dict(os.environ, {"PTAH_HOME": folder}):
                self.assertFalse(hfconvert.ready())
                with self.assertRaises(hfconvert.ConverterError):
                    hfconvert.convert(folder, os.path.join(folder, "o.gguf"))


if __name__ == "__main__":
    unittest.main()
