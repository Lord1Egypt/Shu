import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from ptah import actions, cli, engine, monitor, ollama, planner, sources
from ptah.hardware import GIB, MIB
from tests import fakegguf


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.model = fakegguf.language_model(os.path.join(self.dir.name, "tiny.gguf"))

    def test_no_arguments_without_a_terminal_prints_help_and_returns(self):
        code, out, _ = run()
        self.assertEqual(code, 0)
        self.assertIn("usage: ptah", out)

    def test_detect_json(self):
        code, out, _ = run("detect", "--json")
        self.assertEqual(code, 0)
        self.assertIn("logical_cpus", json.loads(out))

    def test_inspect(self):
        code, out, _ = run("inspect", self.model)
        self.assertEqual(code, 0)
        self.assertIn("Tiny Test", out)
        code, out, _ = run("inspect", self.model, "--json")
        self.assertEqual(json.loads(out)["layers"], 4)

    def test_plan_json_respects_overrides(self):
        code, out, _ = run("plan", self.model, "--ram", "2", "--vram", "0", "--cpu", "2",
                           "--ctx", "2048", "--json")
        self.assertEqual(code, 0)
        plan = json.loads(out)
        self.assertEqual(plan["budget"]["ram"], 2 * GIB)
        self.assertEqual(plan["settings"]["num_thread"], 2)
        self.assertEqual(plan["settings"]["num_ctx"], 2048)
        self.assertEqual(plan["settings"]["num_gpu"], 0)
        self.assertEqual(plan["status"], "cpu")

    def test_plan_human(self):
        code, out, _ = run("plan", self.model, "--percent", "25")
        self.assertEqual(code, 0)
        self.assertIn("Settings: num_gpu=", out)

    def test_apply_no_create_writes_files(self):
        out_dir = os.path.join(self.dir.name, "out")
        code, out, _ = run("apply", self.model, "--no-create", "--out", out_dir, "--vram", "0")
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(os.listdir(out_dir)),
                         ["Modelfile", "lmstudio.md", "plan.json", "run-llama-server.sh"])
        with open(os.path.join(out_dir, "Modelfile")) as handle:
            modelfile = handle.read()
        self.assertTrue(modelfile.startswith(f"FROM {self.model}\n"))
        self.assertIn("PARAMETER num_gpu 0", modelfile)
        self.assertIn("PARAMETER use_mmap true", modelfile)
        with open(os.path.join(out_dir, "plan.json")) as handle:
            self.assertEqual(json.load(handle)["model_file"], self.model)
        self.assertTrue(os.access(os.path.join(out_dir, "run-llama-server.sh"), os.X_OK))

    def test_errors_return_one(self):
        for argv in (("plan", os.path.join(self.dir.name, "missing.gguf")),
                     ("plan", self.model, "--percent", "0"),
                     ("convert", self.model),
                     ("apply", self.model, "--no-create", "--name=-rf")):
            code, _, err = run(*argv)
            self.assertEqual(code, 1, argv)
            self.assertTrue(err.startswith("ptah: "), err)

    def test_safetensors_folder_is_recognised(self):
        folder = os.path.join(self.dir.name, "hf")
        os.mkdir(folder)
        for name in ("config.json", "model.safetensors"):
            open(os.path.join(folder, name), "w").close()
        self.assertEqual(sources.resolve(folder).kind, "safetensors")
        code, _, err = run("plan", folder)
        self.assertEqual(code, 1)
        self.assertIn("ptah convert", err)

    def test_convert_safetensors_needs_the_converter(self):
        folder = os.path.join(self.dir.name, "hf")
        os.mkdir(folder)
        for name in ("config.json", "model.safetensors"):
            open(os.path.join(folder, name), "w").close()
        with mock.patch.dict(os.environ, {"PTAH_HOME": os.path.join(self.dir.name, "cache")}):
            code, _, err = run("convert", folder, "--out", os.path.join(self.dir.name, "o.gguf"))
        self.assertEqual(code, 1)
        self.assertIn("setup-converter", err)


class PiecesTest(unittest.TestCase):
    def test_model_names_are_validated(self):
        for bad in ("-flag", "a b", "a;b", "$(x)", "", "a" * 300):
            with self.assertRaises(ollama.OllamaError):
                ollama.check_name(bad)
        for good in ("qwen3.5:4b", "hf.co/user/repo-GGUF:Q4_K_M", "Thoth"):
            self.assertEqual(ollama.check_name(good), good)

    def test_capped_modelfile(self):
        text = ollama.capped_modelfile("qwen3.5:4b", {"num_gpu": 7, "num_thread": 3, "num_ctx": 4096,
                                                      "num_batch": 512, "use_mmap": True})
        self.assertEqual(text.splitlines(), [
            "FROM qwen3.5:4b", "PARAMETER num_gpu 7", "PARAMETER num_thread 3",
            "PARAMETER num_ctx 4096", "PARAMETER num_batch 512", "PARAMETER use_mmap true"])

    def test_rewritten_modelfile_drops_projector_and_old_limits(self):
        original = ("FROM /blobs/model\nFROM /blobs/projector\nTEMPLATE \"x\"\n"
                    "PARAMETER temperature 1\nPARAMETER num_ctx 99\n")
        text = ollama.rewritten_modelfile(original, {"num_ctx": 2048}, ["/blobs/model"])
        self.assertIn("FROM /blobs/model", text)
        self.assertNotIn("/blobs/projector", text)
        self.assertIn("PARAMETER temperature 1", text)
        self.assertNotIn("num_ctx 99", text)
        self.assertIn("PARAMETER num_ctx 2048", text)
        swapped = ollama.rewritten_modelfile(original, {}, ["/blobs/model", "/blobs/projector"],
                                             replace={"/blobs/model": "/new.gguf"})
        self.assertIn("FROM /new.gguf", swapped)
        self.assertIn("FROM /blobs/projector", swapped)

    def test_source_files(self):
        self.assertEqual(ollama.source_files("FROM /a\nfrom /b\nFROM name:tag\nSYSTEM FROM /c"),
                         ["/a", "/b"])

    def test_host_normalisation(self):
        for value, expected in (("127.0.0.1:11435", "http://127.0.0.1:11435"),
                                ("localhost", "http://localhost:11434"),
                                ("http://box:1234/", "http://box:1234")):
            with mock.patch.dict(os.environ, {"OLLAMA_HOST": value}):
                self.assertEqual(ollama.host(), expected)

    def test_default_names(self):
        self.assertEqual(sources.Source("ollama", "qwen3.5:4b").default_name(), "qwen3.5-4b-ptah")
        self.assertEqual(sources.Source("ollama", "Thoth:latest").default_name(), "thoth-ptah")
        self.assertEqual(sources.Source("gguf", "/m/My Model.Q4_K_M.gguf").default_name(),
                         "my-model.q4_k_m-ptah")
        long_name = sources.Source("gguf", "/m/" + "x" * 200 + ".gguf").default_name()
        self.assertLessEqual(len(long_name), 80)

    def test_cage_holds_ram_at_the_budget(self):
        budget = {"ram": 4 * GIB, "cpu_quota": 360}
        prefix = engine.cage_prefix(budget, "ptah-test")
        self.assertIn(f"MemoryMax={4 * GIB}", prefix)
        self.assertIn(f"MemoryHigh={int(4 * GIB * engine.RECLAIM_POINT)}", prefix)
        self.assertIn("MemorySwapMax=0", prefix)
        self.assertIn("CPUQuota=360%", prefix)
        self.assertEqual(prefix[-1], "--")

    def test_server_command(self):
        settings = {"num_ctx": 2048, "num_batch": 128, "num_gpu": 9, "num_thread": 3}
        command = engine.server_command("/bin/llama-server", "/m.gguf", settings, 8080)
        joined = " ".join(command)
        for piece in ("--model /m.gguf", "-c 2048", "-b 128", "-ngl 9", "-t 3", "--load-mode mmap",
                      "--cache-ram 0", "--no-mmproj", "--host 127.0.0.1"):
            self.assertIn(piece, joined)
        with_projector = engine.server_command("/bin/llama-server", "/m.gguf", settings, 8080, "/p.gguf")
        self.assertIn("--mmproj", with_projector)
        self.assertNotIn("--no-mmproj", with_projector)

    def test_judge(self):
        budget = {"ram": 4 * GIB, "vram": 2 * GIB, "cpu_quota": 360, "max_temp": 80}
        good = {"ram_peak": 3 * GIB, "vram_peak": GIB, "cpu_cores_avg": 2.5, "gpu_temp_max": 70}
        self.assertTrue(all(row[3] for row in monitor.judge(good, budget)))
        bad = {"ram_peak": 3 * GIB, "cgroup_peak": 5 * GIB, "vram_peak": 3 * GIB,
               "cpu_cores_avg": 5.0, "gpu_temp_max": 91}
        self.assertFalse(any(row[3] for row in monitor.judge(bad, budget)))
        rows = actions.describe_checks(monitor.judge(bad, budget))
        self.assertEqual(rows[0][:2], ("RAM", "5.00 GiB"))
        self.assertEqual(rows[0][4], "OVER")

    def test_sampler_summary(self):
        sampler = monitor.Sampler(1)
        def gpu(used):
            return {"used": used * MIB, "utilization": 50.0, "temperature": 60.0}

        sampler.samples = [
            {"time": 0.0, "gpu": gpu(300), "cpu_temp": None},
            {"time": 1.0, "gpu": gpu(1500), "cpu_temp": None, "VmRSS": 2 * GIB, "RssAnon": GIB,
             "RssFile": GIB, "RssShmem": 0, "cpu_seconds": 10.0},
            {"time": 2.0, "gpu": gpu(1800), "cpu_temp": None, "VmRSS": 3 * GIB, "RssAnon": GIB,
             "RssFile": 2 * GIB, "RssShmem": 0, "cpu_seconds": 13.0},
            {"time": 3.0, "gpu": gpu(1700), "cpu_temp": None, "VmRSS": 2 * GIB, "RssAnon": GIB,
             "RssFile": GIB, "RssShmem": 0, "cpu_seconds": 15.0},
        ]
        summary = sampler.summary(300 * MIB, since=1.0)
        self.assertEqual(summary["ram_peak"], 3 * GIB)
        self.assertEqual(summary["ram_private"] + summary["ram_model_pages"], 3 * GIB)
        self.assertEqual(summary["vram_peak"], 1500 * MIB)
        self.assertEqual(summary["cpu_cores_avg"], 2.5)
        self.assertEqual(summary["cpu_cores_peak"], 3.0)

    def test_split_model_cannot_go_to_ollama(self):
        with tempfile.TemporaryDirectory() as folder:
            meta = {"general.architecture": "llama", "llama.block_count": 2,
                    "llama.context_length": 4096, "llama.embedding_length": 64,
                    "llama.attention.head_count": 4, "split.count": 2}
            first = os.path.join(folder, "m-00001-of-00002.gguf")
            fakegguf.write(first, meta, [("blk.0.w", [64], 2048)])
            fakegguf.write(os.path.join(folder, "m-00002-of-00002.gguf"), {"split.count": 2},
                           [("blk.1.w", [64], 2048)])
            source = sources.resolve(first)
            with self.assertRaises(actions.ActionError):
                actions.modelfile_for(source, {"num_gpu": 0})

    def test_paths_with_control_characters_never_reach_a_modelfile(self):
        with tempfile.TemporaryDirectory() as folder:
            path = fakegguf.language_model(os.path.join(folder, "evil\nSYSTEM pwned.gguf"))
            with self.assertRaises(actions.ActionError):
                actions.modelfile_for(sources.resolve(path), {"num_gpu": 0})

    def test_tuning_steps_back_after_overshoot(self):
        """The loop grows once, overshoots, and settles on the measured in-budget count."""
        with tempfile.TemporaryDirectory() as folder:
            source = sources.resolve(fakegguf.language_model(os.path.join(folder, "m.gguf"),
                                                             layers=8, layer_bytes=64 * MIB // 1024))
            hw = {"logical_cpus": 12, "ram_total": 16 * GIB, "vram_total": 6 * GIB}
            plan = planner.plan(source.model, planner.make_budget(hw, 30))
            plan["settings"]["num_gpu"] = 4
            step = actions.layer_cost(source, plan["settings"])
            budget = plan["budget"]["vram"]
            usage = {4: int(budget * actions.GROW_TARGET) - 2 * step - 1, 6: budget + 1,
                     5: budget - step // 2}
            calls = []

            def fake_bench(name, bench_budget, tokens=64, **_):
                layers = plan["settings"]["num_gpu"]
                calls.append(layers)
                return {"measured": {"vram_peak": usage[layers]}, "checks": []}

            with mock.patch.object(actions, "bench_ollama", fake_bench), \
                    mock.patch.object(ollama, "create", lambda *a, **k: None):
                result = actions.apply(source, plan, "tiny-ptah", os.path.join(folder, "out"),
                                       verify=True, report=lambda message: None)
            self.assertEqual(calls, [4, 6, 5])
            self.assertEqual(plan["settings"]["num_gpu"], 5)
            self.assertEqual(result["rounds"][-1]["num_gpu"], 5)
            with open(os.path.join(folder, "out", "Modelfile")) as handle:
                self.assertIn("PARAMETER num_gpu 5", handle.read())

    def test_tuning_leaves_headroom_instead_of_filling_the_last_megabytes(self):
        with tempfile.TemporaryDirectory() as folder:
            source = sources.resolve(fakegguf.language_model(os.path.join(folder, "m.gguf"), layers=8))
            hw = {"logical_cpus": 12, "ram_total": 16 * GIB, "vram_total": 6 * GIB}
            plan = planner.plan(source.model, planner.make_budget(hw, 30))
            plan["settings"]["num_gpu"] = 4
            step = actions.layer_cost(source, plan["settings"])
            used = plan["budget"]["vram"] - step - 1    # one more layer fits, but only just
            calls = []

            def fake_bench(name, bench_budget, tokens=64, **_):
                calls.append(plan["settings"]["num_gpu"])
                return {"measured": {"vram_peak": used}, "checks": []}

            with mock.patch.object(actions, "bench_ollama", fake_bench), \
                    mock.patch.object(ollama, "create", lambda *a, **k: None):
                actions.apply(source, plan, "tiny-ptah", os.path.join(folder, "out"),
                              verify=True, report=lambda message: None)
            self.assertEqual(calls, [4])
            self.assertEqual(plan["settings"]["num_gpu"], 4)


if __name__ == "__main__":
    unittest.main()
