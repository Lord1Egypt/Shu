import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from shu import actions, cli, engine, monitor, ollama, planner, sources
from shu.hardware import GIB, MIB
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
        self.assertIn("usage: shu", out)

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
            self.assertTrue(err.startswith("shu: "), err)

    def test_safetensors_folder_is_recognised(self):
        folder = os.path.join(self.dir.name, "hf")
        os.mkdir(folder)
        for name in ("config.json", "model.safetensors"):
            open(os.path.join(folder, name), "w").close()
        self.assertEqual(sources.resolve(folder).kind, "safetensors")
        code, _, err = run("plan", folder)
        self.assertEqual(code, 1)
        self.assertIn("shu convert", err)

    def test_convert_safetensors_needs_the_converter(self):
        folder = os.path.join(self.dir.name, "hf")
        os.mkdir(folder)
        for name in ("config.json", "model.safetensors"):
            open(os.path.join(folder, name), "w").close()
        with mock.patch.dict(os.environ, {"SHU_HOME": os.path.join(self.dir.name, "cache")}):
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
        self.assertEqual(sources.Source("ollama", "qwen3.5:4b").default_name(), "qwen3.5-4b-shu")
        self.assertEqual(sources.Source("ollama", "Thoth:latest").default_name(), "thoth-shu")
        self.assertEqual(sources.Source("gguf", "/m/My Model.Q4_K_M.gguf").default_name(),
                         "my-model.q4_k_m-shu")
        long_name = sources.Source("gguf", "/m/" + "x" * 200 + ".gguf").default_name()
        self.assertLessEqual(len(long_name), 80)

    def test_cage_holds_ram_at_the_budget(self):
        budget = {"ram": 4 * GIB, "cpu_quota": 360}
        prefix = engine.cage_prefix(budget, "shu-test")
        self.assertIn(f"MemoryMax={4 * GIB}", prefix)
        self.assertFalse(any(part.startswith("MemoryHigh") for part in prefix))
        self.assertIn("MemorySwapMax=0", prefix)
        self.assertIn("CPUQuota=360%", prefix)
        self.assertEqual(prefix[-1], "--")

    def test_server_command(self):
        settings = {"num_ctx": 2048, "num_batch": 128, "num_gpu": 9, "num_thread": 3}
        command = engine.server_command("/bin/llama-server", "/m.gguf", settings, 8080)
        joined = " ".join(command)
        for piece in ("--model /m.gguf", "-c 2048", "-b 128", "-ngl 9", "-t 3", "--load-mode mmap",
                      "--cache-ram 0", "--no-mmproj", "--host 127.0.0.1", "--cors-origins localhost"):
            self.assertIn(piece, joined)
        old_build = engine.server_command("/bin/llama-server", "/m.gguf", settings, 8080,
                                          "/p.gguf", flags={"--cache-ram"})
        self.assertIn("--cache-ram", old_build)
        for unknown in ("--load-mode", "--fit", "--cors-origins", "--offline", "--no-mmproj-offload"):
            self.assertNotIn(unknown, old_build)
        self.assertIn("--mmproj", old_build)
        with_projector = engine.server_command("/bin/llama-server", "/m.gguf", settings, 8080, "/p.gguf")
        self.assertIn("--mmproj", with_projector)
        self.assertNotIn("--no-mmproj", with_projector)

    def test_judge(self):
        budget = {"ram": 4 * GIB, "vram": 2 * GIB, "cpu_quota": 360, "max_temp": 80}
        good = {"ram_peak": 3 * GIB, "vram_peak": GIB, "cpu_cores_avg": 2.5, "gpu_temp_max": 70}
        rows = monitor.judge(good, budget)
        self.assertEqual([row[0] for row in rows], ["RAM", "VRAM", "CPU cores", "Temperature"])
        self.assertTrue(all(row[3] is True for row in rows))
        self.assertEqual(monitor.verdict(rows), 0)
        bad = {"ram_peak": 5 * GIB, "vram_peak": 3 * GIB, "cpu_cores_avg": 5.0, "gpu_temp_max": 91}
        self.assertTrue(all(row[3] is False for row in monitor.judge(bad, budget)))
        self.assertEqual(monitor.verdict(monitor.judge(bad, budget)), 2)
        rows = actions.describe_checks(monitor.judge(bad, budget))
        self.assertEqual(rows[0][:2], ("RAM", "5.00 GiB"))
        self.assertEqual(rows[0][4], "OVER")

    def test_in_a_cage_ram_is_what_the_kernel_charged(self):
        budget = {"ram": 4 * GIB, "vram": 2 * GIB, "cpu_quota": 360}
        # Resident size a little over (shared libraries), cage exactly at its limit plus slack.
        run = {"ram_peak": 4 * GIB + 40 * MIB, "cgroup_peak": 4 * GIB + 4096, "vram_peak": GIB,
               "cpu_cores_avg": 1.0}
        caged = monitor.judge(run, budget, caged=True)[0]
        self.assertEqual((caged[1], caged[3]), (4 * GIB + 4096, True))
        loose = monitor.judge(run, budget, caged=False)[0]
        self.assertEqual((loose[1], loose[3]), (4 * GIB + 40 * MIB, False))
        beyond = dict(run, cgroup_peak=4 * GIB + 64 * MIB)
        self.assertFalse(monitor.judge(beyond, budget, caged=True)[0][3])

    def test_missing_readings_are_never_a_pass(self):
        """A run where the process or the GPU could not be read must not look like success."""
        budget = {"ram": 4 * GIB, "vram": 2 * GIB, "cpu_quota": 360, "max_temp": 80}
        blind = monitor.Sampler(None).summary(0)
        self.assertIsNone(blind["ram_peak"])
        self.assertIsNone(blind["vram_peak"])
        rows = monitor.judge(blind, budget)
        self.assertEqual([row[3] for row in rows], [None, None, None, None])
        self.assertEqual(monitor.verdict(rows), 3)
        self.assertEqual({row[4] for row in actions.describe_checks(rows)}, {"NOT MEASURED"})
        partly = {"ram_peak": GIB, "vram_peak": None, "cpu_cores_avg": 1.0, "gpu_temp_max": None}
        self.assertEqual(monitor.verdict(monitor.judge(partly, budget)), 3)
        over_and_blind = {"ram_peak": 9 * GIB, "vram_peak": None, "cpu_cores_avg": None}
        self.assertEqual(monitor.verdict(monitor.judge(over_and_blind, budget)), 2)
        # A cage reading alone does not stand in for the process reading.
        self.assertIsNone(monitor.judge({"cgroup_peak": GIB}, budget, caged=True)[0][3])

    def test_no_vram_row_on_a_machine_without_a_gpu(self):
        budget = {"ram": 4 * GIB, "vram": 0, "cpu_quota": 360}
        rows = monitor.judge({"ram_peak": GIB, "cpu_cores_avg": 1.0}, budget, gpu_expected=False)
        self.assertEqual([row[0] for row in rows], ["RAM", "CPU cores"])
        self.assertEqual(monitor.verdict(rows), 0)

    def test_sampler_summary(self):
        sampler = monitor.Sampler(1)

        def gpu(used):
            return {"used": used * MIB, "utilization": 50.0, "temperature": 60.0}

        for sample in (
            {"time": 0.0, "gpu": gpu(300), "cpu_temp": None},
            {"time": 1.0, "gpu": gpu(1500), "cpu_temp": None, "VmRSS": 2 * GIB, "RssAnon": GIB,
             "RssFile": GIB, "RssShmem": 0, "cpu_seconds": 10.0, "disk_read": 100 * MIB},
            {"time": 2.0, "gpu": gpu(1800), "cpu_temp": None, "VmRSS": 3 * GIB, "RssAnon": GIB,
             "RssFile": 2 * GIB, "RssShmem": 0, "cpu_seconds": 13.0, "disk_read": 400 * MIB},
            {"time": 3.0, "gpu": gpu(1700), "cpu_temp": None, "VmRSS": 2 * GIB, "RssAnon": GIB,
             "RssFile": GIB, "RssShmem": 0, "cpu_seconds": 15.0, "disk_read": 900 * MIB},
        ):
            sampler.record(sample)
        summary = sampler.summary(300 * MIB, since=1.0)
        self.assertEqual(summary["ram_peak"], 3 * GIB)
        self.assertEqual(summary["ram_private"] + summary["ram_model_pages"], 3 * GIB)
        self.assertEqual(summary["vram_peak"], 1500 * MIB)
        self.assertEqual(summary["cpu_cores_avg"], 2.5)
        self.assertEqual(summary["cpu_cores_peak"], 3.0)
        self.assertEqual(summary["disk_read_bytes"], 800 * MIB)
        self.assertEqual((summary["process_samples"], summary["gpu_samples"]), (3, 4))

    def test_sampler_history_is_bounded_but_peaks_survive(self):
        sampler = monitor.Sampler(1)
        sampler.record({"time": 0.0, "gpu": {"used": 5 * GIB, "utilization": 90.0, "temperature": 88.0},
                        "cpu_temp": None, "VmRSS": 7 * GIB, "RssAnon": 7 * GIB, "cpu_seconds": 0.0})
        for index in range(1, monitor.MAX_SAMPLES * 2):
            sampler.record({"time": float(index), "gpu": None, "cpu_temp": None,
                            "VmRSS": GIB, "RssAnon": GIB, "cpu_seconds": float(index)})
        self.assertEqual(len(sampler.samples), monitor.MAX_SAMPLES)
        summary = sampler.summary(0)
        self.assertEqual(summary["ram_peak"], 7 * GIB)        # from a sample long since dropped
        self.assertEqual(summary["vram_peak"], 5 * GIB)
        self.assertEqual(summary["gpu_temp_max"], 88.0)

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

    def tuning(self, folder, measurements, layers=8, start_gpu=4, status=None, rounds=None):
        """Run the tuning loop against scripted measurements; returns (plan, result, calls)."""
        source = sources.resolve(fakegguf.language_model(os.path.join(folder, "m.gguf"), layers=layers))
        hw = {"logical_cpus": 12, "ram_total": 16 * GIB, "vram_total": 6 * GIB}
        plan = planner.plan(source.model, planner.make_budget(hw, 30))
        plan["settings"]["num_gpu"] = start_gpu
        if status:
            plan["status"] = status
        calls, creations = [], []

        def fake_bench(name, budget, tokens=64, **_):
            settings = dict(plan["settings"])
            calls.append(settings)
            measured = measurements(settings, plan, actions.layer_cost(source, settings))
            return {"measured": measured, "checks": monitor.judge(measured, budget),
                    "installed": creations[-1]}

        def fake_create(name, path, **_):
            with open(path) as handle:
                creations.append(handle.read())

        with mock.patch.object(actions, "bench_ollama", fake_bench), \
                mock.patch.object(ollama, "create", fake_create), \
                mock.patch.object(actions, "MAX_TUNE_ROUNDS", rounds or actions.MAX_TUNE_ROUNDS):
            result = actions.apply(source, plan, "tiny-shu", os.path.join(folder, "out"),
                                   verify=True, report=lambda message: None)
        return plan, result, calls

    def test_tuning_steps_back_after_overshoot(self):
        """The loop grows once, overshoots, and settles on a measured in-budget count."""
        def measurements(settings, plan, step):
            budget = plan["budget"]["vram"]
            usage = {4: int(budget * actions.GROW_TARGET) - 2 * step - 1, 6: budget + 1,
                     5: budget - step // 2}
            return {"ram_peak": GIB, "vram_peak": usage[settings["num_gpu"]], "cpu_cores_avg": 1.0}

        with tempfile.TemporaryDirectory() as folder:
            plan, result, calls = self.tuning(folder, measurements)
            self.assertEqual([call["num_gpu"] for call in calls], [4, 6, 5])
            self.assertEqual(plan["settings"]["num_gpu"], 5)
            self.assertEqual(result["rounds"][-1]["num_gpu"], 5)
            self.assertTrue(result["verified"])
            with open(os.path.join(folder, "out", "Modelfile")) as handle:
                self.assertIn("PARAMETER num_gpu 5", handle.read())

    def test_tuning_leaves_headroom_instead_of_filling_the_last_megabytes(self):
        def measurements(settings, plan, step):
            return {"ram_peak": GIB, "vram_peak": plan["budget"]["vram"] - step - 1, "cpu_cores_avg": 1.0}

        with tempfile.TemporaryDirectory() as folder:
            plan, result, calls = self.tuning(folder, measurements)
            self.assertEqual(len(calls), 1)
            self.assertEqual(plan["settings"]["num_gpu"], 4)
            self.assertTrue(result["verified"])

    def test_tuning_shrinks_batch_then_context_when_ram_is_over(self):
        def measurements(settings, plan, step):
            over = settings["num_ctx"] > 2048
            return {"ram_peak": plan["budget"]["ram"] + (GIB if over else -GIB),
                    "vram_peak": plan["budget"]["vram"] - 1, "cpu_cores_avg": 1.0}

        with tempfile.TemporaryDirectory() as folder:
            plan, result, calls = self.tuning(folder, measurements)
            self.assertEqual([(c["num_batch"], c["num_ctx"]) for c in calls],
                             [(512, 4096), (128, 4096), (128, 2048)])
            self.assertTrue(result["verified"])
            self.assertEqual(plan["settings"]["num_ctx"], 2048)

    def test_tuning_does_not_fight_ram_it_cannot_win(self):
        """Weights larger than the RAM budget are the cage's job; shrinking context is pointless."""
        def measurements(settings, plan, step):
            return {"ram_peak": plan["budget"]["ram"] + GIB, "vram_peak": plan["budget"]["vram"] - 1,
                    "cpu_cores_avg": 1.0}

        with tempfile.TemporaryDirectory() as folder:
            plan, result, calls = self.tuning(folder, measurements, status="paged")
            self.assertEqual(len(calls), 1)
            self.assertEqual(plan["settings"]["num_ctx"], 4096)
            self.assertFalse(result["verified"])

    def test_tuning_drops_a_thread_when_cpu_is_over(self):
        def measurements(settings, plan, step):
            return {"ram_peak": GIB, "vram_peak": plan["budget"]["vram"] - 1,
                    "cpu_cores_avg": settings["num_thread"] + 1.5}

        with tempfile.TemporaryDirectory() as folder:
            plan, result, calls = self.tuning(folder, measurements)
            self.assertEqual([call["num_thread"] for call in calls], [3, 2])
            self.assertTrue(result["verified"])

    def test_the_installed_settings_are_always_the_measured_ones(self):
        """Even when the rounds run out, the last thing installed gets measured."""
        def measurements(settings, plan, step):
            over = settings["num_gpu"] > 1
            return {"ram_peak": GIB, "vram_peak": plan["budget"]["vram"] + (1 if over else -step),
                    "cpu_cores_avg": 1.0}

        with tempfile.TemporaryDirectory() as folder:
            plan, result, calls = self.tuning(folder, measurements, start_gpu=4, rounds=2)
            self.assertEqual([call["num_gpu"] for call in calls], [4, 3, 2])
            last = result["rounds"][-1]
            self.assertEqual(last["settings"], plan["settings"])
            self.assertIn(f"PARAMETER num_gpu {plan['settings']['num_gpu']}", last["installed"])
            self.assertFalse(result["verified"])      # still over, and it says so

    def test_estimates_follow_the_tuned_settings(self):
        def measurements(settings, plan, step):
            return {"ram_peak": GIB, "vram_peak": plan["budget"]["vram"] + step + 1, "cpu_cores_avg": 1.0} \
                if settings["num_gpu"] > 2 else \
                {"ram_peak": GIB, "vram_peak": plan["budget"]["vram"] - 1, "cpu_cores_avg": 1.0}

        with tempfile.TemporaryDirectory() as folder:
            plan, result, calls = self.tuning(folder, measurements)
            with open(os.path.join(folder, "out", "plan.json")) as handle:
                saved = json.load(handle)
            self.assertEqual(saved["layers"]["on_gpu"], plan["settings"]["num_gpu"])
            fresh = planner.estimate(sources.resolve(saved["model_file"]).model,
                                     plan["settings"]["num_gpu"], plan["settings"]["num_ctx"],
                                     plan["settings"]["num_batch"])
            self.assertEqual(saved["estimate"]["vram"], fresh["vram"])

    def test_other_ollama_models_are_not_unloaded_without_permission(self):
        unloaded = []
        loaded = [{"name": "someone-elses:latest"}, {"name": "mine:latest"}]
        with mock.patch.object(ollama, "available", lambda: True), \
                mock.patch.object(ollama, "running", lambda: loaded), \
                mock.patch.object(ollama, "unload", unloaded.append):
            with self.assertRaises(actions.ActionError) as caught:
                actions.clear_ollama("mine")
            self.assertIn("someone-elses:latest", str(caught.exception))
            self.assertEqual(unloaded, [])
            actions.clear_ollama("mine", unload_others=True)
            self.assertEqual(unloaded, ["someone-elses:latest", "mine:latest"])
            del unloaded[:]
            loaded.pop(0)
            actions.clear_ollama("mine")          # only the model under test: no permission needed
            self.assertEqual(unloaded, ["mine:latest"])


if __name__ == "__main__":
    unittest.main()
