import os
import tempfile
import unittest

from ptah import gguf, planner
from ptah.hardware import GIB, MIB
from tests import fakegguf

HW = {"logical_cpus": 12, "physical_cores": 6, "ram_total": 16 * GIB, "vram_total": 6 * GIB}


class Shape:
    """A model description with realistic sizes, without writing gigabytes to disk."""

    def __init__(self, layers=32, layer_bytes=64 * MIB, input_bytes=500 * MIB, output_bytes=0,
                 embd=2560, kv_width=2048, context=32768, interval=None, parameters=4_000_000_000):
        self.path = "shape.gguf"
        self.architecture = "test"
        self.layer_bytes = [layer_bytes] * layers
        self.input_bytes = input_bytes
        self.output_bytes = output_bytes
        self.weights_bytes = layers * layer_bytes + input_bytes + output_bytes
        self.embedding_length = embd
        self.kv_width = kv_width
        self.context_length = context
        self.parameters = parameters
        self.metadata = {"test.full_attention_interval": interval} if interval else {}
        self.is_language_model = True

    @property
    def bits_per_weight(self):
        return self.weights_bytes * 8 / self.parameters


class BudgetTest(unittest.TestCase):
    def test_percent_budget(self):
        budget = planner.make_budget(HW, 30)
        self.assertEqual(budget["ram"], int(16 * GIB * 0.3))
        self.assertEqual(budget["vram"], int(6 * GIB * 0.3))
        self.assertEqual(budget["threads"], 3)
        self.assertEqual(budget["cpu_quota"], 360)

    def test_overrides(self):
        budget = planner.make_budget(HW, 30, ram_gb=4, vram_gb=2, cpu=2, ctx=2048, max_temp=75)
        self.assertEqual((budget["ram"], budget["vram"], budget["threads"]), (4 * GIB, 2 * GIB, 2))
        self.assertEqual(budget["cpu_quota"], 300)
        self.assertEqual((budget["ctx"], budget["max_temp"]), (2048, 75))

    def test_vram_is_capped_at_the_card(self):
        self.assertEqual(planner.make_budget(HW, 30, vram_gb=64)["vram"], 6 * GIB)

    def test_rejects_nonsense(self):
        for kwargs in ({"percent": 0}, {"percent": 101}, {"ram_gb": 0}, {"cpu": 0}, {"ctx": 10},
                       {"vram_gb": -1}):
            with self.assertRaises(ValueError):
                planner.make_budget(HW, **{"percent": 30, **kwargs})

    def test_no_gpu(self):
        budget = planner.make_budget(dict(HW, vram_total=0), 30)
        self.assertEqual(budget["vram"], 0)
        plan = planner.plan(Shape(), budget)
        self.assertEqual(plan["settings"]["num_gpu"], 0)
        self.assertEqual(plan["estimate"]["vram"], 0)


class PlanTest(unittest.TestCase):
    def test_never_plans_past_the_vram_budget(self):
        for percent in (10, 25, 30, 35, 60, 100):
            budget = planner.make_budget(HW, percent)
            plan = planner.plan(Shape(), budget)
            self.assertLessEqual(plan["estimate"]["vram"], budget["vram"])

    def test_one_more_layer_would_not_fit(self):
        shape = Shape()
        budget = planner.make_budget(HW, 30)
        plan = planner.plan(shape, budget)
        layers = plan["settings"]["num_gpu"]
        self.assertGreater(layers, 0)
        self.assertLess(layers, len(shape.layer_bytes) + 1)
        more = planner.estimate(shape, layers + 1, plan["settings"]["num_ctx"],
                                plan["settings"]["num_batch"])
        self.assertGreater(more["vram"], budget["vram"])

    def test_small_model_goes_fully_to_gpu(self):
        shape = Shape(layers=12, layer_bytes=20 * MIB, input_bytes=60 * MIB, embd=768, kv_width=512)
        plan = planner.plan(shape, planner.make_budget(HW, 30))
        self.assertEqual(plan["status"], "gpu")
        self.assertEqual(plan["settings"]["num_gpu"], 13)

    def test_split_when_weights_fit_in_ram(self):
        plan = planner.plan(Shape(), planner.make_budget(HW, 30))
        self.assertEqual(plan["status"], "split")
        self.assertLessEqual(plan["estimate"]["ram_resident"], plan["budget"]["ram"])

    def test_cpu_only_without_vram(self):
        plan = planner.plan(Shape(), planner.make_budget(HW, 30, vram_gb=0))
        self.assertEqual(plan["status"], "cpu")
        self.assertEqual(plan["settings"]["num_gpu"], 0)

    def test_oversized_model_is_paged_and_says_so(self):
        big = Shape(layers=80, layer_bytes=500 * MIB, input_bytes=1 * GIB, embd=8192,
                    kv_width=2048, parameters=70_000_000_000)
        plan = planner.plan(big, planner.make_budget(HW, 30))
        self.assertEqual(plan["status"], "paged")
        self.assertGreater(plan["estimate"]["paged_bytes"], 30 * GIB)
        self.assertLess(plan["estimate"]["paged_tokens_per_second"], 0.1)
        self.assertIn("recommendation", plan)
        self.assertNotIn("quantization", plan["recommendation"])
        self.assertTrue(any("s per token" in note for note in plan["notes"]))
        self.assertTrue(any("ptah serve" in note for note in plan["notes"]))

    def test_recommends_a_quantization_when_one_fits(self):
        model = Shape(layers=32, layer_bytes=200 * MIB, input_bytes=300 * MIB,
                      parameters=7_000_000_000)
        plan = planner.plan(model, planner.make_budget(HW, 30))
        self.assertEqual(plan["status"], "paged")
        self.assertIn(plan["recommendation"]["quantization"], dict(planner.QUANT_LADDER))

    def test_context_shrinks_before_giving_up(self):
        model = Shape(kv_width=16384, context=131072)
        budget = planner.make_budget(HW, 30, ram_gb=3, vram_gb=0, ctx=32768)
        plan = planner.plan(model, budget)
        self.assertLess(plan["settings"]["num_ctx"], 32768)
        self.assertGreaterEqual(plan["settings"]["num_ctx"], planner.MIN_CONTEXT)
        self.assertLessEqual(plan["estimate"]["ram_working"], budget["ram"])

    def test_impossible_when_working_memory_cannot_fit(self):
        plan = planner.plan(Shape(), planner.make_budget(HW, 30, ram_gb=0.1, vram_gb=0))
        self.assertEqual(plan["status"], "impossible")

    def test_context_never_exceeds_training_context(self):
        plan = planner.plan(Shape(context=2048), planner.make_budget(HW, 30, ctx=8192))
        self.assertEqual(plan["settings"]["num_ctx"], 2048)

    def test_hybrid_models_have_fewer_kv_layers(self):
        plain = planner.estimate(Shape(), 0, 4096, 512)["kv_total"]
        hybrid = planner.estimate(Shape(interval=4), 0, 4096, 512)["kv_total"]
        self.assertEqual(hybrid * 4, plain)

    def test_projector_counts_against_ram(self):
        budget = planner.make_budget(HW, 30)
        without = planner.plan(Shape(), budget)["estimate"]["ram_working"]
        with_projector = planner.plan(Shape(), budget, 600 * MIB)["estimate"]["ram_working"]
        self.assertEqual(with_projector - without, 600 * MIB)

    def test_matches_measured_run(self):
        """qwen3.5 4B, 10 GPU layers, context 4096: llama.cpp used 1326 MiB of VRAM."""
        shape = Shape(layers=32, layer_bytes=66_129_472, input_bytes=521_472_000, interval=4)
        vram = planner.estimate(shape, 10, 4096, 512)["vram"] / MIB
        self.assertGreater(vram, 1326)
        self.assertLess(vram, 1326 * 1.10)

    def test_rejects_projector_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "p.gguf")
            fakegguf.write(path, {"general.architecture": "clip"}, [("v.blk.0.w", [8], 64)])
            with self.assertRaises(ValueError):
                planner.plan(gguf.read(path), planner.make_budget(HW, 30))

    def test_plans_a_70b_shaped_file(self):
        """Llama 3.3 70B at Q4_K_M is about 42 GB: far past 30% of a 16 GB / 6 GB machine."""
        with tempfile.TemporaryDirectory() as folder:
            path = fakegguf.transformer(
                os.path.join(folder, "70b.gguf"), "Seventy", layers=80, layer_bytes=515 * MIB,
                embd=8192, heads=64, heads_kv=8, vocab=128256, context=131072,
                input_bytes=600 * MIB, output_bytes=860 * MIB)
            self.assertLess(os.stat(path).st_blocks * 512, 16 * MIB)
            model = gguf.read(path)
            self.assertAlmostEqual(model.file_size / GIB, 41.7, delta=0.5)
            self.assertAlmostEqual(model.parameters / 1e9, 66.5, delta=3)
            plan = planner.plan(model, planner.make_budget(HW, 30))
            self.assertEqual(plan["status"], "paged")
            self.assertLessEqual(plan["estimate"]["vram"], plan["budget"]["vram"])
            self.assertLessEqual(plan["estimate"]["ram_working"], plan["budget"]["ram"])
            self.assertGreater(plan["settings"]["num_gpu"], 0)

    def test_plans_a_real_file(self):
        with tempfile.TemporaryDirectory() as folder:
            model = gguf.read(fakegguf.language_model(os.path.join(folder, "m.gguf")))
            plan = planner.plan(model, planner.make_budget(HW, 30))
            self.assertEqual(plan["status"], "gpu")


if __name__ == "__main__":
    unittest.main()
