import os
import struct
import tempfile
import unittest

from ptah import gguf
from tests import fakegguf


class GGUFReaderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def test_reads_shape_and_sizes(self):
        model = gguf.read(fakegguf.language_model(self.path("m.gguf"), layers=4, layer_bytes=4096))
        self.assertEqual(model.architecture, "llama")
        self.assertEqual(model.name, "Tiny Test")
        self.assertEqual(model.file_type, "Q4_K_M")
        self.assertEqual(model.layer_bytes, [4096] * 4)
        self.assertEqual(model.block_count, 4)
        self.assertEqual(model.input_bytes, 2048)
        self.assertEqual(model.output_bytes, 0)
        self.assertEqual(model.context_length, 8192)
        self.assertEqual(model.weights_bytes, 2048 + 4 * 4096 + 64)
        self.assertTrue(model.is_language_model)

    def test_kv_width_uses_kv_heads(self):
        model = gguf.read(fakegguf.language_model(self.path("m.gguf"), embd=64, heads=4, heads_kv=2))
        self.assertEqual(model.kv_width, (16 + 16) * 2)

    def test_large_arrays_are_skipped_not_stored(self):
        model = gguf.read(fakegguf.language_model(self.path("m.gguf"), tokens=600))
        self.assertEqual(model.metadata["tokenizer.ggml.tokens"], {"array_length": 600})

    def test_untied_output_layer(self):
        model = gguf.read(fakegguf.language_model(self.path("m.gguf"), tied=False))
        self.assertEqual(model.output_bytes, 1024)

    def test_projector_is_not_a_language_model(self):
        fakegguf.write(self.path("p.gguf"), {"general.architecture": "clip"},
                       [("v.blk.0.weight", [8, 8], 256), ("mm.0.weight", [8], 64)])
        model = gguf.read(self.path("p.gguf"))
        self.assertFalse(model.is_language_model)
        self.assertEqual(model.layer_bytes, [])

    def test_rejects_non_gguf(self):
        with open(self.path("x.gguf"), "wb") as handle:
            handle.write(b"NOPE" + b"\0" * 64)
        with self.assertRaises(gguf.GGUFError):
            gguf.read(self.path("x.gguf"))

    def test_rejects_truncated_header(self):
        source = fakegguf.language_model(self.path("m.gguf"))
        with open(source, "rb") as handle:
            data = handle.read(60)
        with open(self.path("t.gguf"), "wb") as handle:
            handle.write(data)
        with self.assertRaises(gguf.GGUFError):
            gguf.read(self.path("t.gguf"))

    def test_rejects_implausible_lengths(self):
        with open(self.path("h.gguf"), "wb") as handle:
            handle.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1 << 60))
        with self.assertRaises(gguf.GGUFError):
            gguf.read(self.path("h.gguf"))
        with open(self.path("c.gguf"), "wb") as handle:
            handle.write(b"GGUF" + struct.pack("<IQQ", 3, 1 << 40, 0))
        with self.assertRaises(gguf.GGUFError):
            gguf.read(self.path("c.gguf"))

    def test_rejects_tensor_outside_file(self):
        source = fakegguf.language_model(self.path("m.gguf"))
        size = os.path.getsize(source)
        with open(source, "rb") as handle:
            data = handle.read(size - 9000)
        with open(self.path("cut.gguf"), "wb") as handle:
            handle.write(data)
        with self.assertRaises(gguf.GGUFError):
            gguf.read(self.path("cut.gguf"))

    def test_split_model_merges_shards(self):
        meta = {
            "general.architecture": "llama", "llama.block_count": 4,
            "llama.context_length": 4096, "llama.embedding_length": 64,
            "llama.attention.head_count": 4, "split.count": 2, "split.no": 0,
        }
        first = self.path("big-00001-of-00002.gguf")
        second = self.path("big-00002-of-00002.gguf")
        fakegguf.write(first, meta, [("token_embd.weight", [64, 10], 1024),
                                     ("blk.0.w", [64], 2048), ("blk.1.w", [64], 2048)])
        fakegguf.write(second, {"split.count": 2, "split.no": 1},
                       [("blk.2.w", [64], 2048), ("blk.3.w", [64], 2048)])
        for start in (first, second):
            model = gguf.read(start)
            self.assertEqual(model.layer_bytes, [2048] * 4)
            self.assertEqual(model.shards, [first, second])
            self.assertEqual(model.path, first)
            self.assertEqual(model.file_size, os.path.getsize(first) + os.path.getsize(second))

    def test_split_model_missing_shard(self):
        first = self.path("big-00001-of-00002.gguf")
        fakegguf.write(first, {"general.architecture": "llama", "split.count": 2},
                       [("blk.0.w", [64], 2048)])
        with self.assertRaises(gguf.GGUFError):
            gguf.read(first)


if __name__ == "__main__":
    unittest.main()
