"""Read-only GGUF header parser (versions 2 and 3), standard library only.

Only the header is read: metadata and tensor descriptors. Tensor sizes are
derived from tensor offsets, so unknown quantization types still size correctly.
A GGUF file is untrusted input, so every length is bounded before it is used.
"""

import os
import re
import struct

MAGIC = b"GGUF"
MAX_STRING = 64 * 1024 * 1024
MAX_COUNT = 50_000_000
MAX_DIMS = 8
KEEP_ARRAY_ITEMS = 64

_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_STRING, _ARRAY = 8, 9

FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}

_BLOCK = re.compile(r"^blk\.(\d+)\.")
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
MAX_SHARDS = 999


class GGUFError(ValueError):
    pass


class _Reader:
    def __init__(self, handle):
        self.handle = handle

    def take(self, size):
        data = self.handle.read(size)
        if len(data) != size:
            raise GGUFError("file ends inside the GGUF header")
        return data

    def scalar(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def count(self):
        value = self.scalar("<Q")
        if value > MAX_COUNT:
            raise GGUFError(f"implausible element count {value}")
        return value

    def string(self):
        length = self.scalar("<Q")
        if length > MAX_STRING:
            raise GGUFError(f"implausible string length {length}")
        return self.take(length).decode("utf-8", "replace")

    def value(self, kind):
        if kind in _SCALARS:
            return self.scalar(_SCALARS[kind])
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            item_kind = self.scalar("<I")
            if item_kind == _ARRAY:
                raise GGUFError("nested arrays are not supported")
            length = self.count()
            if item_kind in _SCALARS and length > KEEP_ARRAY_ITEMS:
                self.handle.seek(struct.calcsize(_SCALARS[item_kind]) * length, os.SEEK_CUR)
                return {"array_length": length}
            items = []
            for index in range(length):
                item = self.value(item_kind)
                if index < KEEP_ARRAY_ITEMS:
                    items.append(item)
            return items if length <= KEEP_ARRAY_ITEMS else {"array_length": length}
        raise GGUFError(f"unknown metadata value type {kind}")


class Model:
    """Sizes and shape of one GGUF file, as needed for memory planning."""

    def __init__(self, path, version, metadata, tensors, file_size):
        self.path = path
        self.version = version
        self.metadata = metadata
        self.tensors = tensors
        self.file_size = file_size
        self.shards = [path]
        self.architecture = str(metadata.get("general.architecture", "unknown"))
        self.name = str(metadata.get("general.name", "") or os.path.basename(path))

        blocks = {}
        other = 0
        self.output_bytes = 0
        self.input_bytes = 0
        for tensor in tensors:
            match = _BLOCK.match(tensor["name"])
            if match:
                index = int(match.group(1))
                blocks[index] = blocks.get(index, 0) + tensor["size"]
            elif tensor["name"].startswith("output."):
                self.output_bytes += tensor["size"]
            elif tensor["name"].startswith("token_embd."):
                self.input_bytes += tensor["size"]
            else:
                other += tensor["size"]
        self.other_bytes = other
        self.layer_bytes = [blocks[i] for i in sorted(blocks)]
        self.weights_bytes = sum(t["size"] for t in tensors)
        self.parameters = sum(t["elements"] for t in tensors)

    def _arch(self, key, default=None):
        return self.metadata.get(f"{self.architecture}.{key}", default)

    @staticmethod
    def _as_int(value, default):
        if isinstance(value, list) and value:
            value = max(v for v in value if isinstance(v, int))
        return value if isinstance(value, int) and value > 0 else default

    @property
    def block_count(self):
        return self._as_int(self._arch("block_count"), len(self.layer_bytes)) or 0

    @property
    def context_length(self):
        return self._as_int(self._arch("context_length"), 2048)

    @property
    def embedding_length(self):
        return self._as_int(self._arch("embedding_length"), 0)

    @property
    def head_count(self):
        return self._as_int(self._arch("attention.head_count"), 0)

    @property
    def head_count_kv(self):
        return self._as_int(self._arch("attention.head_count_kv"), self.head_count)

    @property
    def kv_width(self):
        """Elements stored per token and layer in the KV cache (keys plus values)."""
        heads = self.head_count
        if not heads or not self.embedding_length:
            return 0
        head_dim = self.embedding_length // heads
        key = self._as_int(self._arch("attention.key_length"), head_dim)
        value = self._as_int(self._arch("attention.value_length"), head_dim)
        return (key + value) * (self.head_count_kv or heads)

    @property
    def file_type(self):
        code = self.metadata.get("general.file_type")
        return FILE_TYPES.get(code, f"type-{code}" if code is not None else "unknown")

    @property
    def bits_per_weight(self):
        return self.weights_bytes * 8 / self.parameters if self.parameters else 0.0

    @property
    def is_language_model(self):
        return bool(self.layer_bytes) and self.kv_width > 0

    def summary(self):
        return {
            "path": self.path,
            "name": self.name,
            "architecture": self.architecture,
            "gguf_version": self.version,
            "file_size": self.file_size,
            "weights_bytes": self.weights_bytes,
            "parameters": self.parameters,
            "quantization": self.file_type,
            "bits_per_weight": round(self.bits_per_weight, 3),
            "layers": len(self.layer_bytes),
            "block_count": self.block_count,
            "context_length": self.context_length,
            "embedding_length": self.embedding_length,
            "head_count": self.head_count,
            "head_count_kv": self.head_count_kv,
            "layer_bytes_avg": int(sum(self.layer_bytes) / len(self.layer_bytes)) if self.layer_bytes else 0,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "other_bytes": self.other_bytes,
            "tensor_count": len(self.tensors),
        }


def _read_file(path):
    """Parse one GGUF file; return (version, metadata, tensors, file_size)."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as handle:
        reader = _Reader(handle)
        if reader.take(4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file")
        version = reader.scalar("<I")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = reader.count()
        metadata_count = reader.count()

        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            metadata[key] = reader.value(reader.scalar("<I"))

        tensors = []
        for _ in range(tensor_count):
            name = reader.string()
            n_dims = reader.scalar("<I")
            if n_dims > MAX_DIMS:
                raise GGUFError(f"tensor {name!r} has {n_dims} dimensions")
            elements = 1
            for _ in range(n_dims):
                elements *= reader.scalar("<Q")
            kind = reader.scalar("<I")
            offset = reader.scalar("<Q")
            tensors.append({"name": name, "elements": elements, "type": kind, "offset": offset})

        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment <= 0:
            raise GGUFError("invalid general.alignment")
        header_end = handle.tell()
        data_start = (header_end + alignment - 1) // alignment * alignment

    data_size = file_size - data_start
    ordered = sorted(tensors, key=lambda t: t["offset"])
    for tensor, following in zip(ordered, ordered[1:] + [None]):
        end = following["offset"] if following else data_size
        tensor["size"] = end - tensor["offset"]
        if tensor["size"] < 0 or end > data_size:
            raise GGUFError(f"tensor {tensor['name']!r} lies outside the file")
    return version, metadata, tensors, file_size


def shard_paths(path, count):
    """Paths of every shard of a split GGUF, given any one of them."""
    match = _SHARD.search(os.path.basename(path))
    if not match or int(match.group(2)) != count:
        raise GGUFError(f"{path} is part of a split model but is not named *-00001-of-{count:05d}.gguf")
    folder, base = os.path.dirname(path), os.path.basename(path)
    return [os.path.join(folder, base[:match.start()] + f"-{index:05d}-of-{count:05d}.gguf")
            for index in range(1, count + 1)]


def read(path):
    """Parse the header of the GGUF at ``path`` (all shards of a split model) into a :class:`Model`."""
    version, metadata, tensors, file_size = _read_file(path)
    count = metadata.get("split.count", 1)
    shards = [path]
    if isinstance(count, int) and count > 1:
        if count > MAX_SHARDS:
            raise GGUFError(f"implausible shard count {count}")
        shards = shard_paths(path, count)
        if os.path.abspath(path) != os.path.abspath(shards[0]):
            version, metadata, tensors, file_size = _read_file(shards[0])
        for shard in shards[1:]:
            if not os.path.isfile(shard):
                raise GGUFError(f"shard {shard} of the split model is missing")
            _, _, more, size = _read_file(shard)
            tensors += more
            file_size += size
    model = Model(shards[0], version, metadata, tensors, file_size)
    model.shards = shards
    return model
