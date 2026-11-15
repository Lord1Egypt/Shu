"""Build small synthetic GGUF files for the tests."""

import struct

_TYPES = {"u32": 4, "u16": 2, "str": 8}


def _string(text):
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def _value(value):
    if isinstance(value, str):
        return struct.pack("<I", 8) + _string(value)
    if isinstance(value, list):
        if value and isinstance(value[0], str):
            return struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_string(v) for v in value)
        return struct.pack("<IIQ", 9, 4, len(value)) + b"".join(struct.pack("<I", v) for v in value)
    return struct.pack("<I", 4) + struct.pack("<I", value)


def write(path, metadata, tensors, alignment=32):
    """Write a GGUF v3 file. ``tensors`` is a list of (name, dims, byte_size).

    Tensor data is left as a hole in the file, so even a 40 GB model takes no disk space.
    """
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += _string(key) + _value(value)
    offset = 0
    for name, dims, size in tensors:
        header += _string(name) + struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<Q", d) for d in dims)
        header += struct.pack("<IQ", 0, offset)
        offset += (size + alignment - 1) // alignment * alignment
    header += b"\0" * (-len(header) % alignment)
    with open(path, "wb") as handle:
        handle.write(header)
        handle.truncate(len(header) + offset)


def transformer(path, name, layers, layer_bytes, embd, heads, heads_kv, vocab, context,
                input_bytes, output_bytes, file_type=15):
    """A header-only stand-in with the shape and sizes of a real large model."""
    metadata = {
        "general.architecture": "llama", "general.name": name, "general.file_type": file_type,
        "llama.block_count": layers, "llama.context_length": context,
        "llama.embedding_length": embd, "llama.attention.head_count": heads,
        "llama.attention.head_count_kv": heads_kv,
    }
    per_layer = 12 * embd * embd
    tensors = [("token_embd.weight", [embd, vocab], input_bytes)]
    tensors += [(f"blk.{index}.weight", [per_layer], layer_bytes) for index in range(layers)]
    tensors.append(("output.weight", [embd, vocab], output_bytes))
    write(path, metadata, tensors)
    return path


def language_model(path, layers=4, layer_bytes=4096, embd=64, heads=4, heads_kv=2,
                   context=8192, extra=None, tied=True, tokens=600):
    """A tiny model shaped like a transformer: embeddings, N blocks, optional output."""
    metadata = {
        "general.architecture": "llama",
        "general.name": "Tiny Test",
        "general.file_type": 15,
        "llama.block_count": layers,
        "llama.context_length": context,
        "llama.embedding_length": embd,
        "llama.attention.head_count": heads,
        "llama.attention.head_count_kv": heads_kv,
        "tokenizer.ggml.tokens": [f"t{i}" for i in range(tokens)],
    }
    metadata.update(extra or {})
    tensors = [("token_embd.weight", [embd, 100], 2048)]
    for index in range(layers):
        tensors.append((f"blk.{index}.attn_q.weight", [embd, embd], layer_bytes // 2))
        tensors.append((f"blk.{index}.ffn_up.weight", [embd, embd], layer_bytes // 2))
    if not tied:
        tensors.append(("output.weight", [embd, 100], 1024))
    tensors.append(("output_norm.weight", [embd], 64))
    write(path, metadata, tensors)
    return path
