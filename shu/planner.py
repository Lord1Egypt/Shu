"""Turn a resource budget and a GGUF model into concrete runtime settings.

The memory model follows llama.cpp. The constants below were calibrated against
measurements of llama.cpp's server on an RTX 2060; ``shu apply --verify``
corrects any remaining error by measuring the real load.
"""

from .hardware import GIB, MIB, fmt_bytes

CUDA_OVERHEAD = 200 * MIB        # CUDA context and libraries, present whenever a GPU is used
PROCESS_BASE = 300 * MIB         # server process, tokenizer, scratch
TYPICAL_WORKING = 1200 * MIB     # process, KV cache and buffers of a mid-sized model at context 4096
GPU_COMPUTE_FACTOR = 19          # bytes of GPU compute buffer per batch token per embedding float
CPU_COMPUTE_FACTOR = 45
KV_BYTES = 2                     # f16 KV cache
MIN_CONTEXT = 1024
DEFAULT_CONTEXT = 4096
DEFAULT_BATCH = 512
ASSUMED_DISK_SPEED = 200 * MIB   # bytes per second; only used when the disk cannot be probed

# Quantization ladder that llama-quantize can produce without an importance matrix.
QUANT_LADDER = (
    ("Q8_0", 8.50), ("Q6_K", 6.56), ("Q5_K_M", 5.69), ("Q4_K_M", 4.85),
    ("Q4_K_S", 4.58), ("Q3_K_M", 3.91), ("Q3_K_S", 3.50), ("Q2_K", 3.00),
)

STATUS_TEXT = {
    "gpu": "FITS: the whole model runs on the GPU inside the budget",
    "split": "FITS: the model is split between GPU and CPU inside the budget",
    "cpu": "FITS: the model runs on the CPU inside the budget",
    "paged": "OVERSIZED: the weights exceed the RAM budget and will be paged from disk",
    "impossible": "IMPOSSIBLE: even the working memory does not fit this budget",
}


def make_budget(hw, percent=30.0, ram_gb=None, vram_gb=None, cpu=None, ctx=None,
                batch=None, max_temp=None):
    """Build a budget from a percentage of the machine, with optional absolute overrides."""
    if not 1 <= percent <= 100:
        raise ValueError("percent must be between 1 and 100")
    share = percent / 100
    logical = hw["logical_cpus"]
    threads = cpu if cpu is not None else max(1, int(logical * share))
    if threads < 1:
        raise ValueError("cpu must be at least 1 thread")
    threads = min(threads, logical)
    ram = int(ram_gb * GIB) if ram_gb is not None else int(hw["ram_total"] * share)
    vram = int(vram_gb * GIB) if vram_gb is not None else int(hw["vram_total"] * share)
    if ram <= 0:
        raise ValueError("the RAM budget must be positive")
    if vram < 0 or (ctx is not None and ctx < 256) or (batch is not None and batch < 8):
        raise ValueError("vram, ctx (>= 256) and batch (>= 8) must be sensible positive values")
    return {
        "percent": percent,
        "ram": ram,
        "vram": min(vram, hw["vram_total"]),
        "threads": threads,
        # llama.cpp keeps one extra thread busy feeding the GPU, hence the extra core.
        "cpu_quota": (threads + 1) * 100 if cpu is not None else max(100, round(logical * share * 100)),
        "ctx": ctx,
        "batch": batch,
        "max_temp": max_temp,
    }


def _kv_layers(model):
    """Layers that own a KV cache; hybrid models attend fully only every Nth layer."""
    layers = len(model.layer_bytes)
    interval = model.metadata.get(f"{model.architecture}.full_attention_interval")
    if isinstance(interval, int) and interval > 1:
        return max(1, layers // interval)
    return layers


def estimate(model, gpu_layers, ctx, batch, projector_bytes=0):
    """Memory use for ``gpu_layers`` in llama.cpp terms (the output layer counts as one)."""
    layers = model.layer_bytes
    total = len(layers)
    gpu_layers = max(0, min(gpu_layers, total + 1))
    repeating = max(0, gpu_layers - 1)
    on_gpu = sum(layers[total - repeating:]) if repeating else 0
    # Models with tied embeddings duplicate the embedding table onto the output device.
    output = model.output_bytes or model.input_bytes
    kv_total = ctx * model.kv_width * KV_BYTES * _kv_layers(model)
    kv_gpu = kv_total * repeating // total if total else 0
    width = model.embedding_length

    vram = 0
    if gpu_layers:
        vram = CUDA_OVERHEAD + on_gpu + output + kv_gpu + batch * width * 4 * GPU_COMPUTE_FACTOR
        if gpu_layers == total + 1:
            vram += projector_bytes
    cpu_weights = model.weights_bytes - on_gpu - (model.output_bytes if gpu_layers else 0)
    # Layers are read in full for every token; embedding tables are sparse lookups.
    cpu_hot = sum(layers[:total - repeating]) + (0 if gpu_layers else output)
    working = (PROCESS_BASE + projector_bytes + (kv_total - kv_gpu)
               + batch * width * 4 * CPU_COMPUTE_FACTOR)
    return {
        "vram": vram,
        "ram_working": working,
        "cpu_weights": cpu_weights,
        "cpu_hot": cpu_hot,
        "ram_resident": working + cpu_weights,
        "kv_total": kv_total,
    }


def largest_fitting_parameters(budget):
    """Parameter count of the largest Q4_K_M model this budget holds entirely in memory."""
    usable = budget["ram"] - TYPICAL_WORKING
    if budget["vram"] > CUDA_OVERHEAD:
        usable += budget["vram"] - CUDA_OVERHEAD
    return max(0, int(usable * 8 / 4.85))


def _recommend(model, budget, working, kv_total):
    """Suggest a smaller quantization, or say how large a model this budget can hold."""
    usable = budget["ram"] - working
    if budget["vram"] > CUDA_OVERHEAD:
        usable += budget["vram"] - CUDA_OVERHEAD
    usable = max(0, usable - kv_total)
    if not model.parameters:
        return None
    fit_bpw = usable * 8 / model.parameters
    result = {
        "bits_per_weight_that_fit": round(fit_bpw, 2),
        "largest_q4_model_params": largest_fitting_parameters(budget),
    }
    current = model.bits_per_weight
    for name, bpw in QUANT_LADDER:
        if bpw <= fit_bpw and bpw < current - 0.2:
            result["quantization"] = name
            result["estimated_size"] = int(model.parameters * bpw / 8)
            break
    return result


def refresh(plan, model, projector_bytes=0):
    """Recompute a plan's memory estimate after its settings were changed by tuning."""
    settings = plan["settings"]
    numbers = estimate(model, settings["num_gpu"], settings["num_ctx"], settings["num_batch"],
                       projector_bytes)
    plan["estimate"].update(vram=numbers["vram"], ram_working=numbers["ram_working"],
                            ram_weights=numbers["cpu_weights"], ram_resident=numbers["ram_resident"])
    plan["layers"]["on_gpu"] = settings["num_gpu"]
    return plan


def plan(model, budget, projector_bytes=0, disk_speed=None):
    """Return the settings, the memory estimate and a verdict for one model and budget.

    ``disk_speed`` is the measured read rate of the model's disk in bytes per second; without
    it the paging estimate falls back to an assumption and says so.
    """
    if not model.is_language_model:
        raise ValueError(f"{model.path} holds no language-model layers (it may be a projector file)")
    total = len(model.layer_bytes)
    ctx_wanted = budget["ctx"] or min(DEFAULT_CONTEXT, model.context_length)
    ctx_wanted = min(ctx_wanted, model.context_length)
    batch_wanted = budget["batch"] or DEFAULT_BATCH

    candidates = [(ctx_wanted, batch_wanted)]
    ctx, batch = ctx_wanted, min(batch_wanted, 128)
    while True:
        if (ctx, batch) not in candidates:
            candidates.append((ctx, batch))
        if ctx <= MIN_CONTEXT:
            break
        ctx = max(MIN_CONTEXT, ctx // 2)

    chosen = None
    for ctx, batch in candidates:
        batch = min(batch, ctx)
        if estimate(model, 0, ctx, batch, projector_bytes)["ram_working"] > budget["ram"]:
            continue
        gpu_layers = 0
        for layers in range(total + 1, 0, -1):
            if estimate(model, layers, ctx, batch, projector_bytes)["vram"] <= budget["vram"]:
                gpu_layers = layers
                break
        chosen = (ctx, batch, gpu_layers)
        break

    notes = []
    if chosen is None:
        ctx, batch = candidates[-1]
        batch = min(batch, ctx)
        numbers = estimate(model, 0, ctx, batch, projector_bytes)
        status, gpu_layers = "impossible", 0
        notes.append(
            f"Working memory alone needs {fmt_bytes(numbers['ram_working'])} at context {ctx}; "
            f"the RAM budget is {fmt_bytes(budget['ram'])}."
        )
    else:
        ctx, batch, gpu_layers = chosen
        numbers = estimate(model, gpu_layers, ctx, batch, projector_bytes)
        if gpu_layers == total + 1:
            status = "gpu"
        elif numbers["ram_resident"] <= budget["ram"]:
            status = "split" if gpu_layers else "cpu"
        else:
            status = "paged"
        if ctx < ctx_wanted:
            notes.append(f"Context reduced from {ctx_wanted} to {ctx} so the working memory fits.")
        if batch < batch_wanted:
            notes.append(f"Batch reduced from {batch_wanted} to {batch} to shrink compute buffers.")

    # Layers are swept in order for every token. Once they no longer fit beside the working
    # memory, the page cache stops helping: each token re-reads all of them (measured: a 5 GiB
    # model with 2.2 GiB of CPU-side layers in a 2 GiB cage read 2.17 GiB per token).
    paged = 0
    if status == "paged" and numbers["ram_working"] + numbers["cpu_hot"] > budget["ram"]:
        paged = numbers["cpu_hot"]
    result = {
        "status": status,
        "verdict": STATUS_TEXT[status],
        "settings": {
            "num_gpu": gpu_layers,
            "num_thread": budget["threads"],
            "num_ctx": ctx,
            "num_batch": batch,
            "use_mmap": True,
        },
        "estimate": {
            "vram": numbers["vram"],
            "ram_working": numbers["ram_working"],
            "ram_weights": numbers["cpu_weights"],
            "ram_resident": numbers["ram_resident"],
            "paged_bytes": paged,
        },
        "layers": {"total": total + 1, "on_gpu": gpu_layers},
        "budget": budget,
        "notes": notes,
    }
    if status == "paged" and not paged:
        notes.append(
            "Only the embedding tables exceed the RAM budget. They are read a few rows at a "
            "time, so the slowdown is small."
        )
        notes.append("Only a caged engine (`shu serve`) can hold RAM to the budget in this mode.")
    elif status == "paged":
        speed = disk_speed or ASSUMED_DISK_SPEED
        seconds = paged / speed
        basis = (f"this disk just measured {speed / MIB:.0f} MiB/s" if disk_speed
                 else f"assuming a {ASSUMED_DISK_SPEED // MIB} MiB/s disk, not measured")
        result["estimate"]["disk_speed"] = int(speed)
        result["estimate"]["disk_speed_measured"] = bool(disk_speed)
        result["estimate"]["paged_tokens_per_second"] = round(1 / seconds, 4)
        pace = f"{seconds:.0f} s per token" if seconds >= 1.5 else f"{1 / seconds:.1f} tokens/s"
        notes.append(
            f"{fmt_bytes(paged)} of layers do not fit in the RAM budget and are re-read from disk "
            f"for every token: about {pace} ({basis}). Disk caches make this vary; "
            f"`shu bench` measures the real figure."
        )
        notes.append("Only a caged engine (`shu serve`) can hold RAM to the budget in this mode.")
    if status in ("paged", "impossible"):
        advice = _recommend(model, budget, numbers["ram_working"], numbers["kv_total"])
        if advice:
            result["recommendation"] = advice
            if "quantization" in advice:
                notes.append(
                    f"Requantizing to {advice['quantization']} (about "
                    f"{fmt_bytes(advice['estimated_size'])}) would fit entirely in the budget."
                )
            else:
                notes.append(
                    f"No quantization of this model fits the budget entirely (it would need "
                    f"{advice['bits_per_weight_that_fit']} bits per weight; the smallest usable "
                    f"quantizations are near 3). The largest model this budget holds entirely in "
                    f"memory at Q4_K_M has about {advice['largest_q4_model_params'] / 1e9:.0f}B parameters."
                )
    return result
