"""The work behind each command: measuring, creating Ollama models, converting."""

import json
import math
import os
import shlex
import subprocess
import time

from . import engine, hardware, hfconvert, monitor, ollama, planner, sources
from .hardware import fmt_bytes

DEFAULT_PROMPT = "Explain in a few sentences why the Nile was important to ancient Egypt."
SPLIT_ADVICE = ("Ollama cannot import a split GGUF. Run it with `shu serve`, or merge the shards "
                "first with llama.cpp's `llama-gguf-split --merge`.")
UNQUANTIZED = ("F32", "F16", "BF16")
WARMUP_TOKENS = 8            # generated before timing, so first-touch disk reads are not counted as speed
MAX_TUNE_ROUNDS = 4
GROW_TARGET = 0.95


class ActionError(RuntimeError):
    pass


def evict_from_page_cache(path):
    """Drop a file's cached pages so a cage is charged for every page it reads."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def prepare_cage(source, projector=None):
    """Drop the model's files from the page cache so the cage is charged for all of them.

    Pages another cgroup already cached would otherwise ride along uncounted and uncapped.
    The price is a cold load.
    """
    for path in list(source.model.shards) + ([projector] if projector else []):
        evict_from_page_cache(path)


def _speed(count, nanoseconds):
    return round(count / (nanoseconds / 1e9), 2) if count and nanoseconds else None


def _tagged(name):
    return name if ":" in name.split("/")[-1] else name + ":latest"


def clear_ollama(target=None, unload_others=False):
    """Unload the model about to be measured; other loaded models are the user's to keep.

    Another loaded model shares the GPU, and Ollama may evict it mid-run, which would make
    the measured VRAM too low. So their presence is an error unless unloading was authorized.
    """
    if not ollama.available():
        return
    loaded = [entry["name"] for entry in ollama.running()]
    others = [name for name in loaded if target is None or _tagged(name) != _tagged(target)]
    if others and not unload_others:
        raise ActionError(
            f"Ollama has other models loaded ({', '.join(others)}). They share the GPU and would "
            "distort the measurement. Pass --unload-others to unload them, or wait until Ollama does."
        )
    for name in loaded:
        ollama.unload(name)


def _finish(result, measured, budget):
    result["measured"] = measured
    result["checks"] = monitor.judge(measured, budget, gpu_expected=bool(hardware.gpus()),
                                     caged=result["limits"]["ram_hard"])
    result["exit_code"] = monitor.verdict(result["checks"])
    tokens, read = result.get("tokens"), measured.get("disk_read_bytes")
    result["disk_read_per_token"] = int(read / tokens) if read is not None and tokens else None
    return result


def bench_ollama(name, budget, tokens=64, prompt=DEFAULT_PROMPT, keep=False, unload_others=False):
    """Load an Ollama model, generate, and measure what the run really used."""
    source = sources.resolve(name)
    clear_ollama(name, unload_others)
    baseline = monitor.settle_vram()
    path = source.model.path
    sampler = monitor.Sampler(lambda: monitor.find_runner(path))
    sampler.start()
    generation = None
    try:
        warmup = ollama.generate(name, prompt, WARMUP_TOKENS)
        generation = time.monotonic()
        response = ollama.generate(name, prompt, tokens)
    finally:
        sampler.stop()
    measured = sampler.summary(baseline, since=generation)
    if not keep:
        ollama.unload(name)
    return _finish({
        "engine": "ollama",
        "model": name,
        "limits": {"caged": False, "ram_hard": False, "cpu_hard": False},
        "load_seconds": round(warmup.get("load_duration", 0) / 1e9, 1),
        "tokens": response.get("eval_count", 0),
        "tokens_per_second": _speed(response.get("eval_count"), response.get("eval_duration")),
        "prompt_tokens_per_second": _speed(response.get("prompt_eval_count"),
                                           response.get("prompt_eval_duration")),
    }, measured, budget)


def bench_llama(source, plan, tokens=64, prompt=DEFAULT_PROMPT, with_projector=False,
                log_path=None, report=None, unload_others=False):
    """Run the caged llama.cpp server once and measure it, from launch to the last token."""
    budget = plan["budget"]
    clear_ollama(None, unload_others)
    baseline = monitor.settle_vram()
    projector = source.projectors[0].path if with_projector and source.projectors else None
    server = engine.Server(source.model.path, plan["settings"], budget,
                           projector=projector, log_path=log_path, report=report)
    if server.caged:
        prepare_cage(source, projector)
    limits = {"caged": False, "ram_hard": False, "cpu_hard": False}
    generation = None
    started = time.monotonic()
    server.launch()
    sampler = monitor.Sampler(server.process.pid)
    sampler.start()
    try:
        server.wait_ready()
        load_seconds = time.monotonic() - started
        sampler.cgroup = server.cgroup
        limits = server.enforcement()
        server.complete(prompt, WARMUP_TOKENS)
        generation = time.monotonic()
        response = server.complete(prompt, tokens)
    finally:
        sampler.stop()
        measured = sampler.summary(baseline, since=generation)
        server.stop()
    timings = response.get("timings", {})
    return _finish({
        "engine": "llama.cpp (caged)" if limits["caged"] else "llama.cpp (no cage)",
        "model": source.spec,
        "limits": limits,
        "load_seconds": round(load_seconds, 1),
        "tokens": timings.get("predicted_n", 0),
        "tokens_per_second": round(timings.get("predicted_per_second", 0), 2) or None,
        "prompt_tokens_per_second": round(timings.get("prompt_per_second", 0), 2) or None,
        "thermal_pauses": server.pauses,
        "sensor_failures": server.sensor_failures,
    }, measured, budget)


def launcher_script(source, plan, with_projector=False):
    """A shell script that starts the caged llama.cpp server for this plan."""
    binary, lib_dir = hardware.find_engine("llama-server")
    binary = binary or "llama-server"
    projector = source.projectors[0].path if with_projector and source.projectors else None
    environment = engine.engine_environment(lib_dir) if lib_dir else {}
    flags = engine.supported_flags(binary, environment or None) if os.path.isabs(binary) else None
    command = engine.server_command(binary, source.model.path, plan["settings"], 8080, projector,
                                    flags=flags)
    lines = ["#!/bin/sh", "# Generated by Shu: llama.cpp server held to the planned budget.", "set -e"]
    for key in ("LD_LIBRARY_PATH", "GGML_BACKEND_PATH"):
        if lib_dir and key in environment:
            lines.append(f"export {key}={shlex.quote(environment[key])}")
    cage = engine.cage_prefix(plan["budget"], "shu-server") if hardware.can_cage() else []
    lines.append("exec " + " ".join(shlex.quote(part) for part in cage + command))
    return "\n".join(lines) + "\n"


def lmstudio_card(source, plan):
    settings, budget = plan["settings"], plan["budget"]
    total = plan["layers"]["total"]
    ratio = round(settings["num_gpu"] / total, 2) if total else 0
    return f"""# LM Studio settings for {source.model.name}

Generated by Shu for a budget of {fmt_bytes(budget['ram'])} RAM, {fmt_bytes(budget['vram'])} VRAM
and {budget['threads']} CPU threads. Set these in the model's load settings
(My Models, gear icon, or the load dialog with "Manually choose model load parameters").

| Setting | Value |
|---|---|
| GPU Offload | {settings['num_gpu']} of {total} layers |
| Context Length | {settings['num_ctx']} |
| CPU Thread Pool Size | {settings['num_thread']} |
| Evaluation Batch Size | {settings['num_batch']} |
| Try mmap() | on |
| Keep Model in Memory | off |
| Offload KV Cache to GPU Memory | on |

Command line equivalent:

    lms load "<model key>" --gpu {ratio} --context-length {settings['num_ctx']}

LM Studio has no hard RAM or CPU limit of its own. These settings bound VRAM and
threads; for a kernel-enforced RAM limit use `shu serve`.
These values were computed by Shu but not measured inside LM Studio.
"""


def write_outputs(source, plan, out_dir, modelfile_text, with_projector=False):
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "Modelfile": modelfile_text,
        "run-llama-server.sh": launcher_script(source, plan, with_projector),
        "lmstudio.md": lmstudio_card(source, plan),
        "plan.json": json.dumps({"source": source.spec, "model_file": source.model.path, **plan},
                                indent=2) + "\n",
    }
    for filename, text in files.items():
        path = os.path.join(out_dir, filename)
        with open(path, "w") as handle:
            handle.write(text)
        if filename.endswith(".sh"):
            os.chmod(path, 0o755)
    return sorted(files)


def _from_line(path):
    """A Modelfile FROM line; a path that could smuggle in extra directives is refused."""
    if any(ord(character) < 32 for character in path):
        raise ActionError(f"the path {path!r} contains control characters")
    return f"FROM {path}\n"


def modelfile_for(source, settings, text_only=False):
    if source.kind == "ollama":
        if text_only and source.projectors:
            return ollama.rewritten_modelfile(source.modelfile, settings, [source.model.path])
        return ollama.capped_modelfile(source.spec, settings)
    if len(source.model.shards) > 1:
        raise ActionError(SPLIT_ADVICE)
    return _from_line(source.model.path) + "\n".join(ollama.parameter_lines(settings)) + "\n"


def layer_cost(source, settings):
    """VRAM that one more offloaded layer costs (weights plus its share of the KV cache)."""
    model = source.model
    high = planner.estimate(model, len(model.layer_bytes), settings["num_ctx"], settings["num_batch"])
    low = planner.estimate(model, len(model.layer_bytes) - 1, settings["num_ctx"], settings["num_batch"])
    return max(1, high["vram"] - low["vram"])


def next_settings(source, plan, bench, state):
    """One corrective step from a measurement: (new settings, reason), or None when nothing to change.

    Order: get back inside the budget first (VRAM, then RAM, then CPU); only then spend
    spare VRAM on more GPU layers, which is what makes generation faster.
    """
    settings, budget, measured = plan["settings"], plan["budget"], bench["measured"]
    vram, ram, cores = measured.get("vram_peak"), measured.get("ram_peak"), measured.get("cpu_cores_avg")
    step = layer_cost(source, settings)
    changed = dict(settings)
    if vram is not None and vram > budget["vram"] and settings["num_gpu"] > 0:
        changed["num_gpu"] = max(0, settings["num_gpu"] - max(1, math.ceil((vram - budget["vram"]) / step)))
        return changed, f"VRAM {fmt_bytes(vram)} is over {fmt_bytes(budget['vram'])}"
    # A model whose weights exceed the RAM budget cannot be fixed by tuning; only the cage holds it.
    if ram is not None and ram > budget["ram"] and plan["status"] != "paged":
        if settings["num_batch"] > 128:
            changed["num_batch"] = 128
            return changed, f"RAM {fmt_bytes(ram)} is over {fmt_bytes(budget['ram'])}: smaller batch"
        if settings["num_ctx"] > planner.MIN_CONTEXT:
            changed["num_ctx"] = max(planner.MIN_CONTEXT, settings["num_ctx"] // 2)
            changed["num_batch"] = min(settings["num_batch"], changed["num_ctx"])
            return changed, f"RAM {fmt_bytes(ram)} is over {fmt_bytes(budget['ram'])}: smaller context"
    limit = budget["cpu_quota"] / 100
    if cores is not None and cores > limit * 1.05 and settings["num_thread"] > 1:
        changed["num_thread"] = settings["num_thread"] - 1
        return changed, f"CPU {cores:.1f} cores is over {limit:.1f}: one thread fewer"
    if not state["grew"] and vram is not None and settings["num_gpu"] < plan["layers"]["total"]:
        state["grew"] = True
        # Grow only into GROW_TARGET of the budget: idle VRAM use drifts between runs.
        gain = int((budget["vram"] * GROW_TARGET - vram) // step)
        if gain > 0:
            changed["num_gpu"] = min(plan["layers"]["total"], settings["num_gpu"] + gain)
            return changed, f"VRAM {fmt_bytes(vram)} leaves room under {fmt_bytes(budget['vram'])}"
    return None


def apply(source, plan, name, out_dir, text_only=False, create=True, verify=False,
          tokens=64, report=print, unload_others=False):
    """Write the plan's files, create the Ollama model and optionally measure and tune it.

    With ``verify`` the settings that end up installed are always ones that were measured.
    """
    settings = plan["settings"]
    with_projector = bool(source.projectors) and not text_only
    projector_bytes = source.projector_bytes if with_projector else 0
    modelfile_path = os.path.join(out_dir, "Modelfile")

    def install():
        planner.refresh(plan, source.model, projector_bytes)
        written = write_outputs(source, plan, out_dir, modelfile_for(source, settings, text_only),
                                with_projector)
        if create:
            ollama.create(name, modelfile_path)
        return written

    result = {"name": name, "out_dir": out_dir, "files": install(), "created": create,
              "rounds": [], "verified": False, "exit_code": 0}
    if not (create and verify):
        return result

    def measure():
        report(f"measuring {name}: " + "  ".join(f"{key}={settings[key]}" for key in
                                                 ("num_gpu", "num_thread", "num_ctx", "num_batch")))
        bench = bench_ollama(name, plan["budget"], tokens=tokens, unload_others=unload_others)
        record = {"num_gpu": settings["num_gpu"], "settings": dict(settings), **bench}
        result["rounds"].append(record)
        seen[tuple(sorted(settings.items()))] = record
        return record

    seen, state, final = {}, {"grew": False}, None
    for _ in range(MAX_TUNE_ROUNDS):
        final = seen.get(tuple(sorted(settings.items())))
        if final:
            # Already measured: a later step overshot and came back, so this is the answer.
            result["rounds"].append(final)
            break
        final = measure()
        step = next_settings(source, plan, final, state)
        if not step:
            break
        changed, reason = step
        report(f"{reason}: retuning")
        settings.update(changed)
        install()
        final = None
    if final is None:
        # The rounds ran out right after a change; what is installed must still be measured.
        final = seen.get(tuple(sorted(settings.items()))) or measure()
    result["exit_code"] = monitor.verdict(final.get("checks", []))
    result["verified"] = result["exit_code"] == 0
    return result


def quantize(source, quant, out_path, threads):
    """Requantize a GGUF with llama-quantize; returns the output path."""
    binary, lib_dir = hardware.find_engine("llama-quantize")
    if not binary:
        raise ActionError(
            "llama-quantize was not found. Install Ollama (it bundles one) or llama.cpp, "
            "or set SHU_ENGINE_DIR to the folder that contains it."
        )
    if os.path.exists(out_path):
        raise ActionError(f"{out_path} already exists; choose another --out or remove it")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    command = [binary]
    if source.model.file_type not in UNQUANTIZED:
        command.append("--allow-requantize")
    command += [source.model.path, out_path, quant.upper(), str(threads)]
    result = subprocess.run(command, env=engine.engine_environment(lib_dir),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.isfile(out_path):
        tail = " | ".join(result.stderr.strip().splitlines()[-3:])
        if os.path.isfile(out_path):
            os.remove(out_path)
        raise ActionError(f"llama-quantize failed (exit {result.returncode}): {tail}")
    return out_path


def _modelfile_path(work_dir, name):
    os.makedirs(work_dir, exist_ok=True)
    safe = ollama.check_name(name).replace("/", "_").replace(":", "_")
    return os.path.join(work_dir, f"{safe}.Modelfile")


def convert(source, name=None, quant=None, out_path=None, threads=2, work_dir="shu-out",
            report=print):
    """Convert between model types; returns a dict describing what was produced.

    safetensors folder -> GGUF (llama.cpp converter), optionally quantized;
    GGUF or Ollama model -> requantized GGUF; any result -> Ollama model with ``name``.
    """
    produced = {}
    base = source.default_name()[: -len("-shu")]
    target = None
    if source.kind == "safetensors":
        final = out_path or os.path.join(work_dir, f"{base}-{(quant or 'F16').upper()}.gguf")
        if quant and quant.lower() not in hfconvert.OUTPUT_TYPES:
            intermediate = os.path.join(work_dir, f"{base}-F16.tmp.gguf")
            report("converting safetensors to GGUF (F16) ...")
            hfconvert.convert(source.spec, intermediate, "f16")
            try:
                report(f"quantizing to {quant.upper()} ...")
                target = quantize(sources.resolve(intermediate), quant, final, threads)
            finally:
                os.remove(intermediate)
        else:
            report("converting safetensors to GGUF ...")
            target = hfconvert.convert(source.spec, final, (quant or "f16").lower())
    elif quant:
        report(f"quantizing to {quant.upper()} ...")
        target = quantize(source, quant,
                          out_path or os.path.join(work_dir, f"{base}-{quant.upper()}.gguf"), threads)
    elif not name:
        raise ActionError("nothing to do: pass --quant to requantize and/or --name to import into Ollama")

    if target:
        produced["gguf"] = os.path.abspath(target)
        produced["size"] = os.path.getsize(target)
    if name:
        if len(source.model.shards if source.model else []) > 1 and not target:
            raise ActionError(SPLIT_ADVICE)
        modelfile_path = _modelfile_path(work_dir, name)
        final = produced["gguf"] if target else source.model.path
        text = _from_line(final)
        if source.kind == "ollama":
            text = ollama.rewritten_modelfile(source.modelfile, {}, source.files,
                                              replace={source.model.path: final})
        with open(modelfile_path, "w") as handle:
            handle.write(text)
        report(f"importing into Ollama as {name} ...")
        ollama.create(name, modelfile_path)
        produced["ollama_model"] = name
    return produced


def describe_checks(checks):
    """Rows of text for a results table; an unmeasured value is shown as such, never as a pass."""
    lines = []
    for label, used, limit, ok in checks:
        as_bytes = label in ("RAM", "VRAM")
        limit_text = fmt_bytes(limit) if as_bytes else f"{limit:.1f}"
        if used is None:
            lines.append((label, "no reading", limit_text, "n/a", "NOT MEASURED"))
            continue
        used_text = fmt_bytes(used) if as_bytes else f"{used:.1f}"
        share = f"{100 * used / limit:.0f}%" if limit else "n/a"
        lines.append((label, used_text, limit_text, share, "PASS" if ok else "OVER"))
    return lines
