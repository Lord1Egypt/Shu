"""Command line interface and interactive wizard."""

import argparse
import json
import os
import signal
import sys
import time

from . import __version__, actions, engine, hardware, hfconvert, monitor, ollama, planner, sources
from .hardware import fmt_bytes

BANNER = r"""
   ___ _
  / __| |_ _  _     Shu {version}
  \__ \ ' \ || |    big models, small footprint
  |___/_||_\_,_|
"""

ERRORS = (sources.SourceError, ollama.OllamaError, engine.EngineError, actions.ActionError,
          hfconvert.ConverterError, ValueError, OSError)


def _color(text, code):
    if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
        return f"\033[{code}m{text}\033[0m"
    return text


def _table(rows, header=None):
    rows = [[str(cell) for cell in row] for row in ([header] if header else []) + list(rows)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
        if header and index == 0:
            print("  " + "  ".join("-" * width for width in widths))


def _emit(args, data, human):
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, default=str))
    else:
        human()


def add_budget_arguments(parser):
    group = parser.add_argument_group("budget")
    group.add_argument("--percent", type=float, default=30.0,
                       help="share of RAM, VRAM and CPU to allow (default: 30)")
    group.add_argument("--ram", type=float, metavar="GB", help="RAM limit in GiB (overrides --percent)")
    group.add_argument("--vram", type=float, metavar="GB", help="VRAM limit in GiB (overrides --percent)")
    group.add_argument("--cpu", type=int, metavar="THREADS", help="CPU threads (overrides --percent)")
    group.add_argument("--ctx", type=int, metavar="TOKENS", help="context length (default: up to 4096)")
    group.add_argument("--batch", type=int, metavar="TOKENS", help="batch size (default: 512)")
    group.add_argument("--max-temp", type=float, metavar="C",
                       help="pause the caged engine at this temperature in Celsius")
    group.add_argument("--text-only", action="store_true",
                       help="leave out the vision/audio projector to save memory")


def budget_from(args, hw):
    return planner.make_budget(hw, percent=args.percent, ram_gb=args.ram, vram_gb=args.vram,
                               cpu=args.cpu, ctx=args.ctx, batch=args.batch, max_temp=args.max_temp)


def make_plan(source, budget, projector_bytes):
    """Plan a model; when it will page from disk, time that disk instead of assuming a speed."""
    plan = planner.plan(source.model, budget, projector_bytes)
    if plan["status"] == "paged" and plan["estimate"]["paged_bytes"]:
        speed = hardware.read_speed(source.model.path)
        if speed:
            plan = planner.plan(source.model, budget, projector_bytes, disk_speed=speed)
    return plan


def planned(args):
    """Resolve the model and compute its plan from the command line."""
    hw = hardware.detect()
    source = sources.resolve(args.model)
    if source.kind == "safetensors":
        raise sources.SourceError(
            f"{args.model} is a safetensors folder. Convert it first: "
            f"shu convert {args.model} --quant Q4_K_M"
        )
    projector = 0 if args.text_only else source.projector_bytes
    return hw, source, make_plan(source, budget_from(args, hw), projector)


def print_hardware(hw):
    rows = [
        ("CPU", f"{hw['cpu_name']} ({hw['physical_cores']} cores, {hw['logical_cpus']} threads)"),
        ("RAM", f"{fmt_bytes(hw['ram_total'])} total, {fmt_bytes(hw['ram_available'])} available"),
        ("Swap", f"{fmt_bytes(hw['swap_total'])} total"),
    ]
    for card in hw["gpus"]:
        rows.append(("GPU", f"{card['name']}, {fmt_bytes(card['total'])} VRAM, "
                            f"{fmt_bytes(card['used'])} in use, {card['temperature']:.0f} C"))
    if not hw["gpus"]:
        rows.append(("GPU", "no NVIDIA GPU found: plans will be CPU-only"))
    rows.append(("CPU temperature", f"{hw['cpu_temperature']:.0f} C" if hw["cpu_temperature"]
                 else "no sensor (normal inside WSL)"))
    rows.append(("Cage", "available (systemd user scope: hard RAM and CPU limits)" if hw["can_cage"]
                 else "not available: limits are settings only, not enforced by the kernel"))
    rows.append(("Ollama", hw["ollama"] or "not found"))
    rows.append(("llama-server", hw["llama_server"] or "not found"))
    rows.append(("llama-quantize", hw["llama_quantize"] or "not found"))
    _table(rows)


def print_model(source):
    model = source.model
    rows = [
        ("Model", model.name),
        ("Architecture", model.architecture),
        ("Parameters", f"{model.parameters / 1e9:.2f} B"),
        ("Quantization", f"{model.file_type} ({model.bits_per_weight:.2f} bits per weight)"),
        ("Size on disk", fmt_bytes(model.file_size)),
        ("Layers", f"{len(model.layer_bytes)} (about {fmt_bytes(sum(model.layer_bytes) // max(1, len(model.layer_bytes)))} each)"),
        ("Trained context", model.context_length),
    ]
    for projector in source.projectors:
        rows.append(("Projector", f"{fmt_bytes(projector.file_size)} (vision/audio, {projector.file_type})"))
    _table(rows)


def print_plan(plan):
    budget, settings, numbers = plan["budget"], plan["settings"], plan["estimate"]
    good = plan["status"] in ("gpu", "split", "cpu")
    print("  " + _color(plan["verdict"], "32" if good else ("33" if plan["status"] == "paged" else "31")))
    print()
    _table([
        ("RAM", fmt_bytes(budget["ram"]), fmt_bytes(numbers["ram_resident"]),
         f"working {fmt_bytes(numbers['ram_working'])} + weights {fmt_bytes(numbers['ram_weights'])}"),
        ("VRAM", fmt_bytes(budget["vram"]), fmt_bytes(numbers["vram"]),
         f"{settings['num_gpu']} of {plan['layers']['total']} layers on the GPU"),
        ("CPU", f"{budget['cpu_quota'] / 100:.1f} cores", f"{settings['num_thread']} threads", ""),
    ], header=("Resource", "Budget", "Planned", "Detail"))
    print()
    print(f"  Settings: num_gpu={settings['num_gpu']}  num_thread={settings['num_thread']}  "
          f"num_ctx={settings['num_ctx']}  num_batch={settings['num_batch']}  use_mmap=true")
    for note in plan["notes"]:
        print("  - " + note)


def print_bench(bench):
    measured, limits = bench["measured"], bench.get("limits", {})
    print(f"  Engine: {bench['engine']}    load {bench['load_seconds']} s    "
          f"{bench['tokens']} tokens at {bench['tokens_per_second'] or '?'} tokens/s")
    colors = {"PASS": "32", "OVER": "31", "NOT MEASURED": "33"}
    _table([(label, used, limit, share, _color(state, colors[state]))
            for label, used, limit, share, state in actions.describe_checks(bench["checks"])],
           header=("Resource", "Measured", "Budget", "Of budget", "Result"))
    detail = []
    if limits.get("ram_hard") and limits.get("cpu_hard"):
        detail.append("RAM and CPU limits are enforced by the kernel (read back from the cage)")
    else:
        soft = [name for name, key in (("RAM", "ram_hard"), ("CPU", "cpu_hard")) if not limits.get(key)]
        detail.append(f"{' and '.join(soft)} not enforced by the kernel here: bounded by the "
                      "settings only, so the measurement above is the evidence")
    if measured.get("ram_peak") is not None:
        detail.append(f"process memory at its peak: {fmt_bytes(measured['ram_private'])} private + "
                      f"{fmt_bytes(measured['ram_model_pages'])} model pages (reclaimable page cache)")
    if measured.get("cgroup_peak"):
        detail.append(f"the cage was charged {fmt_bytes(measured['cgroup_peak'])} at its peak")
    if bench.get("disk_read_per_token"):
        detail.append(f"read {fmt_bytes(bench['disk_read_per_token'])} from disk per generated token")
    if measured.get("gpu_temp_max") is not None:
        detail.append(f"GPU peaked at {measured['gpu_temp_max']:.0f} C, "
                      f"average utilization {measured['gpu_util_avg']}%")
    if bench.get("thermal_pauses"):
        detail.append(f"thermal guard paused the engine {bench['thermal_pauses']} time(s)")
    if bench.get("sensor_failures"):
        detail.append(f"temperature sensors failed to answer {bench['sensor_failures']} time(s)")
    if bench.get("exit_code") == 3:
        detail.append("some values could not be measured, so this run does NOT confirm the budget")
    for line in detail:
        print("  - " + line)


def cmd_detect(args):
    hw = hardware.detect()
    _emit(args, hw, lambda: print_hardware(hw))
    return 0


def cmd_inspect(args):
    source = sources.resolve(args.model)
    if source.kind == "safetensors":
        print(f"  {args.model}: safetensors folder. Convert it with `shu convert`.")
        return 0
    data = dict(source.model.summary(), projectors=[p.summary() for p in source.projectors])
    _emit(args, data, lambda: print_model(source))
    return 0


def cmd_plan(args):
    _, source, plan = planned(args)

    def human():
        print_model(source)
        print()
        print_plan(plan)
    _emit(args, plan, human)
    return 0


def cmd_apply(args):
    _, source, plan = planned(args)
    name = ollama.check_name(args.name or source.default_name())
    out_dir = args.out or os.path.join("shu-out", name.replace("/", "_").replace(":", "_"))
    quiet = args.json
    if not quiet:
        print_model(source)
        print()
        print_plan(plan)
        print()
    result = actions.apply(source, plan, name, out_dir, text_only=args.text_only,
                           create=not args.no_create, verify=args.verify, tokens=args.tokens,
                           unload_others=args.unload_others,
                           report=(lambda message: None) if quiet else (lambda m: print("  " + m)))
    if quiet:
        print(json.dumps({"plan": plan, **result}, indent=2, default=str))
        return 0
    print(f"  Wrote {', '.join(result['files'])} to {out_dir}")
    if result["created"]:
        print(f"  Created Ollama model {_color(name, '36')}. Run it with: ollama run {name}")
    if result["rounds"]:
        print()
        print_bench(result["rounds"][-1])
        print("  " + (_color("Verified: the installed settings were measured inside the budget.", "32")
                      if result["verified"] else
                      _color("Not verified: the installed settings did not measure inside the budget.", "31")))
    if plan["status"] == "paged":
        print(f"  This model is larger than the RAM budget. For a kernel-enforced limit run: "
              f"shu serve {args.model} --percent {args.percent:g}")
    return 0


def cmd_bench(args):
    hw, source, plan = planned(args)
    if args.engine == "ollama":
        if source.kind != "ollama":
            raise sources.SourceError("--engine ollama needs an Ollama model name")
        bench = actions.bench_ollama(args.model, plan["budget"], tokens=args.tokens,
                                     unload_others=args.unload_others)
    else:
        bench = actions.bench_llama(
            source, plan, tokens=args.tokens,
            with_projector=bool(source.projectors) and not args.text_only, log_path=args.log,
            unload_others=args.unload_others,
            report=None if args.json else (lambda message: print("  " + message)))
    _emit(args, bench, lambda: print_bench(bench))
    return bench["exit_code"]


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def cmd_serve(args):
    hw, source, plan = planned(args)
    budget = plan["budget"]
    print_plan(plan)
    print()
    if plan["status"] == "impossible":
        print("  Not starting: the working memory does not fit this budget.")
        return 2
    if plan["status"] == "paged":
        actions.evict_from_page_cache(source.model.path)
    projector = source.projectors[0].path if source.projectors and not args.text_only else None
    server = engine.Server(source.model.path, plan["settings"], budget, projector=projector,
                           port=args.port, log_path=args.log,
                           report=lambda message: print("  thermal guard: " + message))
    print(f"  Loading {source.model.name} ..."
          + (f" (thermal guard active from now: pauses at {budget['max_temp']:.0f} C)"
             if budget.get("max_temp") else ""))
    signal.signal(signal.SIGTERM, _interrupt)
    server.start()
    limits = server.enforcement()
    for label, key, value in (("RAM", "ram_hard", fmt_bytes(budget["ram"])),
                              ("CPU", "cpu_hard", f"{budget['cpu_quota'] / 100:.1f} cores")):
        print(f"  {label} <= {value}: " + ("enforced by the kernel" if limits[key]
                                           else "NOT enforced (no cage on this system); settings only"))
    print(f"  Ready: {_color(server.url, '36')}  (OpenAI-compatible API at {server.url}/v1). Ctrl+C stops it.")
    sampler = monitor.Sampler(server.process.pid, cgroup=server.cgroup, interval=2.0)
    sampler.start()
    try:
        while server.process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sampler.stop()
        server.stop()
    measured = sampler.summary(0)
    peak = max(measured.get("cgroup_peak") or 0, measured.get("ram_peak") or 0)
    print(f"\n  Stopped. Peak RAM {fmt_bytes(peak) if peak else 'not measured'}, "
          f"thermal pauses: {server.pauses}")
    return 0


def cmd_convert(args):
    source = sources.resolve(args.model)
    threads = args.threads or max(1, int((os.cpu_count() or 1) * 0.3))
    quiet = args.json
    produced = actions.convert(source, name=args.name, quant=args.quant, out_path=args.out,
                               threads=threads,
                               report=(lambda message: None) if quiet else (lambda m: print("  " + m)))

    def human():
        if "gguf" in produced:
            print(f"  Wrote {produced['gguf']} ({fmt_bytes(produced['size'])})")
        if "ollama_model" in produced:
            print(f"  Created Ollama model {produced['ollama_model']}")
    _emit(args, produced, human)
    return 0


def cmd_setup_converter(args):
    if hfconvert.ready() and not args.force:
        print(f"  The safetensors converter is already installed in {hfconvert.home()}")
        return 0
    root = hfconvert.install(report=lambda message: print("  " + message))
    print(f"  Installed in {root}. You can now run: shu convert <safetensors folder> --quant Q4_K_M")
    return 0


def _ask(prompt, default=None, convert=str, valid=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            raw = input(f"  {prompt}{suffix}: ").strip()
        except EOFError:
            raise KeyboardInterrupt from None
        if not raw and default is not None:
            return default
        try:
            value = convert(raw)
        except ValueError:
            print("  Please enter a valid value.")
            continue
        if valid is None or valid(value):
            return value
        print("  That value is out of range.")


def wizard():
    print(BANNER.format(version=__version__))
    hw = hardware.detect()
    print_hardware(hw)

    print("\n  Step 1: choose a model")
    names = []
    if ollama.available():
        names = [entry["name"] for entry in ollama.local_models()]
        for index, name in enumerate(names, 1):
            print(f"    {index}. {name}")
    print("    Or type the path of a .gguf file or a safetensors folder.")
    while True:
        choice = _ask("Model")
        spec = names[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(names) else \
            os.path.expanduser(choice)
        try:
            source = sources.resolve(spec)
        except ERRORS as error:
            print(f"  {error}")
            continue
        if source.kind == "safetensors":
            if not hfconvert.ready():
                answer = _ask("Safetensors needs the llama.cpp converter (PyTorch, about 1 GB). "
                              "Install it now? (y/n)", "n")
                if not answer.lower().startswith("y"):
                    continue
                try:
                    hfconvert.install(report=lambda message: print("  " + message))
                except ERRORS as error:
                    print(f"  {error}")
                    continue
            quant = _ask("Quantization for the converted model", "Q4_K_M")
            try:
                produced = actions.convert(source, quant=quant, threads=max(1, hw["logical_cpus"] // 3),
                                           report=lambda message: print("  " + message))
                source = sources.resolve(produced["gguf"])
            except ERRORS as error:
                print(f"  {error}")
                continue
        break
    print()
    print_model(source)

    print("\n  Step 2: choose the budget")
    percent = _ask("Share of the machine in percent (25 to 35 keeps a laptop cool)", 30.0, float,
                   lambda value: 1 <= value <= 100)
    custom = _ask("Set RAM, VRAM and CPU separately? (y/n)", "n").lower().startswith("y")
    ram = vram = cpu = None
    if custom:
        ram = _ask("RAM in GiB", round(hw["ram_total"] * percent / 100 / hardware.GIB, 1), float,
                   lambda value: value > 0)
        if hw["vram_total"]:
            vram = _ask("VRAM in GiB", round(hw["vram_total"] * percent / 100 / hardware.GIB, 1), float,
                        lambda value: value >= 0)
        cpu = _ask("CPU threads", max(1, int(hw["logical_cpus"] * percent / 100)), int,
                   lambda value: 1 <= value <= hw["logical_cpus"])
    ctx = _ask("Context length in tokens (0 = automatic)", 0, int, lambda value: value == 0 or value >= 256)
    max_temp = _ask("Pause above this temperature in Celsius (0 = no thermal guard)", 80.0, float,
                    lambda value: value == 0 or 40 <= value <= 110)
    text_only = False
    if source.projectors:
        text_only = _ask(f"Drop the vision/audio projector to save {fmt_bytes(source.projector_bytes)}? (y/n)",
                         "n").lower().startswith("y")
    budget = planner.make_budget(hw, percent=percent, ram_gb=ram, vram_gb=vram, cpu=cpu,
                                 ctx=ctx or None, max_temp=max_temp or None)
    plan = make_plan(source, budget, 0 if text_only else source.projector_bytes)
    print()
    print_plan(plan)

    advice = plan.get("recommendation", {})
    if advice.get("quantization") and hw["llama_quantize"]:
        answer = _ask(f"Requantize to {advice['quantization']} so it fits (lower quality, "
                      f"about {fmt_bytes(advice['estimated_size'])})? (y/n)", "n")
        if answer.lower().startswith("y"):
            name = source.default_name()[: -len("-shu")] + "-" + advice["quantization"].lower()
            print("  Requantizing (this can take several minutes) ...")
            actions.convert(source, name=name if ollama.available() else None,
                            quant=advice["quantization"], threads=budget["threads"])
            if ollama.available():
                source = sources.resolve(name)
                plan = make_plan(source, budget, 0 if text_only else source.projector_bytes)
                print()
                print_plan(plan)

    print("\n  Step 3: choose what to do")
    options = []
    if ollama.available():
        options += [("Create the limited Ollama model, then measure and tune it", "verify"),
                    ("Create the limited Ollama model", "create")]
    if hw["llama_server"]:
        options += [("Measure it once in the caged llama.cpp engine", "bench"),
                    ("Start the caged llama.cpp server now", "serve")]
    options += [("Only write the settings files", "files"), ("Quit", "quit")]
    for index, (label, _) in enumerate(options, 1):
        print(f"    {index}. {label}")
    action = options[_ask("Action", 1, int, lambda value: 1 <= value <= len(options)) - 1][1]
    if action == "quit":
        return 0

    name = source.default_name()
    if action in ("verify", "create"):
        name = ollama.check_name(_ask("Name of the new Ollama model", name))
    unload_others = False
    if action in ("verify", "bench") and ollama.available():
        busy = [entry["name"] for entry in ollama.running()]
        if busy:
            print(f"  Ollama has models loaded: {', '.join(busy)}. They would distort the measurement.")
            if not _ask("Unload them now? (y/n)", "n").lower().startswith("y"):
                print("  Nothing was changed. Run again when they are unloaded.")
                return 0
            unload_others = True
    out_dir = os.path.join("shu-out", name.replace("/", "_").replace(":", "_"))
    print()
    if action in ("verify", "create", "files"):
        result = actions.apply(source, plan, name, out_dir, text_only=text_only,
                               create=action != "files", verify=action == "verify",
                               unload_others=unload_others,
                               report=lambda message: print("  " + message))
        print(f"  Wrote {', '.join(result['files'])} to {out_dir}")
        if result["created"]:
            print(f"  Created Ollama model {name}. Run it with: ollama run {name}")
        if result["rounds"]:
            print()
            print_bench(result["rounds"][-1])
            print("  Verified: the installed settings were measured inside the budget."
                  if result["verified"] else
                  "  Not verified: the installed settings did not measure inside the budget.")
    elif action == "bench":
        print("  Loading and measuring ...")
        print_bench(actions.bench_llama(source, plan, unload_others=unload_others,
                                        with_projector=bool(source.projectors) and not text_only,
                                        report=lambda m: print("  " + m)))
    elif action == "serve":
        arguments = ["serve", source.spec, "--percent", str(percent)]
        for flag, value in (("--ram", ram), ("--vram", vram), ("--cpu", cpu),
                            ("--ctx", ctx or None), ("--max-temp", max_temp or None)):
            if value is not None:
                arguments += [flag, str(value)]
        if text_only:
            arguments.append("--text-only")
        return main(arguments)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="shu",
        description="Fit GGUF models inside a fixed share of this machine's RAM, CPU and GPU. "
                    "Run without arguments for the interactive wizard.",
    )
    parser.add_argument("--version", action="version", version=f"shu {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="command")

    def command(name, handler, summary, model=True, budget=False):
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.set_defaults(handler=handler)
        if model:
            sub.add_argument("model", help="a .gguf file, an Ollama model name, or a model folder")
        if budget:
            add_budget_arguments(sub)
        sub.add_argument("--json", action="store_true", help="print machine-readable JSON")
        return sub

    commands.add_parser("wizard", help="interactive guided setup").set_defaults(handler=lambda args: wizard())
    command("detect", cmd_detect, "show the CPU, RAM, GPU and engines of this machine", model=False)
    command("inspect", cmd_inspect, "show what is inside a model")
    command("plan", cmd_plan, "compute settings that keep a model inside the budget", budget=True)

    sub = command("apply", cmd_apply, "create an Ollama model limited to the budget", budget=True)
    sub.add_argument("--name", help="name of the new Ollama model (default: <model>-shu)")
    sub.add_argument("--out", metavar="DIR", help="folder for the generated files")
    sub.add_argument("--no-create", action="store_true", help="only write the files")
    sub.add_argument("--verify", action="store_true",
                     help="load the model, measure real usage and retune the GPU layers")
    sub.add_argument("--tokens", type=int, default=64, help="tokens to generate when verifying")
    sub.add_argument("--unload-others", action="store_true",
                     help="allow unloading other models Ollama has loaded, for a clean measurement")

    sub = command("bench", cmd_bench, "run a model once and compare real usage with the budget",
                  budget=True)
    sub.add_argument("--engine", choices=("llama", "ollama"), default="llama",
                     help="llama: caged llama.cpp server (default); ollama: an existing Ollama model")
    sub.add_argument("--tokens", type=int, default=64, help="tokens to generate")
    sub.add_argument("--unload-others", action="store_true",
                     help="allow unloading other models Ollama has loaded, for a clean measurement")
    sub.add_argument("--log", metavar="FILE", help="write the llama.cpp server log here")

    sub = command("serve", cmd_serve, "serve a model from a caged llama.cpp server", budget=True)
    sub.add_argument("--port", type=int, help="port to listen on (default: a free port)")
    sub.add_argument("--log", metavar="FILE", help="write the llama.cpp server log here")

    sub = command("convert", cmd_convert,
                  "convert a model: safetensors folder to GGUF, requantize a GGUF or Ollama model, "
                  "import the result into Ollama")
    sub.add_argument("--quant", metavar="TYPE", help="target quantization, for example Q4_K_M or Q3_K_M")
    sub.add_argument("--name", help="create an Ollama model with this name from the result")
    sub.add_argument("--out", metavar="FILE", help="path of the requantized GGUF")
    sub.add_argument("--threads", type=int, help="threads for quantizing (default: 30%% of the CPU)")

    sub = commands.add_parser("setup-converter",
                              help="install the safetensors converter (PyTorch, about 1 GB)")
    sub.add_argument("--force", action="store_true", help="reinstall even if already present")
    sub.set_defaults(handler=cmd_setup_converter)
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    if not argv:
        if sys.stdin.isatty() and sys.stdout.isatty():
            argv = ["wizard"]
        else:
            parser.print_help()
            return 0
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 0
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("\n  Cancelled.", file=sys.stderr)
        return 130
    except ERRORS as error:
        print(f"shu: {error}", file=sys.stderr)
        return 1
