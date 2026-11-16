#!/usr/bin/env python3
"""Run every check: the unit tests, then each command of the CLI as a real subprocess.

    python tools/test_all.py                  offline checks (no GPU, Ollama or network needed)
    python tools/test_all.py --live MODEL     also load MODEL (an Ollama model name) for real:
                                              caged llama.cpp run and an Ollama create + verify

Exits 0 only if everything passed.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests import fakegguf  # noqa: E402

RESULTS = []


def shu(*arguments, stdin=None):
    return subprocess.run([sys.executable, "-m", "shu", *arguments], cwd=ROOT, text=True,
                          capture_output=True, input=stdin, timeout=3600)


def check(label, result, expect=0, contains=None, validate=None):
    ok = result.returncode == expect
    if ok and contains:
        ok = contains in result.stdout + result.stderr
    if ok and validate:
        try:
            ok = bool(validate(json.loads(result.stdout)))
        except (ValueError, KeyError, TypeError):
            ok = False
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        exit {result.returncode} (expected {expect})")
        print("        " + (result.stdout + result.stderr).strip().replace("\n", "\n        ")[-1500:])


def offline(folder):
    model = fakegguf.language_model(os.path.join(folder, "tiny.gguf"))
    big = fakegguf.transformer(os.path.join(folder, "big.gguf"), "Big", layers=80,
                               layer_bytes=515 << 20, embd=8192, heads=64, heads_kv=8,
                               vocab=128256, context=131072, input_bytes=600 << 20,
                               output_bytes=860 << 20)
    out_dir = os.path.join(folder, "out")

    check("shu --version", shu("--version"), contains="shu ")
    check("shu --help", shu("--help"), contains="wizard")
    check("shu (no arguments, no terminal) does not block", shu(stdin=""), contains="usage: shu")
    for command in ("detect", "inspect", "plan", "apply", "bench", "serve", "convert",
                    "setup-converter", "wizard"):
        check(f"shu {command} --help", shu(command, "--help"), contains="usage: shu")
    check("shu detect", shu("detect"), contains="CPU")
    check("shu detect --json", shu("detect", "--json"), validate=lambda d: d["logical_cpus"] >= 1)
    check("shu inspect FILE", shu("inspect", model), contains="Tiny Test")
    check("shu inspect FILE --json", shu("inspect", model, "--json"),
          validate=lambda d: d["layers"] == 4)
    check("shu plan FILE", shu("plan", model), contains="Settings: num_gpu=")
    for percent in ("25", "30", "35"):
        check(f"shu plan FILE --percent {percent} --json", shu("plan", model, "--percent", percent, "--json"),
              validate=lambda d: d["estimate"]["vram"] <= d["budget"]["vram"])
    check("shu plan FILE with every budget flag",
          shu("plan", model, "--ram", "2", "--vram", "0", "--cpu", "2", "--ctx", "2048",
               "--batch", "128", "--max-temp", "75", "--text-only", "--json"),
          validate=lambda d: d["settings"] == {"num_gpu": 0, "num_thread": 2, "num_ctx": 2048,
                                               "num_batch": 128, "use_mmap": True})
    check("shu plan of a 70B-sized model reports paging",
          shu("plan", big, "--ram", "4", "--vram", "2", "--json"),
          validate=lambda d: d["status"] == "paged" and d["estimate"]["paged_bytes"] > 30 << 30)
    check("shu apply FILE --no-create", shu("apply", model, "--no-create", "--out", out_dir),
          contains="Wrote Modelfile")
    check("shu apply FILE --no-create --json",
          shu("apply", model, "--no-create", "--out", out_dir, "--json"),
          validate=lambda d: d["created"] is False and len(d["files"]) == 4)
    check("generated files exist", subprocess.run(
        ["test", "-x", os.path.join(out_dir, "run-llama-server.sh"), "-a",
         "-s", os.path.join(out_dir, "Modelfile"), "-a", "-s", os.path.join(out_dir, "plan.json"),
         "-a", "-s", os.path.join(out_dir, "lmstudio.md")], capture_output=True, text=True))
    check("missing file is a clean error", shu("plan", os.path.join(folder, "nope.gguf")),
          expect=1, contains="does not exist")
    check("bad percent is a clean error", shu("plan", model, "--percent", "0"), expect=1,
          contains="shu: ")
    check("unsafe model name is rejected", shu("apply", model, "--no-create", "--name=-rf"),
          expect=1, contains="invalid model name")
    check("convert with nothing to do is a clean error", shu("convert", model), expect=1,
          contains="nothing to do")
    with open(os.path.join(folder, "junk.gguf"), "wb") as handle:
        handle.write(b"not a model")
    check("non-GGUF file is a clean error", shu("inspect", os.path.join(folder, "junk.gguf")),
          expect=1, contains="not a GGUF file")
    check("wizard stops cleanly when input ends", shu("wizard", stdin=""), expect=130)


def live(model, folder):
    check(f"shu inspect {model}", shu("inspect", model, "--json"),
          validate=lambda d: d["layers"] > 0)
    check(f"shu bench {model} (caged llama.cpp, 30%)",
          shu("bench", model, "--percent", "30", "--text-only", "--tokens", "32", "--json"),
          validate=lambda d: all(row[3] for row in d["checks"]) and d["tokens"] > 0)
    name = "shu-selftest"
    check(f"shu apply {model} --verify (Ollama, 30%)",
          shu("apply", model, "--percent", "30", "--verify", "--tokens", "32", "--name", name,
               "--out", os.path.join(folder, "live"), "--json"),
          validate=lambda d: d["created"] and all(row[3] for row in d["rounds"][-1]["checks"]
                                                  if row[0] == "VRAM"))
    check("shu bench --engine ollama on the created model",
          shu("bench", name, "--engine", "ollama", "--percent", "30", "--tokens", "32", "--json"),
          validate=lambda d: d["tokens"] > 0)
    subprocess.run(["ollama", "rm", name], capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--live", metavar="MODEL", help="Ollama model to load for the live checks")
    args = parser.parse_args()

    print("Unit tests")
    unit = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                          cwd=ROOT, text=True, capture_output=True)
    summary = unit.stderr.strip().splitlines()
    print("  " + (summary[-3] if len(summary) >= 3 else "") + "  " + (summary[-1] if summary else ""))
    RESULTS.append(unit.returncode == 0)
    if unit.returncode != 0:
        print(unit.stderr[-3000:])

    with tempfile.TemporaryDirectory() as folder:
        print("Command line (offline)")
        offline(folder)
        if args.live:
            print(f"Command line (live, {args.live})")
            live(args.live, folder)

    failed = RESULTS.count(False)
    print(f"\n{len(RESULTS) - failed} of {len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
