"""Convert Hugging Face safetensors models to GGUF with llama.cpp's official converter.

The converter needs PyTorch and transformers. Ptah itself has no dependencies, so
they are installed on request into a private virtual environment under the Ptah
cache, next to a pinned checkout of the converter script.
"""

import os
import shutil
import subprocess
import sys

REPOSITORY = "https://github.com/ggml-org/llama.cpp"
DEFAULT_TAG = "b12426"
SCRIPT = "convert_hf_to_gguf.py"
REQUIREMENTS = os.path.join("requirements", "requirements-convert_hf_to_gguf.txt")
OUTPUT_TYPES = ("f32", "f16", "bf16", "q8_0", "auto")


class ConverterError(RuntimeError):
    pass


def home():
    base = os.environ.get("PTAH_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "ptah")
    return os.path.join(base, "converter")


def _paths():
    root = home()
    source = os.path.join(root, "llama.cpp")
    return {
        "root": root,
        "source": source,
        "script": os.path.join(source, SCRIPT),
        "python": os.path.join(root, "venv", "bin", "python"),
        "ready": os.path.join(root, "ready"),
    }


def ready():
    paths = _paths()
    return all(os.path.isfile(paths[key]) for key in ("script", "python", "ready"))


def _run(command, what, **options):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, **options)
    if result.returncode != 0:
        tail = " | ".join(result.stdout.strip().splitlines()[-4:])
        raise ConverterError(f"{what} failed (exit {result.returncode}): {tail}")
    return result.stdout


def install(report=print):
    """Download the converter and its Python packages (about 1 GB on disk)."""
    if not shutil.which("git"):
        raise ConverterError("git is required to download the llama.cpp converter")
    paths = _paths()
    tag = os.environ.get("PTAH_LLAMA_CPP_TAG", DEFAULT_TAG)
    if os.path.isdir(paths["root"]):
        shutil.rmtree(paths["root"])
    os.makedirs(paths["root"])
    report(f"downloading the llama.cpp converter ({tag}) ...")
    _run(["git", "clone", "--quiet", "--depth", "1", "--branch", tag, "--filter=blob:none",
          "--sparse", REPOSITORY, paths["source"]], "git clone")
    _run(["git", "-C", paths["source"], "sparse-checkout", "set", "gguf-py", "requirements", "conversion"],
         "git sparse-checkout")
    if not os.path.isfile(paths["script"]):
        raise ConverterError(f"{SCRIPT} is missing from llama.cpp {tag}")
    report("creating a private Python environment ...")
    _run([sys.executable, "-m", "venv", os.path.join(paths["root"], "venv")], "python -m venv")
    report("installing PyTorch (CPU) and transformers; this takes a few minutes ...")
    _run([paths["python"], "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
          "-r", os.path.join(paths["source"], REQUIREMENTS)], "pip install", cwd=paths["source"])
    with open(paths["ready"], "w") as handle:
        handle.write(tag + "\n")
    return paths["root"]


def convert(model_dir, out_path, output_type="f16"):
    """Write ``model_dir`` (config.json plus safetensors) as a GGUF file at ``out_path``."""
    if not ready():
        raise ConverterError(
            "the safetensors converter is not installed. Run `ptah setup-converter` once "
            "(downloads PyTorch and transformers, about 1 GB)."
        )
    if output_type not in OUTPUT_TYPES:
        raise ConverterError(f"output type must be one of {', '.join(OUTPUT_TYPES)}")
    if os.path.exists(out_path):
        raise ConverterError(f"{out_path} already exists; choose another --out or remove it")
    paths = _paths()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        _run([paths["python"], paths["script"], os.path.abspath(model_dir),
              "--outfile", os.path.abspath(out_path), "--outtype", output_type],
             "safetensors conversion")
    except ConverterError:
        if os.path.isfile(out_path):
            os.remove(out_path)
        raise
    if not os.path.isfile(out_path):
        raise ConverterError("the converter finished without writing a GGUF file")
    return out_path
