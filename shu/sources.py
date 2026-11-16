"""Resolve what the user pointed at: a GGUF file, an Ollama model or a safetensors folder."""

import glob
import os
import re

from . import gguf, ollama


class SourceError(ValueError):
    pass


class Source:
    def __init__(self, kind, spec, model=None, projectors=(), modelfile=None, files=()):
        self.kind = kind                # "gguf", "ollama" or "safetensors"
        self.spec = spec
        self.model = model              # gguf.Model of the language model
        self.projectors = list(projectors)
        self.modelfile = modelfile      # Modelfile text for Ollama sources
        self.files = list(files)        # every GGUF path behind an Ollama model

    @property
    def projector_bytes(self):
        return sum(projector.weights_bytes for projector in self.projectors)

    def default_name(self):
        base = os.path.splitext(os.path.basename(self.spec.rstrip("/")))[0] if self.kind != "ollama" \
            else self.spec.split("/")[-1]
        if self.kind == "ollama" and base.endswith(":latest"):
            base = base[: -len(":latest")]
        slug = re.sub(r"[^a-z0-9._-]+", "-", base.lower()).strip("-.") or "model"
        return slug[:70] + "-shu"


def is_safetensors_dir(path):
    return (os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))
            and bool(glob.glob(os.path.join(path, "*.safetensors"))))


def resolve(spec):
    """Classify ``spec`` and read the GGUF headers behind it."""
    if os.path.isfile(spec):
        try:
            return Source("gguf", os.path.abspath(spec), model=gguf.read(spec))
        except (gguf.GGUFError, OSError) as error:
            raise SourceError(str(error)) from error
    if os.path.isdir(spec):
        if is_safetensors_dir(spec):
            return Source("safetensors", os.path.abspath(spec))
        found = sorted(glob.glob(os.path.join(spec, "*.gguf")))
        if len(found) == 1:
            return resolve(found[0])
        raise SourceError(
            f"{spec} is a folder without exactly one .gguf file or a safetensors model "
            "(config.json plus *.safetensors)"
        )
    if spec.lower().endswith((".gguf", ".safetensors")) or spec.startswith(("/", "./", "~")):
        raise SourceError(f"{spec} does not exist")
    try:
        details = ollama.show(spec)
    except ollama.OllamaError as error:
        raise SourceError(
            f"{spec!r} is neither a file nor a local Ollama model ({error}). "
            f"Pull it first with: ollama pull {spec}"
        ) from error
    modelfile = details.get("modelfile", "")
    files = ollama.source_files(modelfile)
    language, projectors = None, []
    for path in files:
        try:
            parsed = gguf.read(path)
        except (gguf.GGUFError, OSError):
            continue
        if parsed.is_language_model and language is None:
            language = parsed
        else:
            projectors.append(parsed)
    if language is None:
        raise SourceError(
            f"the Ollama model {spec!r} has no readable GGUF language model "
            "(it may use a non-GGUF format, or its files are not readable by this user)"
        )
    return Source("ollama", spec, model=language, projectors=projectors,
                  modelfile=modelfile, files=files)
