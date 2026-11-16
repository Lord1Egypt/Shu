"""Minimal Ollama client: HTTP API through urllib, model creation through the CLI."""

import json
import os
import re
import subprocess
import urllib.error
import urllib.request

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,199}$")
SHU_PARAMETERS = ("num_gpu", "num_thread", "num_ctx", "num_batch", "use_mmap")


class OllamaError(RuntimeError):
    pass


def check_name(name):
    if not NAME_PATTERN.match(name):
        raise OllamaError(f"invalid model name {name!r}")
    return name


def host():
    value = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if "://" not in value:
        value = "http://" + value
    if value.count(":") < 2:
        value += ":11434"
    return value.rstrip("/")


def api(path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(host() + path, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        raise OllamaError(f"Ollama answered {error.code} for {path}: {detail}") from error
    except (urllib.error.URLError, OSError) as error:
        raise OllamaError(f"cannot reach Ollama at {host()}: {error}") from error


def available():
    try:
        api("/api/version", timeout=5)
        return True
    except OllamaError:
        return False


def local_models():
    return api("/api/tags").get("models", [])


def show(name):
    return api("/api/show", {"model": check_name(name)})


def source_files(modelfile):
    """Absolute paths named by the FROM lines of a Modelfile."""
    paths = []
    for line in modelfile.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].upper() == "FROM" and parts[1].startswith("/"):
            paths.append(parts[1].strip())
    return paths


def running():
    return api("/api/ps").get("models", [])


def unload(name):
    api("/api/generate", {"model": name, "keep_alive": 0}, timeout=120)


def unload_all():
    for entry in running():
        unload(entry["name"])


def create(name, modelfile_path, quantize=None):
    command = ["ollama", "create", check_name(name), "-f", modelfile_path]
    if quantize:
        command += ["--quantize", quantize]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["unknown error"]
        raise OllamaError(f"ollama create failed: {detail[0]}")


def generate(name, prompt, num_predict, options=None, timeout=3600):
    body = {
        "model": check_name(name),
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": dict(options or {}, num_predict=num_predict),
    }
    return api("/api/generate", body, timeout=timeout)


def capped_modelfile(source_name, settings):
    """Modelfile that inherits everything from ``source_name`` and pins the settings."""
    lines = [f"FROM {check_name(source_name)}"]
    lines += parameter_lines(settings)
    return "\n".join(lines) + "\n"


def rewritten_modelfile(modelfile, settings, keep_paths, replace=None):
    """Copy a Modelfile, keeping only ``keep_paths`` as sources and pinning the settings.

    ``replace`` maps an old source path to a new one (used after requantizing).
    """
    replace = replace or {}
    lines = []
    for line in modelfile.splitlines():
        parts = line.split(None, 2)
        keyword = parts[0].upper() if parts else ""
        if keyword == "FROM" and len(parts) >= 2:
            path = line.split(None, 1)[1].strip()
            if path not in keep_paths:
                continue
            line = f"FROM {replace.get(path, path)}"
        elif keyword == "PARAMETER" and len(parts) >= 2 and parts[1] in SHU_PARAMETERS:
            continue
        lines.append(line)
    lines += parameter_lines(settings)
    return "\n".join(lines) + "\n"


def parameter_lines(settings):
    lines = []
    for key in SHU_PARAMETERS:
        if key in settings:
            value = settings[key]
            if isinstance(value, bool):
                value = "true" if value else "false"
            lines.append(f"PARAMETER {key} {value}")
    return lines
