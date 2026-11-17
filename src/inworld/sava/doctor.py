# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`sava-doctor`: whether this machine is ready to run SAVA.

The default level reads only this machine: versions, the device inference
would use, whether the microphone extra is installed and whether the released
weights are already in the Hugging Face cache with the registered checksum.
``--full`` also fetches the model (or the given checkpoint) and runs half a
second of silence through it, as a whole recording and as a stream. Every line
carries a level, OK, INFO, WARN or FAIL, and the report ends with one summary
line; the exit status is 1 when any line failed. ``--json`` prints the same
report as one JSON object.

Running this module imports the package, so it cannot diagnose an install that
failed; it reports the versions an install produced. The package does not
import this module, and it imports the Hub client only when it needs it.
"""

import argparse
import dataclasses
import importlib.util
import json
import platform
import sys

import numpy as np
import torch

from inworld import sava
from inworld.sava import checkpoint
from inworld.sava import config
from inworld.sava import registry
from inworld.sava import streaming

OK, INFO, WARN, FAIL = "OK", "INFO", "WARN", "FAIL"
DEFAULT_MODEL = "inworld/sava"
# the CLI's and sava.load's default: SAVA runs on the CPU unless told otherwise
DEFAULT_DEVICE = "cpu"


@dataclasses.dataclass
class Check:
    """One line of the report."""
    level: str
    name: str
    message: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the doctor's options."""
    parser = argparse.ArgumentParser(
        prog="sava-doctor",
        description="Checks whether this machine is ready to run Inworld SAVA.")
    parser.add_argument(
        "--full",
        action="store_true",
        help=("Also fetch the model (downloading it on first use) and run half"
              " a second of silence through it."))
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=("With --full: the checkpoint, as a registered Hub repository, an"
              " hf:// location or a local file (default: the released"
              f" {DEFAULT_MODEL})."))
    parser.add_argument(
        "--device",
        default=None,
        help=f"torch device to check (default: {DEFAULT_DEVICE}, as the CLI).")
    parser.add_argument("--json",
                        action="store_true",
                        help="Print the report as one JSON object.")
    return parser.parse_args(argv)


def _environment() -> list[Check]:
    return [
        Check(INFO, "package", f"inworld-sava {sava.__version__}"),
        Check(INFO, "python",
              f"Python {platform.python_version()} on {platform.platform()}"),
        Check(INFO, "torch", f"torch {torch.__version__}"),
    ]


def _unavailable(device: torch.device) -> str | None:
    """Says why this machine cannot run on ``device``, or None when it can."""
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return "cuda is not available on this machine"
        count = torch.cuda.device_count()
        if device.index is not None and device.index >= count:
            return (f"{device} does not exist: this machine has {count} CUDA"
                    f" device{'' if count == 1 else 's'}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        return "mps is not available on this machine"
    return None


def _device_check(device: torch.device, chosen: bool) -> Check:
    """Reports the device; fails on one this machine does not have, and says
    when a GPU is there but not used."""
    reason = _unavailable(device)
    if reason is not None:
        return Check(FAIL, "device", f"{reason}: run with --device cpu")
    if not chosen and device.type == "cpu" and torch.cuda.is_available():
        return Check(
            INFO, "device",
            "inference runs on cpu, the default; --device cuda uses the GPU")
    return Check(INFO, "device", f"inference runs on {device}")


def _microphone_check() -> Check:
    if importlib.util.find_spec("sounddevice") is not None:
        return Check(INFO, "microphone", "sava mic is available")
    return Check(
        INFO, "microphone",
        'sava mic needs the mic extra: pip install "inworld-sava[mic]"')


def _cached_path(release: registry.Release) -> str | None:
    """The cached file of a release, or None when it is not cached."""
    # pylint: disable-next=import-outside-toplevel
    import huggingface_hub

    found = huggingface_hub.try_to_load_from_cache(release.repo_id,
                                                   release.filename,
                                                   revision=release.revision)
    return found if isinstance(found, str) else None


def _cache_checks() -> tuple[list[Check], bool]:
    """Checks each released model's cached file; also says whether all are."""
    checks, all_cached = [], True
    for name, release in registry.MODELS.items():
        if not release.sha256:
            checks.append(Check(INFO, name, "not released yet"))
            continue
        path = _cached_path(release)
        if path is None:
            all_cached = False
            checks.append(
                Check(INFO, name, "not downloaded yet; run with --full"))
        elif registry.sha256_file(path) != release.sha256:
            checks.append(
                Check(FAIL, name,
                      f"checksum mismatch: delete {path} and run again"))
        else:
            checks.append(Check(OK, name, f"cached at {path}"))
    return checks, all_cached


def _fetch_failure(spec: str, error: RuntimeError) -> str:
    """Says why ``registry.resolve`` could not fetch ``spec`` from the Hub.

    ``registry.resolve`` wraps the Hub client's error in a RuntimeError; the
    cause underneath tells a refused download from a missing connection.
    """
    # pylint: disable-next=import-outside-toplevel
    from huggingface_hub import errors as hub_errors

    cause = error.__cause__
    # what the Hub client raises when it has no connection and no cached copy
    if isinstance(cause, hub_errors.LocalEntryNotFoundError):
        return ("cannot reach huggingface.co: check the network, or pass a"
                " local file with --model")
    if isinstance(cause, hub_errors.HfHubHTTPError):
        if spec.startswith(registry.HF_SCHEME):
            return ("not reachable: check the location and your access to it,"
                    " or pass a local .safetensors file with --model")
        return (f"{spec} is not reachable: pass a local .safetensors file"
                " with --model")
    return f"could not fetch {spec}: {cause or error}"


def _resolve_failure(spec: str, error: ValueError) -> str:
    """Says why ``registry.resolve`` refused ``spec``."""
    if "registered checksum" in str(error) and spec in registry.MODELS:
        path = _cached_path(registry.MODELS[spec])
        return (f"checksum mismatch: delete {path or 'the cached file'} and"
                " run again")
    return str(error)


def _run_model(path: str, device: torch.device) -> None:
    """Loads a checkpoint and runs half a second of silence through it."""
    model = checkpoint.load(path, device)
    silence = np.zeros(config.SAMPLE_RATE // 2, dtype=np.float32)
    streaming.run(model, silence)
    session = streaming.Session(model)
    session.push(silence)
    session.flush()


def _model_check(spec: str, device: torch.device) -> Check:
    """Fetches the model and runs it; reports it under the name it was given."""
    try:
        path = registry.resolve(spec)
    except RuntimeError as error:
        return Check(FAIL, spec, _fetch_failure(spec, error))
    except ValueError as error:
        return Check(FAIL, spec, _resolve_failure(spec, error))
    except FileNotFoundError as error:
        return Check(FAIL, spec, str(error))
    try:
        _run_model(path, device)
    # any failure to load or run is reported as this model's FAIL line
    except Exception as error:  # pylint: disable=broad-exception-caught
        hint = "" if device.type == "cpu" else ": try --device cpu"
        return Check(FAIL, spec, f"failed on {device}: {error}{hint}")
    return Check(OK, spec, f"loads and runs on {device}")


def _summary(checks: list[Check], full: bool, all_cached: bool, name: str,
             device: torch.device | None) -> str:
    failed = sum(check.level == FAIL for check in checks)
    if failed:
        noun = "problem" if failed == 1 else "problems"
        return f"{failed} {noun}: see the FAIL lines above"
    if full:
        return f"All set: {name} on {device} (torch {torch.__version__})"
    if not all_cached:
        return "Ready to install weights: run sava-doctor --full"
    return "Ready: run sava-doctor --full to run the model"


def diagnose(args: argparse.Namespace) -> tuple[list[Check], str]:
    """Runs the checks the options ask for; returns them and the summary."""
    try:
        device = torch.device(args.device or DEFAULT_DEVICE)
    except RuntimeError:
        device = None
        device_check = Check(
            FAIL, "device",
            f"{args.device!r} is not a torch device: use cpu or cuda")
    else:
        device_check = _device_check(device, chosen=args.device is not None)
    # the model is not run when the device check failed; it is fetched
    # before the cache is read, so the cache lines show what the fetch left
    model = None
    if args.full and device_check.level != FAIL:
        model = _model_check(args.model, device)
    checks = _environment() + [device_check, _microphone_check()]
    cache, all_cached = _cache_checks()
    checks.extend(cache)
    if model is not None:
        checks.append(model)
    return checks, _summary(checks, args.full, all_cached, args.model, device)


def main(argv: list[str] | None = None) -> int:
    """Prints the report and returns 1 when any check failed, else 0."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    checks, summary = diagnose(args)
    failed = any(check.level == FAIL for check in checks)
    if args.json:
        print(
            json.dumps(
                {
                    "checks": [dataclasses.asdict(c) for c in checks],
                    "summary": summary,
                    "ok": not failed,
                },
                indent=2))
    else:
        for check in checks:
            print(f"{check.level:<4}  {check.name}: {check.message}")
        print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
