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
"""The `sava-doctor` readiness check.

No network and no released weights: the Hub client's cache lookup and
download are replaced, and ``--full`` runs a tiny checkpoint.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
from unittest import mock

import huggingface_hub
from huggingface_hub import errors as hub_errors
import pytest
import torch

import conftest
from inworld.sava import checkpoint
from inworld.sava import doctor
from inworld.sava import registry


@pytest.fixture(name="empty_cache")
def no_cached_weights():
    """No released weights in the Hugging Face cache."""
    with mock.patch.object(huggingface_hub,
                           "try_to_load_from_cache",
                           autospec=True,
                           return_value=None):
        yield


@pytest.fixture(name="tiny_checkpoint")
def tiny_checkpoint_file(tmp_path):
    """The tiny model as a release safetensors file."""
    path = str(tmp_path / "tiny.safetensors")
    checkpoint.write(path, conftest.tiny_model())
    return path


def _lines(capsys) -> list[str]:
    return capsys.readouterr().out.strip().splitlines()


def _http_error(cls, message: str = "401 Client Error"):
    """A Hub HTTP error, with the bare response its constructor reads."""
    return cls(message,
               response=types.SimpleNamespace(headers={}, request=None))


@pytest.mark.usefixtures("empty_cache")
def test_the_default_level_without_weights_is_ready_and_succeeds(capsys):

    status = doctor.main([])

    lines = _lines(capsys)
    assert status == 0
    assert lines[-1] == "Ready to install weights: run sava-doctor --full"
    assert "INFO  inworld/sava: not downloaded yet; run with --full" in lines
    assert "INFO  device: inference runs on cpu" in lines
    assert any(line.startswith("INFO  torch: torch ") for line in lines)
    assert not any(line.startswith("FAIL") for line in lines)


@pytest.mark.parametrize("argv, line", [
    ([], "INFO  device: inference runs on cpu, the default; --device cuda uses"
     " the GPU"),
    (["--device", "cpu"], "INFO  device: inference runs on cpu"),
])
@pytest.mark.usefixtures("empty_cache")
def test_the_device_line_says_when_a_gpu_is_there_but_not_used(
        capsys, argv, line):

    with mock.patch.object(torch.cuda,
                           "is_available",
                           autospec=True,
                           return_value=True):
        doctor.main(argv)

    assert line in _lines(capsys)


@pytest.mark.parametrize("argv, count, message", [
    (["--device", "cuda"], 0, "cuda is not available on this machine"),
    (["--device", "cuda:1"
     ], 1, "cuda:1 does not exist: this machine has 1 CUDA device"),
])
@pytest.mark.usefixtures("empty_cache")
def test_a_cuda_device_this_machine_lacks_fails(capsys, argv, count, message):

    with mock.patch.object(torch.cuda, "is_available", autospec=True,
                           return_value=count > 0), \
            mock.patch.object(torch.cuda, "device_count", autospec=True,
                              return_value=count):
        status = doctor.main(argv)

    lines = _lines(capsys)
    assert status == 1
    assert f"FAIL  device: {message}: run with --device cpu" in lines
    assert lines[-1] == "1 problem: see the FAIL lines above"


@pytest.mark.usefixtures("empty_cache")
def test_an_mps_device_this_machine_lacks_fails(capsys):

    with mock.patch.object(torch.backends.mps,
                           "is_available",
                           autospec=True,
                           return_value=False):
        status = doctor.main(["--device", "mps"])

    assert status == 1
    assert ("FAIL  device: mps is not available on this machine: run with"
            " --device cpu") in _lines(capsys)


@pytest.mark.usefixtures("empty_cache")
def test_a_device_that_is_not_a_torch_device_fails(capsys):

    status = doctor.main(["--device", "gpu"])

    assert status == 1
    assert ("FAIL  device: 'gpu' is not a torch device: use cpu or cuda"
            in _lines(capsys))


@pytest.mark.usefixtures("empty_cache")
def test_full_does_not_run_the_model_on_a_device_that_failed(
        tiny_checkpoint, capsys):

    with mock.patch.object(torch.cuda,
                           "is_available",
                           autospec=True,
                           return_value=False):
        status = doctor.main(
            ["--full", "--model", tiny_checkpoint, "--device", "cuda"])

    lines = _lines(capsys)
    assert status == 1
    assert [line for line in lines if line.startswith("FAIL")] == [
        "FAIL  device: cuda is not available on this machine: run with"
        " --device cpu"
    ]
    assert not any(tiny_checkpoint in line for line in lines)


@pytest.mark.parametrize("spec, line", [
    (None, "INFO  microphone: sava mic needs the mic extra: pip install"
     " \"inworld-sava[mic]\""),
    (object(), "INFO  microphone: sava mic is available"),
])
@pytest.mark.usefixtures("empty_cache")
def test_the_microphone_line_follows_the_mic_extra(capsys, spec, line):

    with mock.patch.object(importlib.util,
                           "find_spec",
                           autospec=True,
                           return_value=spec):
        doctor.main([])

    assert line in _lines(capsys)


@pytest.mark.parametrize("matches", [True, False])
def test_cached_weights_are_checked_against_the_registry(
        tmp_path, capsys, matches):
    cached = tmp_path / "sava.safetensors"
    cached.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights" if matches else b"other").hexdigest()
    release = registry.Release("inworld/sava", "sava.safetensors", digest)

    with mock.patch.dict(registry.MODELS, {"inworld/sava": release}), \
            mock.patch.object(huggingface_hub, "try_to_load_from_cache",
                              autospec=True, return_value=str(cached)):
        status = doctor.main([])

    lines = _lines(capsys)
    if matches:
        assert status == 0
        assert f"OK    inworld/sava: cached at {cached}" in lines
        assert lines[-1] == "Ready: run sava-doctor --full to run the model"
    else:
        assert status == 1
        assert (f"FAIL  inworld/sava: checksum mismatch: delete {cached} and"
                " run again") in lines
        assert lines[-1] == "1 problem: see the FAIL lines above"


@pytest.mark.usefixtures("empty_cache")
def test_full_runs_the_given_checkpoint_and_is_all_set(tiny_checkpoint, capsys):

    status = doctor.main(["--full", "--model", tiny_checkpoint])

    lines = _lines(capsys)
    assert status == 0
    assert f"OK    {tiny_checkpoint}: loads and runs on cpu" in lines
    assert lines[-1] == (f"All set: {tiny_checkpoint} on cpu"
                         f" (torch {torch.__version__})")


def test_full_reads_the_cache_after_fetching_the_model(tmp_path, capsys):
    downloaded = tmp_path / "sava.safetensors"
    checkpoint.write(str(downloaded), conftest.tiny_model())
    release = registry.Release("inworld/sava", "sava.safetensors",
                               registry.sha256_file(str(downloaded)))
    cache = {}

    def fetch(repo_id, filename, revision=None):
        del repo_id, filename, revision
        cache["sava"] = str(downloaded)
        return str(downloaded)

    def lookup(repo_id, filename, revision=None):
        del repo_id, filename, revision
        return cache.get("sava")

    with mock.patch.dict(registry.MODELS, {"inworld/sava": release}), \
            mock.patch.object(huggingface_hub, "hf_hub_download",
                              autospec=True, side_effect=fetch), \
            mock.patch.object(huggingface_hub, "try_to_load_from_cache",
                              autospec=True, side_effect=lookup):
        status = doctor.main(["--full"])

    lines = _lines(capsys)
    assert status == 0
    assert lines[-3:] == [
        f"OK    inworld/sava: cached at {downloaded}",
        "OK    inworld/sava: loads and runs on cpu",
        f"All set: inworld/sava on cpu (torch {torch.__version__})",
    ]


@pytest.mark.parametrize("error, message", [
    (_http_error(hub_errors.RepositoryNotFoundError),
     "inworld/sava is not reachable: pass a local .safetensors file with"
     " --model"),
    (_http_error(hub_errors.GatedRepoError), "inworld/sava is not reachable"),
    (hub_errors.LocalEntryNotFoundError("connection error"),
     "cannot reach huggingface.co: check the network, or pass a local file"
     " with --model"),
])
@pytest.mark.usefixtures("empty_cache")
def test_full_names_why_the_released_weights_could_not_be_fetched(
        capsys, error, message):

    with mock.patch.object(huggingface_hub,
                           "hf_hub_download",
                           autospec=True,
                           side_effect=error):
        status = doctor.main(["--full"])

    lines = _lines(capsys)
    (failed,) = [line for line in lines if line.startswith("FAIL")]
    assert status == 1
    assert failed.startswith("FAIL  inworld/sava: ")
    assert message in failed
    assert lines[-1] == "1 problem: see the FAIL lines above"


def test_full_says_to_delete_a_download_with_the_wrong_checksum(
        tmp_path, capsys):
    downloaded = tmp_path / "sava.safetensors"
    downloaded.write_bytes(b"not the release")
    release = registry.Release("inworld/sava", "sava.safetensors", "0" * 64)

    with mock.patch.dict(registry.MODELS, {"inworld/sava": release}), \
            mock.patch.object(huggingface_hub, "hf_hub_download",
                              autospec=True, return_value=str(downloaded)), \
            mock.patch.object(huggingface_hub, "try_to_load_from_cache",
                              autospec=True, return_value=None):
        status = doctor.main(["--full"])

    assert status == 1
    assert ("FAIL  inworld/sava: checksum mismatch: delete the cached file and"
            " run again") in _lines(capsys)


@pytest.mark.parametrize("error, message", [
    (_http_error(hub_errors.RepositoryNotFoundError),
     "not reachable: check the location and your access to it, or pass a"
     " local .safetensors file with --model"),
    (hub_errors.LocalEntryNotFoundError("connection error"),
     "cannot reach huggingface.co: check the network, or pass a local file"
     " with --model"),
    (OSError("disk full"),
     "could not fetch hf://org/repo/w.safetensors: disk full"),
])
@pytest.mark.usefixtures("empty_cache")
def test_full_reports_an_hf_location_that_cannot_be_fetched(
        capsys, error, message):
    spec = "hf://org/repo/w.safetensors"

    with mock.patch.object(huggingface_hub,
                           "hf_hub_download",
                           autospec=True,
                           side_effect=error):
        status = doctor.main(["--full", "--model", spec])

    assert status == 1
    assert f"FAIL  {spec}: {message}" in _lines(capsys)


@pytest.mark.usefixtures("empty_cache")
def test_full_reports_a_local_file_that_does_not_exist(tmp_path, capsys):
    missing = str(tmp_path / "missing.safetensors")

    status = doctor.main(["--full", "--model", missing])

    (failed,) = [line for line in _lines(capsys) if line.startswith("FAIL")]
    assert status == 1
    assert failed.startswith(f"FAIL  {missing}: [{missing}] is neither a file")


@pytest.mark.usefixtures("empty_cache")
def test_full_reports_a_checkpoint_that_does_not_load(tmp_path, capsys):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a checkpoint")

    status = doctor.main(["--full", "--model", str(broken)])

    (failed,) = [line for line in _lines(capsys) if line.startswith("FAIL")]
    assert status == 1
    assert failed.startswith(f"FAIL  {broken}: failed on cpu: ")
    assert "try --device cpu" not in failed


@pytest.mark.usefixtures("empty_cache")
def test_json_carries_the_same_checks_and_summary(capsys):
    doctor.main([])
    text = _lines(capsys)

    status = doctor.main(["--json"])

    report = json.loads(capsys.readouterr().out)
    rows = [(c["level"], c["name"], c["message"]) for c in report["checks"]]
    assert status == 0
    assert report["ok"] is True
    assert report["summary"] == text[-1]
    assert [f"{level:<4}  {name}: {message}" for level, name, message in rows
           ] == text[:-1]


def test_importing_the_doctor_pulls_in_neither_the_hub_client_nor_the_network():
    code = textwrap.dedent("""
        import socket
        import sys
        from unittest import mock

        with mock.patch.object(socket.socket, 'connect', autospec=True,
                               side_effect=AssertionError('network on import')):
            from inworld.sava import doctor

        loaded = {'huggingface_hub', 'sounddevice'}.intersection(sys.modules)
        assert not loaded, loaded
    """)

    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True,
                         text=True,
                         check=True,
                         env={
                             **os.environ, "PYTHONPATH":
                                 os.pathsep.join(sys.path)
                         })

    assert not out.stdout.strip()
