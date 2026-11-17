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
"""Model resolution, pinned downloads and local checkpoint contracts."""

import dataclasses
import hashlib
import os
import types
from unittest import mock

import huggingface_hub
import huggingface_hub.utils
import pytest

from inworld.sava import registry


@pytest.fixture(name="weights")
def weights_file(tmp_path):
    path = os.path.join(tmp_path, "weights.safetensors")
    with open(path, "wb") as f:
        f.write(b"sava")
    return path


def test_resolve_returns_an_existing_path_as_is(weights):
    assert registry.resolve(weights) == weights


def test_resolve_downloads_a_registered_model_and_checks_its_sum(weights):
    release = registry.Release("inworld/sava", "sava.safetensors",
                               registry.sha256_file(weights))

    with mock.patch.dict(registry.MODELS, {"inworld/sava": release}), \
            mock.patch.object(huggingface_hub, "hf_hub_download", autospec=True,
                              return_value=weights) as download:
        path = registry.resolve("inworld/sava")

    download.assert_called_once_with(repo_id="inworld/sava",
                                     filename="sava.safetensors",
                                     revision=None)
    assert path == weights


def test_resolve_refuses_a_download_with_the_wrong_sum(weights):
    release = registry.Release("inworld/sava", "sava.safetensors", "0" * 64)

    with mock.patch.dict(registry.MODELS, {"inworld/sava": release}), \
            mock.patch.object(huggingface_hub, "hf_hub_download", autospec=True,
                              return_value=weights), \
            pytest.raises(ValueError, match="checksum"):
        registry.resolve("inworld/sava")


def test_resolve_downloads_an_hf_location_at_its_revision(weights):
    with mock.patch.object(huggingface_hub,
                           "hf_hub_download",
                           autospec=True,
                           return_value=weights) as download:
        path = registry.resolve("hf://org/repo/dir/file.safetensors@v1")

    download.assert_called_once_with(repo_id="org/repo",
                                     filename="dir/file.safetensors",
                                     revision="v1")
    assert path == weights


@pytest.mark.parametrize("source", [
    "hf://org/repo", "hf:///repo/file", "hf://org//file", "hf://org/repo/",
    "hf://org/repo/dir//file", "https://org/repo/file", "hf://org/repo/file@"
])
def test_parse_hf_needs_a_scheme_repo_file_and_nonempty_revision(source):
    with pytest.raises(ValueError):
        registry.parse_hf(source)


def test_resolve_names_what_it_accepts():
    with pytest.raises(FileNotFoundError, match="inworld/sava"):
        registry.resolve("no-such-model")


def test_sha256_of_a_file(weights):
    assert registry.sha256_file(weights) == hashlib.sha256(b"sava").hexdigest()


def test_a_registered_name_takes_precedence_over_a_local_path(
        weights, tmp_path, monkeypatch):
    local = tmp_path / "inworld" / "sava"
    local.parent.mkdir()
    local.write_bytes(b"local weights")
    monkeypatch.chdir(tmp_path)
    release = dataclasses.replace(registry.MODELS["inworld/sava"],
                                  sha256=registry.sha256_file(weights),
                                  revision="pinned")

    with mock.patch.dict(registry.MODELS, {"inworld/sava": release}), \
            mock.patch.object(huggingface_hub, "hf_hub_download",
                              autospec=True, return_value=weights) as download:
        downloaded = registry.resolve("inworld/sava")
        explicit = registry.resolve("./inworld/sava")

    assert downloaded == weights
    assert explicit == "./inworld/sava"
    download.assert_called_once_with(repo_id=release.repo_id,
                                     filename=release.filename,
                                     revision="pinned")


def test_an_unreleased_registry_entry_is_not_downloaded():
    release = dataclasses.replace(registry.MODELS["inworld/sava"], sha256=None)

    with mock.patch.dict(registry.MODELS, {"inworld/sava": release}), \
            mock.patch.object(huggingface_hub, "hf_hub_download",
                              autospec=True) as download, \
            pytest.raises(ValueError, match="not released yet"):
        registry.resolve("inworld/sava")

    download.assert_not_called()


@pytest.mark.parametrize("source", ["inworld/sava", "hf://org/repo/file"])
@pytest.mark.parametrize("error", [
    OSError("offline"),
    huggingface_hub.utils.HfHubHTTPError("missing weights",
                                         response=types.SimpleNamespace(
                                             headers={}, request=None))
])
def test_hub_errors_explain_how_to_load_local_weights(source, error):

    with mock.patch.object(huggingface_hub, "hf_hub_download", autospec=True,
                           side_effect=error), \
            pytest.raises(RuntimeError, match="by its path") as raised:
        registry.resolve(source)

    assert raised.value.__cause__ is error


def test_directories_are_not_checkpoint_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        registry.resolve(str(tmp_path))
