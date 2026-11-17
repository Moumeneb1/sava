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
"""Release checkpoint roundtrips, metadata and strict state loading."""

import os
from unittest import mock

import pytest
import safetensors.torch
import torch

import conftest
from inworld.sava import checkpoint
from inworld.sava import registry


def test_write_then_load_restores_the_model(model, tmp_path):
    path = os.path.join(tmp_path, "tiny.safetensors")
    pcm = conftest.pcm()

    checkpoint.write(path, model)
    loaded = checkpoint.load(path)

    with torch.no_grad():
        torch.testing.assert_close(loaded(pcm).turn, model(pcm).turn)
    assert loaded.cfg == model.cfg
    assert not loaded.training


def test_load_casts_to_the_requested_dtype(model, tmp_path):
    path = os.path.join(tmp_path, "tiny.safetensors")
    checkpoint.write(path, model)

    loaded = checkpoint.load(path, dtype=torch.bfloat16)

    assert loaded.final_norm.weight.dtype == torch.bfloat16


def test_read_rejects_a_foreign_safetensors_file(tmp_path):
    path = os.path.join(tmp_path, "other.safetensors")
    safetensors.torch.save_file({"weight": torch.zeros(2)}, path)

    with pytest.raises(ValueError, match="not a SAVA checkpoint"):
        checkpoint.read(path)


def test_read_rejects_a_checkpoint_without_configuration(tmp_path):
    path = os.path.join(tmp_path, "incomplete.safetensors")
    safetensors.torch.save_file({"weight": torch.zeros(2)},
                                path,
                                metadata={"format": "sava"})

    with pytest.raises(ValueError, match="lacks the 'config' entry"):
        checkpoint.read(path)


def test_build_is_strict_about_the_state(model):
    state = model.state_dict()
    state.pop("final_norm.weight")

    with pytest.raises(RuntimeError):
        checkpoint.build(model.cfg, state)


def test_load_resolves_the_source_through_the_registry(model, tmp_path):
    path = os.path.join(tmp_path, "tiny.safetensors")
    checkpoint.write(path, model)

    with mock.patch.object(registry,
                           "resolve",
                           autospec=True,
                           return_value=path) as resolve:
        loaded = checkpoint.load("inworld/sava")

    resolve.assert_called_once_with("inworld/sava")
    assert loaded.cfg == model.cfg
