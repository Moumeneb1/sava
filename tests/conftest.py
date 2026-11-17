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
"""Shared fixtures: a tiny model whose windows 24-frame sequences cross, the
paths of the fixtures and the released weights, and the benchmark scripts on
the path."""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(1, os.path.join(ROOT, "bench", "scripts"))
sys.path.insert(2, os.path.join(ROOT, "docs"))

from inworld.sava import config  # pylint: disable=wrong-import-position
from inworld.sava import model as model_lib  # pylint: disable=wrong-import-position

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
RELEASED_WEIGHTS = os.environ.get(
    "SAVA_GOLDEN_MODEL", os.path.join(ROOT, "models", "sava.safetensors"))


def tiny_config(**overrides) -> config.SavaConfig:
    attention = config.AttentionConfig(num_heads=4,
                                       num_kv_heads=2,
                                       head_dim=8,
                                       sliding_window=8)
    branch = config.BranchConfig(hidden_size=16,
                                 ffn_size=32,
                                 attention=config.AttentionConfig(
                                     num_heads=2,
                                     num_kv_heads=1,
                                     head_dim=8,
                                     sliding_window=6))
    raw = {
        "hidden_size": 32,
        "ffn_size": 64,
        "layers": ("mamba", "attention", "mamba", "attention"),
        "mamba": config.Mamba2Config(head_dim=8, state_size=8, chunk_size=4),
        "attention": attention,
        "vad_branch": branch,
        "turn_branch": branch,
        "vad_embed_size": 4,
        "lookahead_frames": 2,
    }
    raw.update(overrides)
    return config.SavaConfig(**raw)


def tiny_model(**overrides) -> model_lib.SavaModel:
    torch.manual_seed(0)
    return model_lib.SavaModel(tiny_config(**overrides)).eval()


def pcm(batch: int = 2, frames: int = 24, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 0.1 * torch.randn(
        batch, frames * config.FRAME_SAMPLES, generator=generator)


@pytest.fixture
def model() -> model_lib.SavaModel:
    return tiny_model()


@pytest.fixture
def fixtures_dir() -> str:
    return FIXTURES


@pytest.fixture
def released_weights() -> str:
    if not os.path.exists(RELEASED_WEIGHTS):
        pytest.skip(f"released weights not found at {RELEASED_WEIGHTS} "
                    f"(set SAVA_GOLDEN_MODEL)")
    return RELEASED_WEIGHTS
