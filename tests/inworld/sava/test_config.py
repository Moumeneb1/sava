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
"""Tests for the SAVA config module."""

import pytest

import conftest
from inworld.sava import config


def test_dict_roundtrip_keeps_every_field():
    cfg = conftest.tiny_config()

    restored = config.from_dict(config.to_dict(cfg))

    assert restored == cfg


def test_json_roundtrip_keeps_every_field():
    cfg = conftest.tiny_config()

    restored = config.from_json(config.to_json(cfg))

    assert restored == cfg


def test_partial_dict_keeps_the_defaults():
    cfg = config.from_dict({"attention": {"sliding_window": 256}})

    assert cfg.attention.sliding_window == 256
    assert cfg.attention.num_heads == config.AttentionConfig().num_heads
    assert cfg.vad_branch.attention.sliding_window == 512


def test_lists_become_tuples():
    cfg = config.from_dict({
        "layers": ["mamba", "attention"],
        "turn_horizons": [0, 5]
    })

    assert cfg.layers == ("mamba", "attention")
    assert cfg.turn_horizons == (0, 5)


@pytest.mark.parametrize("raw", [{
    "not_a_field": 1
}, {
    "mamba": {
        "not_a_field": 1
    }
}])
def test_unknown_keys_raise(raw):
    with pytest.raises(TypeError):
        config.from_dict(raw)


@pytest.mark.parametrize("field,value", [
    ("layers", ["mamba", "lstm"]),
    ("layers", []),
    ("turn_horizons", [5, 10]),
    ("turn_horizons", []),
    ("lookahead_frames", -1),
    ("mixer_context_frames", 0),
])
def test_invalid_values_raise(field, value):
    with pytest.raises(ValueError):
        config.from_dict({field: value})


def test_mel_width_is_the_hops_of_a_frame():
    cfg = config.SavaConfig()

    assert cfg.mel_width == config.MEL_BINS * (config.FRAME_SAMPLES //
                                               config.MEL_HOP)


def test_frame_is_40_ms():
    assert config.FRAME_SECONDS == pytest.approx(0.04)
