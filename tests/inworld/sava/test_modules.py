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
"""Tests for the SAVA modules module."""

import dataclasses

import pytest
import torch
from torch.nn import functional

from inworld.sava import config
from inworld.sava import modules

_ATOL = 1e-4


def _stream(block, x, chunk_sizes):
    state = block.init_state(x.shape[0], x.device, x.dtype)
    outs, start = [], 0
    for size in chunk_sizes:
        out, state = block.forward_chunk(x[:, start:start + size], state)
        outs.append(out)
        start += size
    return torch.cat(outs, dim=1)


@pytest.fixture(name="attention_config")
def small_attention_config():
    return config.AttentionConfig(num_heads=4,
                                  num_kv_heads=2,
                                  head_dim=8,
                                  sliding_window=5)


@pytest.fixture(name="mamba_config")
def small_mamba_config():
    return config.Mamba2Config(head_dim=8, state_size=8, chunk_size=4)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_shift_conv_is_a_depthwise_causal_conv1d():
    x, weight = torch.randn(2, 11, 6), torch.randn(6, 4)
    expected = functional.conv1d(functional.pad(x.transpose(1, 2), (3, 0)),
                                 weight.unsqueeze(1),
                                 groups=6).transpose(1, 2)

    out = modules.shift_conv(x, weight)

    torch.testing.assert_close(out, expected)


def test_shift_conv_tail_continues_the_sequence():
    x, weight = torch.randn(2, 11, 6), torch.randn(6, 4)
    full = modules.shift_conv(x, weight)
    tail = x.new_zeros(2, 3, 6)

    first = modules.shift_conv(x[:, :5], weight, tail)
    second = modules.shift_conv(x[:, 5:], weight,
                                modules.next_tail(x[:, :5], tail))

    torch.testing.assert_close(torch.cat([first, second], dim=1), full)


def test_rope_scores_depend_on_relative_position_only():
    q, k = torch.randn(1, 1, 2, 8), torch.randn(1, 1, 2, 8)

    def score(pq, pk):
        rq = modules.apply_rope(q, torch.tensor([pq]), 10000.0)
        rk = modules.apply_rope(k, torch.tensor([pk]), 10000.0)
        return (rq * rk).sum(-1)

    torch.testing.assert_close(score(7, 3),
                               score(1007, 1003),
                               atol=1e-4,
                               rtol=1e-4)


def test_xsa_removes_the_own_value_component():
    out, values = torch.randn(2, 5, 4, 8), torch.randn(2, 5, 2, 8)
    own = values.repeat_interleave(2, dim=2)

    rejected = modules.reject_own_value(out, values, 1.0)

    torch.testing.assert_close((rejected * own).sum(-1),
                               torch.zeros(2, 5, 4),
                               atol=1e-5,
                               rtol=0)
    torch.testing.assert_close(modules.reject_own_value(out, values, 0.0), out)


def test_windowed_attention_masks_outside_the_window():
    q = torch.randn(1, 6, 2, 4)
    keys, values = torch.randn(1, 6, 1, 4), torch.randn(1, 6, 1, 4)
    changed = values.clone()
    # Key 0 is inside query 2's window of 3 and outside query 3's.
    changed[:, 0] += 5.0

    out = modules.windowed_attention(q, keys, values, 0, 0, 3, 0.5)
    out_changed = modules.windowed_attention(q, keys, changed, 0, 0, 3, 0.5)

    assert not torch.allclose(out[:, 2], out_changed[:, 2])
    torch.testing.assert_close(out[:, 3:], out_changed[:, 3:])
    torch.testing.assert_close(out[:, :1],
                               out_changed[:, :1] - 0.0,
                               atol=10,
                               rtol=0)


def test_attention_window_limits_the_context(attention_config):
    # The receptive field of one block is exactly the window: frame 0 reaches
    # frames 0..4 and nothing later.
    block = modules.AttentionBlock(16, 32, attention_config, 1e-6)
    x = torch.randn(1, 12, 16)
    changed = x.clone()
    changed[:, 0] += 5.0

    out, out_changed = block(x), block(changed)

    assert not torch.allclose(out[:, 4], out_changed[:, 4])
    torch.testing.assert_close(out[:, 5:], out_changed[:, 5:])


@pytest.mark.parametrize("chunks", [[1] * 17, [4, 4, 4, 5], [17], [3, 1, 9, 4]])
def test_attention_block_streams_exactly(attention_config, chunks):
    block = modules.AttentionBlock(16, 32, attention_config, 1e-6)
    x = torch.randn(2, 17, 16)

    streamed = _stream(block, x, chunks)

    torch.testing.assert_close(streamed, block(x), atol=_ATOL, rtol=_ATOL)


def test_xsa_changes_the_output_and_streams(attention_config):
    block = modules.AttentionBlock(16, 32, attention_config, 1e-6)
    x = torch.randn(2, 13, 16)
    without = modules.AttentionBlock(
        16, 32, dataclasses.replace(attention_config, xsa=False), 1e-6)
    without.load_state_dict(block.state_dict())

    out = block(x)

    assert not torch.allclose(out, without(x))
    torch.testing.assert_close(_stream(block, x, [1] * 13),
                               out,
                               atol=_ATOL,
                               rtol=_ATOL)


def test_attention_state_is_bounded_by_the_window(attention_config):
    block = modules.AttentionBlock(16, 32, attention_config, 1e-6)
    state = block.init_state(1, torch.device("cpu"), torch.float32)

    for _ in range(20):
        _, state = block.forward_chunk(torch.randn(1, 1, 16), state)

    assert state.keys.shape[1] == 5
    assert state.offset == 20


@pytest.mark.parametrize("chunks", [[1] * 19, [5, 5, 9], [19], [2, 1, 7, 9]])
def test_mamba_block_streams_exactly(mamba_config, chunks):
    block = modules.MambaBlock(16, 32, mamba_config, 1e-6)
    x = torch.randn(2, 19, 16)

    streamed = _stream(block, x, chunks)

    torch.testing.assert_close(streamed, block(x), atol=_ATOL, rtol=_ATOL)


def test_mamba_grouped_state_streams_exactly():
    cfg = config.Mamba2Config(head_dim=8,
                              state_size=8,
                              chunk_size=4,
                              num_groups=2)
    block = modules.MambaBlock(16, 32, cfg, 1e-6)
    x = torch.randn(1, 9, 16)

    streamed = _stream(block, x, [1] * 9)

    torch.testing.assert_close(streamed, block(x), atol=_ATOL, rtol=_ATOL)


def test_mamba_block_is_causal(mamba_config):
    block = modules.MambaBlock(16, 32, mamba_config, 1e-6)
    x = torch.randn(1, 10, 16)
    changed = x.clone()
    changed[:, 6:] += 3.0

    out, out_changed = block(x), block(changed)

    torch.testing.assert_close(out[:, :6], out_changed[:, :6])


def test_mamba_rejects_a_head_dim_that_does_not_divide():
    with pytest.raises(ValueError):
        modules.Mamba2Mixer(16, config.Mamba2Config(head_dim=12), 1e-6)


def test_frame_audio_pads_the_last_frame():
    frames = modules.frame_audio(torch.ones(2, 1300))

    assert frames.shape == (2, 3, 640)
    assert float(frames[:, 2, 20:].abs().sum()) == 0.0


def test_mixer_streams_exactly():
    cfg = config.SavaConfig(mixer_context_frames=3)
    mixer = modules.FrameMixer(16, cfg)
    frames = torch.randn(2, 7, config.FRAME_SAMPLES)
    rows = mixer.rows(torch.randn(2, 7, cfg.mel_width), frames)

    streamed = _stream(mixer, rows, [1] * 7)

    assert rows.shape == (2, 7, cfg.mel_width + 1)
    torch.testing.assert_close(streamed, mixer(rows))


def test_mixer_reads_the_previous_frame_only():
    cfg = config.SavaConfig(mixer_context_frames=2)
    mixer = modules.FrameMixer(16, cfg)
    rows = torch.randn(1, 4, cfg.mel_width + 1)
    changed = rows.clone()
    changed[:, 1] += 1.0

    out, out_changed = mixer(rows), mixer(changed)

    torch.testing.assert_close(out[:, 0], out_changed[:, 0])
    assert not torch.allclose(out[:, 2], out_changed[:, 2])
    torch.testing.assert_close(out[:, 3], out_changed[:, 3])


def test_log_energy_floors_silence():
    cfg = config.SavaConfig()
    mixer = modules.FrameMixer(16, cfg)
    frames = torch.zeros(1, 2, config.FRAME_SAMPLES)

    rows = mixer.rows(torch.zeros(1, 2, cfg.mel_width), frames)

    torch.testing.assert_close(
        rows[..., -1], torch.full((1, 2), float(torch.log(torch.tensor(1e-5)))))
