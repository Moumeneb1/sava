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
"""Tests for the SAVA mel module."""

import pytest
import torch

from inworld.sava import config
from inworld.sava import mel
from inworld.sava import modules

_ATOL = 1e-4


def _whisper_log_mel(pcm: torch.Tensor) -> torch.Tensor:
    """Whisper's log-mel with centred windows and reflect padding, through
    torch.stft: ``[B, S] -> [B, 128, S // hop + 1]``."""
    window = torch.hann_window(config.MEL_FFT, periodic=True)
    stft = torch.stft(pcm,
                      config.MEL_FFT,
                      config.MEL_HOP,
                      window=window,
                      center=True,
                      pad_mode="reflect",
                      return_complex=True)
    return mel.log_compress(mel.mel_filterbank() @ stft.abs().pow(2))


@pytest.fixture(name="frames")
def audio_frames():
    torch.manual_seed(0)
    return modules.frame_audio(0.1 * torch.randn(2, 6 * config.FRAME_SAMPLES))


def test_filterbank_is_slaney_normalised_triangles():
    bank = mel.mel_filterbank()

    peaks = bank.argmax(dim=1)
    assert bank.shape == (config.MEL_BINS, config.MEL_FFT // 2 + 1)
    assert (bank >= 0).all()
    assert (peaks[1:] >= peaks[:-1]).all() and peaks[-1] > peaks[0]
    assert bank[0, 0] == 0.0 and bank[0, 1] > 0.0
    areas = bank.sum(dim=1) * (config.SAMPLE_RATE / config.MEL_FFT)
    assert areas.min() > 0.0
    torch.testing.assert_close(areas[84:], torch.ones(44), atol=0.05, rtol=0)


def test_mel_scale_roundtrips():
    hertz = torch.tensor([0.0, 250.0, 999.0, 1000.0, 4000.0, 8000.0])

    back = mel.mel_to_hertz(mel.hertz_to_mel(hertz))

    torch.testing.assert_close(back, hertz, atol=1e-6, rtol=1e-6)


def test_features_are_whisper_log_mel_with_windows_ending_at_each_hop(frames):
    # Hop j reads samples [160 j - 240, 160 j + 160): Whisper's centred window
    # j on the signal delayed by 40 samples. Hops 0 and 1 are left out (there
    # the centred window reads reflect padding, ours zeros).
    pcm = frames.reshape(2, -1)

    out = mel.MelFrontend()(frames)

    hops = out.reshape(2, 6 * config.MEL_HOPS_PER_FRAME,
                       config.MEL_BINS).transpose(1, 2)
    reference = _whisper_log_mel(torch.nn.functional.pad(pcm,
                                                         (40, 0)))[..., :24]
    assert out.shape == (2, 6, config.SavaConfig().mel_width)
    torch.testing.assert_close(hops[..., 2:],
                               reference[..., 2:],
                               atol=_ATOL,
                               rtol=_ATOL)


@pytest.mark.parametrize("chunks", [[1] * 6, [1, 3, 2], [6]])
def test_streams_exactly(frames, chunks):
    frontend = mel.MelFrontend()
    full = frontend(frames)

    state = frontend.init_state(2, frames.device)
    outs, start = [], 0
    for size in chunks:
        out, state = frontend.forward_chunk(frames[:, start:start + size],
                                            state)
        outs.append(out)
        start += size

    torch.testing.assert_close(torch.cat(outs, dim=1),
                               full,
                               atol=1e-5,
                               rtol=1e-5)


def test_is_causal(frames):
    frontend = mel.MelFrontend()
    changed = frames.clone()
    changed[:, 4] += 0.5

    out, out_changed = frontend(frames), frontend(changed)

    torch.testing.assert_close(out_changed[:, :4], out[:, :4])
    assert not torch.allclose(out_changed[:, 4], out[:, 4])


def test_stays_fp32_under_bf16_autocast(frames):
    reference = mel.MelFrontend()(frames)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = mel.MelFrontend()(frames)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, reference, atol=0, rtol=0)


def test_keeps_level_and_has_no_parameters(frames):
    # 20 dB quieter is -2 in log10, -0.5 after the (x + 4) / 4 scaling, in
    # every bin above the floor.
    frontend = mel.MelFrontend()

    quieter = frontend(frames * 0.1) - frontend(frames)

    assert not list(frontend.parameters())
    torch.testing.assert_close(quieter,
                               torch.full_like(quieter, -0.5),
                               atol=1e-4,
                               rtol=0)
