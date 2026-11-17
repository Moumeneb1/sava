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
"""The causal log-mel front end.

Whisper's log-mel (128 Slaney-normalised mel bins of a Hann-windowed 25 ms
DFT every 10 ms, log10 with a 1e-10 floor, then ``(x + 4) / 4``) with every
window ending where its hop ends, so a frame reads nothing after itself. A
40 ms frame is its four hops, hop-major. The DFT is one GEMM over the
windows and the bases are plain fp32 tensors, so the whole front end stays in
fp32
whatever the model's dtype. It has no parameters.
"""

import math

import torch
from torch.nn import functional

from inworld.sava import config

CONTEXT = config.MEL_FFT - config.MEL_HOP
_LOG_FLOOR = 1e-10


def hertz_to_mel(hertz: torch.Tensor) -> torch.Tensor:
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    log_step = math.log(6.4) / 27.0
    mel = 3.0 * hertz / 200.0
    above = hertz >= 1000.0
    return torch.where(above, 15.0 + torch.log(hertz / 1000.0) / log_step, mel)


def mel_to_hertz(mel: torch.Tensor) -> torch.Tensor:
    log_step = math.log(6.4) / 27.0
    hertz = 200.0 * mel / 3.0
    above = mel >= 15.0
    return torch.where(above, 1000.0 * torch.exp(log_step * (mel - 15.0)),
                       hertz)


def mel_filterbank(
    num_bins: int = config.MEL_BINS,
    n_fft: int = config.MEL_FFT,
    sample_rate: int = config.SAMPLE_RATE,
    max_frequency: float = 8000.0,
) -> torch.Tensor:
    """Builds triangular Slaney-normalized mel filters.

    Triangular Slaney-normalised mel filters ``[num_bins, n_fft // 2 + 1]``
    in fp32, as Whisper's feature extractor builds them."""
    num_frequencies = n_fft // 2 + 1
    fft_frequencies = torch.linspace(0.0,
                                     sample_rate / 2,
                                     num_frequencies,
                                     dtype=torch.float64)
    mel_points = torch.linspace(0.0,
                                float(hertz_to_mel(
                                    torch.tensor(max_frequency))),
                                num_bins + 2,
                                dtype=torch.float64)
    filter_frequencies = mel_to_hertz(mel_points)
    filter_diff = torch.diff(filter_frequencies)
    slopes = filter_frequencies[None, :] - fft_frequencies[:, None]
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    bank = torch.clamp(torch.minimum(down_slopes, up_slopes), min=0.0)
    bank = bank * (2.0 /
                   (filter_frequencies[2:] - filter_frequencies[:-2]))[None, :]
    return bank.T.float()


def dft_basis(n_fft: int = config.MEL_FFT) -> torch.Tensor:
    """Builds the Hann-windowed DFT as a matrix.

    ``[n_fft, 2 * (n_fft // 2 + 1)]``, to multiply windows ``[..., n_fft]``
    by: the cosine columns, then the sine columns."""
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
    k = torch.arange(n_fft, dtype=torch.float64)
    bins = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
    angle = -2.0 * math.pi * bins[:, None] * k[None, :] / n_fft
    basis = torch.cat([torch.cos(angle), torch.sin(angle)]) * window
    return basis.float().T.contiguous()


def log_compress(power: torch.Tensor) -> torch.Tensor:
    return (torch.log10(torch.clamp(power, min=_LOG_FLOOR)) + 4.0) / 4.0


class MelFrontend(torch.nn.Module):
    """Log-mel features per frame, causal.

    ``forward(frames)`` takes ``[B, T, FRAME_SAMPLES]`` and returns
    ``[B, T, mel_width]``; ``forward_chunk`` continues a stream from the
    ``CONTEXT`` samples before the chunk (zeros before the stream).
    """

    def __init__(self):
        super().__init__()
        self._bases: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """Returns zero context; also builds the bases on ``device``.

        Building them here keeps a compiled step from tracing their
        construction."""
        self._bases_on(torch.device(device))
        return torch.zeros(batch, CONTEXT, device=device, dtype=torch.float32)

    def _bases_on(self,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in self._bases:
            self._bases[device] = (dft_basis().to(device),
                                   mel_filterbank().T.contiguous().to(device))
        return self._bases[device]

    def features(self, pcm: torch.Tensor) -> torch.Tensor:
        """Returns ``[B, n, mel_width]`` features from fp32 PCM.

        ``pcm`` has shape ``[B, CONTEXT + n * FRAME_SAMPLES]``. The strided
        DFT is a GEMM over the windows as a view of ``pcm``, which avoids the
        CPU's slow single-channel convolution path.
        """
        dft, bank = self._bases_on(pcm.device)
        with torch.autocast(pcm.device.type, enabled=False):
            windows = pcm.float().unfold(-1, config.MEL_FFT, config.MEL_HOP)
            real, imag = torch.matmul(windows, dft).chunk(2, dim=-1)
            log_mel = log_compress(torch.matmul(real * real + imag * imag,
                                                bank))
        batch, hops, _ = log_mel.shape
        return log_mel.reshape(batch, hops // config.MEL_HOPS_PER_FRAME, -1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        pcm = frames.float().reshape(frames.shape[0], -1)
        return self.features(functional.pad(pcm, (CONTEXT, 0)))

    def forward_chunk(self, frames: torch.Tensor,
                      state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pcm = torch.cat(
            [state, frames.float().reshape(frames.shape[0], -1)], dim=1)
        # A copy, not a view, so the state keeps init_state's strides; see
        # modules.state_copy.
        return self.features(pcm), pcm[:, -CONTEXT:].clone(
            memory_format=torch.contiguous_format)
