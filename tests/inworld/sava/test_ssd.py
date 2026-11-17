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
"""Tests for the SAVA ssd module."""

import pytest
import torch

from inworld.sava import ssd


def _inputs(batch=2, seq_len=37, heads=4, head_dim=8, state=16, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    return (randn(batch, seq_len, heads, head_dim), -torch.rand(
        batch, seq_len, heads, generator=generator, dtype=torch.float64),
            randn(batch, seq_len, heads,
                  state), randn(batch, seq_len, heads, state))


def _recurrent(x, log_decay, b, c):
    batch, seq_len, heads, head_dim = x.shape
    state = torch.zeros(batch, heads, head_dim, b.shape[-1])
    ys = []
    for t in range(seq_len):
        y, state = ssd.recurrent_step(x[:, t], log_decay[:, t], b[:, t],
                                      c[:, t], state)
        ys.append(y)
    return torch.stack(ys, dim=1), state


@pytest.mark.parametrize("chunk", [1, 4, 8, 64])
def test_chunked_scan_matches_the_recurrence(chunk):
    x, a, b, c = _inputs()
    expected, expected_state = _recurrent(x, a, b, c)

    y, state = ssd.chunked_scan(x, a, b, c, chunk)

    torch.testing.assert_close(y, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(state.double(),
                               expected_state.double(),
                               atol=1e-4,
                               rtol=1e-4)


def test_scan_resumes_from_a_state():
    x, a, b, c = _inputs()
    full, final = ssd.chunked_scan(x, a, b, c, 8)

    first, state = ssd.chunked_scan(x[:, :20], a[:, :20], b[:, :20], c[:, :20],
                                    8)
    second, resumed = ssd.chunked_scan(x[:, 20:], a[:, 20:], b[:, 20:],
                                       c[:, 20:], 8, state)

    torch.testing.assert_close(torch.cat([first, second], dim=1),
                               full,
                               atol=1e-4,
                               rtol=1e-4)
    torch.testing.assert_close(resumed, final, atol=1e-4, rtol=1e-4)


def test_scan_keeps_the_input_dtype_and_an_fp32_state():
    x, a, b, c = (t.to(torch.bfloat16) for t in _inputs(seq_len=8))

    y, state = ssd.chunked_scan(x, a.float(), b, c, 4)

    assert y.dtype == torch.bfloat16
    assert state.dtype == torch.float32
