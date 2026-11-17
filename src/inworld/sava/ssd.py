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
"""Mamba-2 state-space duality (SSD) in plain PyTorch.

Two evaluations of the same scalar-decay SSM per head,
``h_t = exp(dt_t * A) h_{t-1} + dt_t * B_t x_t^T``, ``y_t = C_t h_t``:

- ``chunked_scan`` over a whole sequence: within a chunk the scan is a masked
  matmul, across chunks a short recurrence carries the state. Every heavy op
  is an einsum, so it needs no custom kernel.
- ``recurrent_step``: one frame of the recurrence, for streaming.

Both run the decay math in fp32 and agree to rounding.
"""

import torch
from torch.nn import functional


def _segment_sum(x: torch.Tensor) -> torch.Tensor:
    """Computes lower-triangular cumulative segment sums.

    ``out[..., i, j] = sum(x[..., j+1 : i+1])`` for ``j <= i``, -inf above
    the diagonal: the log of the decay from position ``j`` to ``i``."""
    length = x.shape[-1]
    x = x.unsqueeze(-1).expand(*x.shape, length)
    strict = torch.ones(length, length, dtype=torch.bool,
                        device=x.device).tril(-1)
    x = x.masked_fill(~strict, 0.0)
    sums = torch.cumsum(x, dim=-2)
    lower = torch.ones(length, length, dtype=torch.bool,
                       device=x.device).tril(0)
    return sums.masked_fill(~lower, float("-inf"))


def chunked_scan(
    x: torch.Tensor,
    log_decay: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    chunk: int,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SSD over a full sequence.

    Args:
        x: ``[B, T, H, P]`` inputs already scaled by ``dt``.
        log_decay: ``[B, T, H]`` per-step log decay ``dt * A`` (<= 0).
        b: ``[B, T, H, N]`` input projections.
        c: ``[B, T, H, N]`` output projections.
        chunk: Chunk length; ``T`` is right-padded to a multiple of it.
        initial_state: Optional ``[B, H, P, N]`` state before position 0.

    Returns:
        ``y [B, T, H, P]`` in ``x.dtype`` and the final state ``[B, H, P, N]``
        (fp32), the state after the last unpadded position.
    """
    batch, seq_len, heads, head_dim = x.shape
    pad = (-seq_len) % chunk
    if pad:
        # Zero input and zero log-decay (decay 1) past the end: the state
        # passes through the padding unchanged, so the final state is exact.
        x = functional.pad(x, (0, 0, 0, 0, 0, pad))
        log_decay = functional.pad(log_decay, (0, 0, 0, pad))
        b = functional.pad(b, (0, 0, 0, 0, 0, pad))
        c = functional.pad(c, (0, 0, 0, 0, 0, pad))
    num_chunks = x.shape[1] // chunk
    out_dtype = x.dtype

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.float().reshape(batch, num_chunks, chunk, *t.shape[2:])

    xs, bs, cs = split(x), split(b), split(c)
    a = split(log_decay).permute(0, 3, 1, 2)  # [B, H, C, L]
    a_cumsum = torch.cumsum(a, dim=-1)

    decay_matrix = torch.exp(_segment_sum(a))  # [B, H, C, L, L]
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", cs, bs,
                          decay_matrix, xs)

    decay_to_end = torch.exp(a_cumsum[..., -1:] - a_cumsum)  # [B, H, C, L]
    chunk_states = torch.einsum("bclhn,bhcl,bclhp->bchpn", bs, decay_to_end, xs)

    if initial_state is None:
        initial_state = torch.zeros(batch,
                                    heads,
                                    head_dim,
                                    bs.shape[-1],
                                    device=x.device,
                                    dtype=torch.float32)
    states = torch.cat([initial_state.float().unsqueeze(1), chunk_states],
                       dim=1)
    chunk_decay = torch.exp(
        _segment_sum(functional.pad(a_cumsum[..., -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)
    carried, final_state = states[:, :-1], states[:, -1]

    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", cs, carried,
                         torch.exp(a_cumsum))
    y = (y_diag + y_off).reshape(batch, num_chunks * chunk, heads, head_dim)
    return y[:, :seq_len].to(out_dtype), final_state


def recurrent_step(
    x: torch.Tensor,
    log_decay: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advances the state-space recurrence by one frame.

    One frame: ``x [B, H, P]`` (dt-scaled), ``log_decay [B, H]``, ``b`` and
    ``c [B, H, N]``, ``state [B, H, P, N]`` fp32 -> (``y [B, H, P]`` in
    ``x.dtype``, the new state)."""
    decay = torch.exp(log_decay.float())[..., None, None]
    state = state * decay + x.float().unsqueeze(-1) * b.float().unsqueeze(-2)
    y = torch.einsum("bhpn,bhn->bhp", state, c.float())
    return y.to(x.dtype), state
