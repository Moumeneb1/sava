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
"""One frame of many streams, each at its own position, through one model.

The streams' state lives in slots: a ``model.StreamState`` of ``capacity``
rows, with a frame count per row. A step reads and writes the rows of the
streams it steps. It is the model's own streaming step but for attention,
whose cache is a ring of the window: a step writes the newest key over the
one the window drops, at each stream's own position.
"""

import dataclasses
import types

import torch

from inworld.sava import model as model_lib
from inworld.sava import modules


def init_state(model: model_lib.SavaModel, capacity: int,
               device: torch.device) -> model_lib.StreamState:
    """Zeroed slots for ``capacity`` streams, in float32 as a ``Session``
    keeps them, with a frame count per slot."""
    state = model.init_state(capacity, device, torch.float32)
    return dataclasses.replace(state,
                               frames=torch.zeros(capacity,
                                                  dtype=torch.long,
                                                  device=device))


def buffers(state: model_lib.StreamState) -> list[torch.Tensor]:
    """Every tensor of ``state`` that holds one row per slot."""
    tensors = [
        state.frontend, state.mixer.tail, state.prev_speech, state.frames
    ]
    for block in [*state.trunk, state.vad, state.turn]:
        if isinstance(block, modules.MambaState):
            tensors += [block.conv_tail, block.ssm]
        else:
            tensors += [block.keys, block.values]
    return tensors


class Rows:
    """The slots of one step, in its row order.

    Consecutive slots are read and written in place, others gathered and
    scattered."""

    def __init__(self, slots: list[int], device: torch.device) -> None:
        self.index = torch.tensor(slots, device=device)
        first = slots[0]
        consecutive = slots == list(range(first, first + len(slots)))
        self.span = slice(first, first + len(slots)) if consecutive else None

    def take(self, buffer: torch.Tensor) -> torch.Tensor:
        if self.span is not None:
            return buffer[self.span]
        return buffer.index_select(0, self.index)

    def put(self, buffer: torch.Tensor, value: torch.Tensor) -> None:
        if self.span is not None:
            buffer[self.span].copy_(value)
        else:
            buffer.index_copy_(0, self.index, value)


def step(
    model: model_lib.SavaModel,
    state: model_lib.StreamState,
    pcm: torch.Tensor,
    rows: Rows,
    kernels: types.ModuleType | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Steps one frame ``pcm [B, FRAME_SAMPLES]`` of the streams in ``rows``.

    ``kernels`` is the ``kernels`` module, to advance the Mamba-2 state and
    attend with Triton (CUDA graphs only), or None.

    Returns:
        The probabilities of speech, as a ``Session`` emits them, ``[B]``
        and of the turn having ended ``[B, K]``.
    """
    positions = rows.take(state.frames).clone()
    frames = pcm[:, None]
    features, frontend = model.embedder.forward_chunk(frames,
                                                      rows.take(state.frontend))
    rows.put(state.frontend, frontend)
    hidden, mixer = model.mixer.forward_chunk(
        model.mixer.rows(features, frames),
        modules.MixerState(rows.take(state.mixer.tail)))
    rows.put(state.mixer.tail, mixer.tail)
    for block, block_state in zip(model.trunk, state.trunk, strict=True):
        if isinstance(block, modules.MambaBlock):
            hidden = _mamba_block(block, block_state, hidden, rows, kernels)
        else:
            hidden = _attention_block(block, block_state, hidden, rows,
                                      positions, kernels)
    hidden = model.final_norm(hidden)
    vad = _branch(model.vad_branch, state.vad, model.vad_norm(hidden), rows,
                  positions, kernels)
    on_time = model.vad_branch.readout(vad)[:, 0, 0]
    speech = on_time
    if model.vad_la_readout is not None:
        speech = torch.where(
            positions < model.lookahead_frames,
            torch.full_like(on_time, model_lib.NO_SPEECH_LOGIT),
            model.vad_la_readout(vad)[:, 0, 0])
    turn_input = model._turn_input(  # pylint: disable=protected-access
        hidden,
        rows.take(state.prev_speech)[:, None])
    turn = model.turn_branch.readout(
        _branch(model.turn_branch, state.turn, turn_input, rows, positions,
                kernels))
    rows.put(state.prev_speech, torch.sigmoid(on_time.float()))
    rows.put(state.frames, positions + 1)
    return torch.sigmoid(speech.float()), torch.sigmoid(turn[:, 0].float())


def _branch(branch: model_lib.Branch, state: modules.AttentionState,
            x: torch.Tensor, rows: Rows, positions: torch.Tensor,
            kernels: types.ModuleType | None) -> torch.Tensor:
    return branch.out_norm(
        _attention_block(branch.block, state, branch.down(x), rows, positions,
                         kernels))


def _mamba_block(block: modules.MambaBlock, state: modules.MambaState,
                 x: torch.Tensor, rows: Rows,
                 kernels: types.ModuleType | None) -> torch.Tensor:
    # pylint: disable=protected-access
    mixer = block.mixer
    tail = rows.take(state.conv_tail)
    z, xs, log_decay, b, c, dt, raw = mixer._split(block.mixer_norm(x), tail)
    rows.put(state.conv_tail, modules.next_tail(raw, tail))
    y = _advance(state.ssm, rows, (xs * dt[..., None])[:, 0], log_decay[:, 0],
                 b[:, 0], c[:, 0], kernels)
    x = x + mixer._output(y[:, None], xs, z)
    return x + block.mlp(block.mlp_norm(x))


def _advance(ssm: torch.Tensor, rows: Rows, x: torch.Tensor,
             log_decay: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
             kernels: types.ModuleType | None) -> torch.Tensor:
    """``ssd.recurrent_step`` of the rows' state, written back: the same
    operations in the same order, in place rather than into a new state."""
    if kernels is not None and kernels.supports_advance(ssm.shape[-1]):
        return kernels.advance(ssm, rows.index, x, log_decay, b, c)
    state = rows.take(ssm)
    state.mul_(torch.exp(log_decay.float())[..., None, None])
    state.add_(x.float().unsqueeze(-1) * b.float().unsqueeze(-2))
    if rows.span is None:
        rows.put(ssm, state)
    return torch.einsum("bhpn,bhn->bhp", state, c.float()).to(x.dtype)


def _attention_block(block: modules.AttentionBlock,
                     state: modules.AttentionState, x: torch.Tensor, rows: Rows,
                     positions: torch.Tensor,
                     kernels: types.ModuleType | None) -> torch.Tensor:
    attention = block.attention
    cfg = attention.cfg
    batch = x.shape[0]
    q, k, v = attention.qkvr(block.attn_norm(x)).split(
        (attention.q_width, attention.kv_width, attention.kv_width), dim=-1)
    q = q.reshape(batch, 1, cfg.num_heads, cfg.head_dim)
    k = k.reshape(batch, 1, cfg.num_kv_heads, cfg.head_dim)
    v = v.reshape(batch, 1, cfg.num_kv_heads, cfg.head_dim)
    freqs = attention._freqs_on(x.device)  # pylint: disable=protected-access
    q, k = (_rotate(modules.rms_norm(t, norm, attention.eps), positions,
                    cfg.rope_theta, freqs)
            for t, norm in ((q, attention.q_norm), (k, attention.k_norm)))
    if kernels is not None and kernels.supports_attention(
            cfg.head_dim, cfg.sliding_window,
            cfg.num_heads // cfg.num_kv_heads):
        out = kernels.attend(state.keys, state.values, rows.index, positions,
                             q.contiguous(), k.contiguous(), v.contiguous(),
                             attention.scale)
    else:
        out = _attend(state, rows, positions, q, k, v, attention.scale)
    if cfg.xsa:
        out = modules.reject_own_value(out, v, cfg.xsa_alpha)
    x = x + attention.wo(out.reshape(batch, 1, -1))
    return x + block.mlp(block.mlp_norm(x))


def _rotate(x: torch.Tensor, positions: torch.Tensor, theta: float,
            freqs: torch.Tensor) -> torch.Tensor:
    """``modules.apply_rope`` of ``x [B, 1, H, D]`` at a position per row."""
    return modules.apply_rope(x.transpose(0, 1), positions, theta,
                              freqs).transpose(0, 1)


def _attend(state: modules.AttentionState, rows: Rows, positions: torch.Tensor,
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            scale: float) -> torch.Tensor:
    """Writes the frame's key and value into each row's ring and attends over
    the ring: the ``window`` newest positions, none before 0."""
    window = state.keys.shape[1]
    newest = positions % window
    state.keys[rows.index, newest] = k[:, 0].to(state.keys.dtype)
    state.values[rows.index, newest] = v[:, 0].to(state.values.dtype)
    keys, values = rows.take(state.keys), rows.take(state.values)
    ring = torch.arange(window, device=q.device)
    held = (positions[:, None] - ring[None]) % window <= positions[:, None]
    batch, _, heads, head_dim = q.shape
    kv_heads = keys.shape[2]
    qh = q.float().reshape(batch, kv_heads, heads // kv_heads, head_dim)
    kh = keys.permute(0, 2, 1, 3).float()
    vh = values.permute(0, 2, 1, 3).float()
    scores = (qh @ kh.transpose(-1, -2) * scale).masked_fill(
        ~held[:, None, None], float("-inf"))
    out = torch.softmax(scores, dim=-1) @ vh
    return out.reshape(batch, 1, heads, head_dim).to(q.dtype)
