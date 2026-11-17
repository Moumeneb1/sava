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
"""Building blocks of the SAVA model.

Every sequence module computes the same thing two ways: ``forward(x)`` over a
whole causal sequence ``[B, T, C]`` and ``forward_chunk(x, state)`` over the
next frames of a stream, returning the new state. A full forward equals the
stream of its frames.
"""

import dataclasses
import math

import torch
from torch.nn import functional

from inworld.sava import config
from inworld.sava import ssd


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None,
             eps: float) -> torch.Tensor:
    """Runs RMSNorm in fp32 and casts back to ``x.dtype``.

    ``weight`` None is scale-free.
    """
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


class RMSNorm(torch.nn.Module):

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


def shift_conv(x: torch.Tensor,
               weight: torch.Tensor,
               tail: torch.Tensor | None = None) -> torch.Tensor:
    """Applies a depthwise causal convolution.

    Depthwise causal conv ``y_t = sum_k weight[:, k] * x_{t - (K-1) + k}``
    as K shifted multiply-adds in fp32. ``x [B, T, C]``, ``weight [C, K]``,
    ``tail [B, K-1, C]`` the inputs before ``x`` (zeros when None)."""
    kernel = weight.shape[1]
    if tail is None:
        tail = x.new_zeros(x.shape[0], kernel - 1, x.shape[2])
    joined = torch.cat([tail.to(x.dtype), x], dim=1).float()
    seq_len = x.shape[1]
    w = weight.float()
    y = joined[:, 0:seq_len] * w[:, 0]
    for k in range(1, kernel):
        y = y + joined[:, k:k + seq_len] * w[:, k]
    return y


def next_tail(x: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """The last ``K-1`` inputs after appending ``x`` to ``tail``."""
    context = tail.shape[1]
    joined = torch.cat([tail.to(x.dtype), x], dim=1)
    return state_copy(joined[:, joined.shape[1] - context:])


def state_copy(x: torch.Tensor) -> torch.Tensor:
    """Copies a slice kept as streaming state into a tensor of its own.

    The copy has the default strides of a freshly allocated state, which a
    view does not, even where a size-1 batch lets the view count as
    contiguous; so a compiled step's stride guards hold at every frame."""
    return x.clone(memory_format=torch.contiguous_format)


class SwiGLU(torch.nn.Module):
    """SiLU-gated FFN; ``w13`` rows are [gate; up]."""

    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.w13 = torch.nn.Linear(hidden_size, 2 * ffn_size, bias=False)
        self.w2 = torch.nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.w2((functional.silu(gate.float()) * up.float()).to(x.dtype))


def rope_frequencies(head_dim: int, theta: float,
                     device: torch.device) -> torch.Tensor:
    """Returns the fp32 rotary frequencies of a ``head_dim``-wide head."""
    half = head_dim // 2
    return theta**(-torch.arange(half, device=device, dtype=torch.float32) /
                   half)


def apply_rope(x: torch.Tensor,
               positions: torch.Tensor,
               theta: float,
               freqs: torch.Tensor | None = None) -> torch.Tensor:
    """Applies rotary positions to the query or key vectors.

    Rotary embedding (rotate-half) of ``x [B, T, H, D]`` at absolute
    ``positions [T]``, in fp32. ``freqs`` are ``rope_frequencies`` for ``x``,
    computed when None."""
    half = x.shape[-1] // 2
    if freqs is None:
        freqs = rope_frequencies(x.shape[-1], theta, x.device)
    angles = positions.float()[:, None] * freqs[None]
    cos, sin = angles.cos()[None, :, None], angles.sin()[None, :, None]
    x1, x2 = x.float()[..., :half], x.float()[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin],
                     dim=-1).to(x.dtype)


def reject_own_value(out: torch.Tensor, values: torch.Tensor,
                     alpha: float) -> torch.Tensor:
    """Removes each attention head’s own-value component.

    Exclusive self attention: removes from each head's output
    ``out [B, T, Hq, D]`` ``alpha`` times its projection on the frame's own
    value ``values [B, T, Hkv, D]`` (expanded to the query heads), in fp32."""
    batch, seq_len, heads, head_dim = out.shape
    kv_heads = values.shape[2]
    v = values.float().unsqueeze(3)
    z = out.float().reshape(batch, seq_len, kv_heads, heads // kv_heads,
                            head_dim)
    dot = (z * v).sum(-1, keepdim=True)
    norm = (v * v).sum(-1, keepdim=True).clamp_min(1e-12)
    rejected = z - alpha * (dot / norm) * v
    return rejected.reshape(out.shape).to(out.dtype)


def windowed_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    query_start: int | torch.Tensor,
    key_start: int | torch.Tensor,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Computes causal sliding-window grouped-query attention in fp32.

    Causal sliding-window grouped-query attention in fp32 for queries at
    absolute positions ``query_start + i`` over keys at ``key_start + j``: a
    query reads the keys at most ``window - 1`` positions before it, and
    never a key at a negative position (an unfilled cache slot). Each key
    head serves its group of query heads without being copied."""
    batch, num_queries, heads, head_dim = q.shape
    kv_heads = keys.shape[2]
    group = heads // kv_heads
    qh = q.permute(0, 2, 1, 3).float().reshape(batch, kv_heads,
                                               group * num_queries, head_dim)
    kh = keys.permute(0, 2, 1, 3).float()
    vh = values.permute(0, 2, 1, 3).float()
    scores = qh @ kh.transpose(-1, -2) * scale
    query_positions = query_start + torch.arange(num_queries, device=q.device)
    key_positions = key_start + torch.arange(keys.shape[1], device=q.device)
    distance = query_positions.view(-1, 1) - key_positions.view(1, -1)
    keep = (distance >= 0) & (distance < window) & (key_positions >= 0)
    scores = scores.view(batch, kv_heads, group, num_queries, -1).masked_fill(
        ~keep, float("-inf")).view(batch, kv_heads, group * num_queries, -1)
    out = torch.softmax(scores, dim=-1) @ vh
    out = out.reshape(batch, heads, num_queries, head_dim)
    return out.permute(0, 2, 1, 3).to(q.dtype)


@dataclasses.dataclass
class AttentionState:
    """Caches the keys and values retained by sliding-window attention.

    Normed, rotated keys and values of the last ``window`` positions
    ``[B, W, Hkv, D]``, zero-filled before the stream's start, and the
    number of positions seen as a 0-d integer tensor. The fixed shapes keep
    a streaming step's graph the same at every frame."""
    keys: torch.Tensor
    values: torch.Tensor
    offset: torch.Tensor


class Attention(torch.nn.Module):
    """Runs sliding-window attention with rotary positions.

    Sliding-window GQA with RoPE, RMS-normed q / k and optional exclusive
    self attention. ``qkvr`` holds the query, key and value projections."""

    def __init__(self, hidden_size: int, cfg: config.AttentionConfig,
                 eps: float):
        super().__init__()
        self.cfg = cfg
        self.eps = eps
        self.q_width = cfg.num_heads * cfg.head_dim
        self.kv_width = cfg.num_kv_heads * cfg.head_dim
        self.scale = cfg.head_dim**-0.5
        self.qkvr = torch.nn.Linear(hidden_size,
                                    self.q_width + 2 * self.kv_width,
                                    bias=False)
        self.wo = torch.nn.Linear(self.q_width, hidden_size, bias=False)
        self.q_norm = torch.nn.Parameter(torch.ones(cfg.head_dim))
        self.k_norm = torch.nn.Parameter(torch.ones(cfg.head_dim))
        self._freqs: dict[torch.device, torch.Tensor] = {}

    def _freqs_on(self, device: torch.device) -> torch.Tensor:
        if device not in self._freqs:
            self._freqs[device] = rope_frequencies(self.cfg.head_dim,
                                                   self.cfg.rope_theta, device)
        return self._freqs[device]

    def init_state(self, batch: int, device: torch.device,
                   dtype: torch.dtype) -> AttentionState:
        self._freqs_on(torch.device(device))
        empty = torch.zeros(batch,
                            self.cfg.sliding_window,
                            self.cfg.num_kv_heads,
                            self.cfg.head_dim,
                            device=device,
                            dtype=dtype)
        return AttentionState(keys=empty,
                              values=empty.clone(),
                              offset=torch.zeros((),
                                                 dtype=torch.long,
                                                 device=device))

    def forward_chunk(
            self, x: torch.Tensor,
            state: AttentionState) -> tuple[torch.Tensor, AttentionState]:
        cfg = self.cfg
        batch, seq_len, _ = x.shape
        positions = state.offset + torch.arange(seq_len, device=x.device)
        freqs = self._freqs_on(x.device)
        q, k, v = self.qkvr(x).split(
            (self.q_width, self.kv_width, self.kv_width), dim=-1)
        q = q.reshape(batch, seq_len, cfg.num_heads, cfg.head_dim)
        k = k.reshape(batch, seq_len, cfg.num_kv_heads, cfg.head_dim)
        v = v.reshape(batch, seq_len, cfg.num_kv_heads, cfg.head_dim)
        q = apply_rope(rms_norm(q, self.q_norm, self.eps), positions,
                       cfg.rope_theta, freqs)
        k = apply_rope(rms_norm(k, self.k_norm, self.eps), positions,
                       cfg.rope_theta, freqs)
        keys = torch.cat([state.keys, k], dim=1)
        values = torch.cat([state.values, v], dim=1)
        key_start = state.offset + seq_len - keys.shape[1]
        out = windowed_attention(q, keys, values, state.offset, key_start,
                                 cfg.sliding_window, self.scale)
        if cfg.xsa:
            out = reject_own_value(out, v, cfg.xsa_alpha)
        window = cfg.sliding_window
        state = AttentionState(keys=state_copy(keys[:, -window:]),
                               values=state_copy(values[:, -window:]),
                               offset=state.offset + seq_len)
        return self.wo(out.reshape(batch, seq_len, -1)), state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return stream_in_windows(self, x, self.cfg.sliding_window)


def stream_in_windows(module: torch.nn.Module, x: torch.Tensor,
                      window: int) -> torch.Tensor:
    """Runs a module in bounded windows from a fresh state.

    ``module.forward_chunk`` over ``x`` from a fresh state, ``window``
    frames at a time: a whole sequence at the cost of the stream. The cache
    starts empty rather than zero-filled, so the first window does not
    score ``window`` masked slots."""
    state = module.init_state(x.shape[0], x.device, x.dtype)
    state = dataclasses.replace(state,
                                keys=state.keys[:, :0],
                                values=state.values[:, :0])
    outs = []
    for start in range(0, x.shape[1], window):
        out, state = module.forward_chunk(x[:, start:start + window], state)
        outs.append(out)
    return torch.cat(outs, dim=1)


class AttentionBlock(torch.nn.Module):
    """Pre-norm attention and SwiGLU, each with a residual."""

    def __init__(self, hidden_size: int, ffn_size: int,
                 cfg: config.AttentionConfig, eps: float):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(hidden_size, eps)
        self.attention = Attention(hidden_size, cfg, eps)
        self.mlp_norm = RMSNorm(hidden_size, eps)
        self.mlp = SwiGLU(hidden_size, ffn_size)

    def init_state(self, batch: int, device: torch.device,
                   dtype: torch.dtype) -> AttentionState:
        return self.attention.init_state(batch, device, dtype)

    def forward_chunk(
            self, x: torch.Tensor,
            state: AttentionState) -> tuple[torch.Tensor, AttentionState]:
        attended, state = self.attention.forward_chunk(self.attn_norm(x), state)
        x = x + attended
        return x + self.mlp(self.mlp_norm(x)), state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return stream_in_windows(self, x, self.cfg.sliding_window)


@dataclasses.dataclass
class MambaState:
    """Caches the causal convolution inputs and Mamba recurrence.

    The depthwise conv's last K-1 raw inputs ``[B, K-1, conv_dim]`` and the
    SSM state ``[B, H, P, N]`` (fp32)."""
    conv_tail: torch.Tensor
    ssm: torch.Tensor


class Mamba2Mixer(torch.nn.Module):
    """Runs the Mamba-2 state-space mixer.

    Mamba-2 (SSD) mixer: in_proj -> [z | xBC | dt], causal depthwise conv
    and SiLU on xBC, a scalar-decay SSM per head, gated RMSNorm, out_proj."""

    def __init__(self, hidden_size: int, cfg: config.Mamba2Config, eps: float):
        super().__init__()
        self.cfg = cfg
        self.eps = eps
        self.inner = cfg.expand * hidden_size
        if self.inner % cfg.head_dim:
            raise ValueError(f"expand * hidden ({self.inner}) must divide by "
                             f"head_dim ({cfg.head_dim}).")
        self.num_heads = self.inner // cfg.head_dim
        if self.num_heads % cfg.num_groups:
            raise ValueError("num_heads must divide by num_groups.")
        self.bc_width = cfg.num_groups * cfg.state_size
        self.conv_dim = self.inner + 2 * self.bc_width
        self.in_proj = torch.nn.Linear(hidden_size,
                                       self.inner + self.conv_dim +
                                       self.num_heads,
                                       bias=False)
        self.conv_weight = torch.nn.Parameter(
            torch.empty(self.conv_dim, cfg.conv_kernel))
        self.conv_bias = torch.nn.Parameter(torch.zeros(self.conv_dim))
        self.dt_bias = torch.nn.Parameter(torch.empty(self.num_heads))
        self.a_log = torch.nn.Parameter(torch.empty(self.num_heads))
        self.d_skip = torch.nn.Parameter(torch.ones(self.num_heads))
        self.norm = torch.nn.Parameter(torch.ones(self.inner))
        self.out_proj = torch.nn.Linear(self.inner, hidden_size, bias=False)
        self.reset_ssm_parameters()

    @torch.no_grad()
    def reset_ssm_parameters(self, dt_min: float = 1e-3, dt_max: float = 1e-1):
        """Initializes the state-space time steps and decay parameters.

        Mamba-2's init: dt log-uniform in [dt_min, dt_max] (the bias is
        its softplus inverse), A uniform in [1, 16], the conv as nn.Conv1d."""
        cfg = self.cfg
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min)) +
            math.log(dt_min))
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.a_log.copy_(
            torch.log(torch.empty(self.num_heads).uniform_(1.0, 16.0)))
        bound = 1.0 / math.sqrt(cfg.conv_kernel)
        self.conv_weight.uniform_(-bound, bound)
        self.conv_bias.uniform_(-bound, bound)

    def init_state(self, batch: int, device: torch.device,
                   dtype: torch.dtype) -> MambaState:
        return MambaState(conv_tail=torch.zeros(batch,
                                                self.cfg.conv_kernel - 1,
                                                self.conv_dim,
                                                device=device,
                                                dtype=dtype),
                          ssm=torch.zeros(batch,
                                          self.num_heads,
                                          self.cfg.head_dim,
                                          self.cfg.state_size,
                                          device=device,
                                          dtype=torch.float32))

    def _split(self, x: torch.Tensor, conv_tail: torch.Tensor | None):
        """Projects and convolves inputs for the state-space recurrence.

        in_proj and conv -> (z, x ``[B, T, H, P]``, log_decay ``[B, T, H]``,
        b, c ``[B, T, H, N]``, dt ``[B, T, H]``, raw xBC for the conv tail)."""
        cfg = self.cfg
        batch, seq_len, _ = x.shape
        z, xbc_raw, dt = self.in_proj(x).split(
            (self.inner, self.conv_dim, self.num_heads), dim=-1)
        xbc = functional.silu(
            shift_conv(xbc_raw, self.conv_weight, conv_tail) +
            self.conv_bias.float())
        xs, b, c = xbc.split((self.inner, self.bc_width, self.bc_width), dim=-1)
        dt = functional.softplus(dt.float() + self.dt_bias.float())
        log_decay = dt * -torch.exp(self.a_log.float())
        repeat = self.num_heads // cfg.num_groups

        def heads(t: torch.Tensor) -> torch.Tensor:
            t = t.reshape(batch, seq_len, cfg.num_groups, cfg.state_size)
            return t.repeat_interleave(repeat, dim=2)

        xs = xs.reshape(batch, seq_len, self.num_heads, cfg.head_dim)
        return z, xs, log_decay, heads(b), heads(c), dt, xbc_raw

    def _output(self, y: torch.Tensor, xs: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """Applies D skip, gating, grouped RMSNorm and output projection.

        Mamba-2 normalizes after gating.
        """
        batch, seq_len = y.shape[:2]
        y = y.float() + xs * self.d_skip.float()[:, None]
        y = y.reshape(batch, seq_len, self.inner) * functional.silu(z.float())
        group = self.inner // self.cfg.num_groups
        y = y.reshape(batch, seq_len, self.cfg.num_groups, group)
        y = rms_norm(y, None, self.eps).reshape(batch, seq_len, self.inner)
        return self.out_proj((y * self.norm.float()).to(z.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, xs, log_decay, b, c, dt, _ = self._split(x, None)
        y, _ = ssd.chunked_scan(xs * dt[..., None], log_decay, b, c,
                                self.cfg.chunk_size)
        return self._output(y, xs, z)

    def forward_chunk(self, x: torch.Tensor,
                      state: MambaState) -> tuple[torch.Tensor, MambaState]:
        z, xs, log_decay, b, c, dt, xbc_raw = self._split(x, state.conv_tail)
        conv_tail = next_tail(xbc_raw, state.conv_tail)
        if x.shape[1] == 1:
            y, ssm = ssd.recurrent_step((xs * dt[..., None])[:, 0],
                                        log_decay[:, 0], b[:, 0], c[:, 0],
                                        state.ssm)
            y = y.unsqueeze(1)
        else:
            y, ssm = ssd.chunked_scan(xs * dt[..., None], log_decay, b, c,
                                      self.cfg.chunk_size, state.ssm)
        return self._output(y, xs, z), MambaState(conv_tail=conv_tail, ssm=ssm)


class MambaBlock(torch.nn.Module):
    """Pre-norm Mamba-2 mixer and SwiGLU, each with a residual."""

    def __init__(self, hidden_size: int, ffn_size: int,
                 cfg: config.Mamba2Config, eps: float):
        super().__init__()
        self.mixer_norm = RMSNorm(hidden_size, eps)
        self.mixer = Mamba2Mixer(hidden_size, cfg, eps)
        self.mlp_norm = RMSNorm(hidden_size, eps)
        self.mlp = SwiGLU(hidden_size, ffn_size)

    def init_state(self, batch: int, device: torch.device,
                   dtype: torch.dtype) -> MambaState:
        return self.mixer.init_state(batch, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.mixer_norm(x))
        return x + self.mlp(self.mlp_norm(x))

    def forward_chunk(self, x: torch.Tensor,
                      state: MambaState) -> tuple[torch.Tensor, MambaState]:
        mixed, state = self.mixer.forward_chunk(self.mixer_norm(x), state)
        x = x + mixed
        return x + self.mlp(self.mlp_norm(x)), state


def frame_audio(pcm: torch.Tensor) -> torch.Tensor:
    """Splits ``[B, S]`` PCM into ``[B, ceil(S / 640), 640]`` frames.

    Zero-pads the last frame.
    """
    pad = (-pcm.shape[1]) % config.FRAME_SAMPLES
    if pad:
        pcm = functional.pad(pcm, (0, pad))
    return pcm.reshape(pcm.shape[0], -1, config.FRAME_SAMPLES)


@dataclasses.dataclass
class MixerState:
    """The previous mixer input rows, shaped ``[B, k-1, F]``.

    ``k`` is ``mixer_context_frames``.
    """
    tail: torch.Tensor


class FrameMixer(torch.nn.Module):
    """Mixes the current frame features with preceding frames.

    A causal conv over frame features (kernel ``mixer_context_frames``,
    stride one frame) as one GEMM over the concatenated context, then
    RMSNorm. A frame's row is its mel features plus its log RMS energy."""

    def __init__(self, hidden_size: int, cfg: config.SavaConfig):
        super().__init__()
        self.context = cfg.mixer_context_frames
        self.log_energy_floor = cfg.log_energy_floor
        self.row_width = cfg.mel_width + 1
        self.projection = torch.nn.Linear(self.row_width * self.context,
                                          hidden_size,
                                          bias=False)
        self.norm = RMSNorm(hidden_size, cfg.rms_eps)

    def rows(self, features: torch.Tensor,
             frames: torch.Tensor) -> torch.Tensor:
        """Appends each frame's log RMS, returning ``[B, T, row_width]``."""
        rms = frames.float().pow(2).mean(-1, keepdim=True).sqrt()
        log_energy = torch.log(rms.clamp(min=self.log_energy_floor))
        return torch.cat([features, log_energy.to(features.dtype)], dim=-1)

    def init_state(self, batch: int, device: torch.device,
                   dtype: torch.dtype) -> MixerState:
        return MixerState(tail=torch.zeros(
            batch, self.context -
            1, self.row_width, device=device, dtype=dtype))

    def _mix(self, rows: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([tail.to(rows.dtype), rows], dim=1)
        seq_len = rows.shape[1]
        stacked = torch.cat(
            [joined[:, k:k + seq_len] for k in range(self.context)], dim=-1)
        return self.norm(
            self.projection(stacked.to(self.projection.weight.dtype)))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        tail = rows.new_zeros(rows.shape[0], self.context - 1, rows.shape[2])
        return self._mix(rows, tail)

    def forward_chunk(self, rows: torch.Tensor,
                      state: MixerState) -> tuple[torch.Tensor, MixerState]:
        return self._mix(
            rows, state.tail), MixerState(tail=next_tail(rows, state.tail))
