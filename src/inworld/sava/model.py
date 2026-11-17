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
"""The SAVA model: a streaming voice-activity and end-of-turn network.

``forward(pcm)`` runs a whole causal sequence; ``init_state`` and ``step``
run the same computation frame by frame with constant memory. Per 40 ms
frame the model emits three logits:

- ``speech``: P(speech at this frame), on time;
- ``lookahead``: P(speech at the frame ``lookahead_frames`` back), judged with
  the frames since (NO_SPEECH_LOGIT for a stream's first ``lookahead_frames``
  frames, which would judge frames before the stream);
- ``turn``: P(the turn has ended by now), one column per horizon.

The turn head is conditioned on the previous frame's on-time speech
probability, so it needs no lookahead and does not lag.
"""

import dataclasses

import torch
from torch.nn import functional

from inworld.sava import config
from inworld.sava import mel
from inworld.sava import modules

NO_SPEECH_LOGIT = -30.0


@dataclasses.dataclass
class Outputs:
    """Holds speech, lookahead and turn logits for each frame.

    Per-frame logits ``[B, T]`` (``turn``: ``[B, T, K]``, one column per
    horizon); ``lookahead`` is None for a model without lookahead."""
    speech: torch.Tensor
    turn: torch.Tensor
    lookahead: torch.Tensor | None = None


@dataclasses.dataclass
class StreamState:
    """Streaming caches, previous-frame speech probability and frame count."""

    frontend: torch.Tensor
    mixer: modules.MixerState
    trunk: list
    vad: modules.AttentionState
    turn: modules.AttentionState
    prev_speech: torch.Tensor
    frames: torch.Tensor


def _trunk_block(kind: str, cfg: config.SavaConfig) -> torch.nn.Module:
    if kind == "mamba":
        return modules.MambaBlock(cfg.hidden_size, cfg.ffn_size, cfg.mamba,
                                  cfg.rms_eps)
    return modules.AttentionBlock(cfg.hidden_size, cfg.ffn_size, cfg.attention,
                                  cfg.rms_eps)


class Branch(torch.nn.Module):
    """Down-projection, one attention block, RMSNorm and a linear readout."""

    def __init__(self, in_size: int, out_size: int, cfg: config.BranchConfig,
                 eps: float):
        super().__init__()
        self.down = torch.nn.Linear(in_size, cfg.hidden_size, bias=False)
        self.block = modules.AttentionBlock(cfg.hidden_size, cfg.ffn_size,
                                            cfg.attention, eps)
        self.out_norm = modules.RMSNorm(cfg.hidden_size, eps)
        self.readout = torch.nn.Linear(cfg.hidden_size, out_size)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.block(self.down(x)))

    def features_chunk(self, x: torch.Tensor, state: modules.AttentionState):
        hidden, state = self.block.forward_chunk(self.down(x), state)
        return self.out_norm(hidden), state


class SavaModel(torch.nn.Module):
    """Causal network with speech and end-of-turn prediction heads."""

    def __init__(self, cfg: config.SavaConfig):
        super().__init__()
        self.cfg = cfg
        eps = cfg.rms_eps
        self.embedder = mel.MelFrontend()
        self.mixer = modules.FrameMixer(cfg.hidden_size, cfg)
        self.trunk = torch.nn.ModuleList(
            _trunk_block(kind, cfg) for kind in cfg.layers)
        self.final_norm = modules.RMSNorm(cfg.hidden_size, eps)
        self.vad_norm = modules.RMSNorm(cfg.hidden_size, eps)
        self.vad_branch = Branch(cfg.hidden_size, 1, cfg.vad_branch, eps)
        self.vad_la_readout = (torch.nn.Linear(cfg.vad_branch.hidden_size, 1)
                               if cfg.lookahead_frames > 0 else None)
        self.turn_norm = modules.RMSNorm(cfg.hidden_size, eps)
        self.vad_embed = torch.nn.Linear(1, cfg.vad_embed_size)
        self.turn_branch = Branch(cfg.hidden_size + cfg.vad_embed_size,
                                  len(cfg.turn_horizons), cfg.turn_branch, eps)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 0.02):
        """Initializes linear layers and scales residual projections.

        Normal(0, std) Linears, the residual output projections (``wo``,
        ``w2``, ``out_proj``) scaled by 1 / sqrt(2 * depth), zero biases."""
        depth = len(self.trunk) + 2
        residual_std = std / (2 * depth)**0.5
        for name, module in self.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            last = name.rsplit(".", 1)[-1]
            module.weight.normal_(
                0.0, residual_std if last in ("wo", "w2", "out_proj") else std)
            if module.bias is not None:
                module.bias.zero_()

    @property
    def lookahead_frames(self) -> int:
        return self.cfg.lookahead_frames

    def _turn_input(self, hidden: torch.Tensor,
                    prev_speech: torch.Tensor) -> torch.Tensor:
        embed = self.vad_embed(
            prev_speech.unsqueeze(-1).to(self.vad_embed.weight.dtype))
        return torch.cat([self.turn_norm(hidden),
                          embed.to(hidden.dtype)],
                         dim=-1)

    def _heads(self,
               hidden: torch.Tensor,
               vad_state=None,
               turn_state=None,
               prev_speech: torch.Tensor | None = None):
        """Runs both prediction heads on whole sequences or streaming chunks.

        The two heads on trunk output ``hidden [B, T, C]``. Streaming when
        states are given (then ``T`` frames are processed in order inside the
        turn head); otherwise the previous frame's speech probability is the
        same forward's, shifted by one frame."""
        if vad_state is None:
            features = self.vad_branch.features(self.vad_norm(hidden))
        else:
            features, vad_state = self.vad_branch.features_chunk(
                self.vad_norm(hidden), vad_state)
        speech = self.vad_branch.readout(features).squeeze(-1)
        lookahead = None
        if self.vad_la_readout is not None:
            lookahead = self.vad_la_readout(features).squeeze(-1)
        signal = torch.sigmoid(speech.detach().float())
        if turn_state is None:
            prev = functional.pad(signal[:, :-1], (1, 0))
            turn = self.turn_branch.readout(
                self.turn_branch.features(self._turn_input(hidden, prev)))
            return speech, lookahead, turn, vad_state, turn_state, signal[:, -1]
        turns = []
        for t in range(hidden.shape[1]):
            turn_features, turn_state = self.turn_branch.features_chunk(
                self._turn_input(hidden[:, t:t + 1], prev_speech.unsqueeze(1)),
                turn_state)
            turns.append(self.turn_branch.readout(turn_features))
            prev_speech = signal[:, t]
        return speech, lookahead, torch.cat(
            turns, dim=1), vad_state, turn_state, prev_speech

    def forward(self, pcm: torch.Tensor) -> Outputs:
        """Runs a stream from its start on ``pcm [B, S]`` floats at 16 kHz.

        PCM samples lie in [-1, 1].
        """
        frames = modules.frame_audio(pcm)
        hidden = self.mixer(self.mixer.rows(self.embedder(frames), frames))
        for block in self.trunk:
            hidden = block(hidden)
        speech, lookahead, turn, _, _, _ = self._heads(self.final_norm(hidden))
        if lookahead is not None:
            lookahead = _mask_lookahead(lookahead, 0, self.cfg.lookahead_frames)
        return Outputs(speech=speech, turn=turn, lookahead=lookahead)

    def init_state(self, batch: int, device: torch.device,
                   dtype: torch.dtype) -> StreamState:
        return StreamState(
            frontend=self.embedder.init_state(batch, device),
            mixer=self.mixer.init_state(batch, device, dtype),
            trunk=[
                block.init_state(batch, device, dtype) for block in self.trunk
            ],
            vad=self.vad_branch.block.init_state(batch, device, dtype),
            turn=self.turn_branch.block.init_state(batch, device, dtype),
            prev_speech=torch.zeros(batch, device=device, dtype=torch.float32),
            frames=torch.zeros((), dtype=torch.long, device=device))

    def step(self, pcm: torch.Tensor,
             state: StreamState) -> tuple[Outputs, StreamState]:
        """Processes the next whole frames using the stream caches.

        The next whole frames of a stream: ``pcm [B, n * FRAME_SAMPLES]`` ->
        their outputs and the advanced state."""
        if pcm.shape[1] % config.FRAME_SAMPLES:
            raise ValueError(
                f"step() takes whole {config.FRAME_SAMPLES}-sample "
                f"frames, got {pcm.shape[1]} samples.")
        frames = modules.frame_audio(pcm)
        features, frontend = self.embedder.forward_chunk(frames, state.frontend)
        hidden, mixer = self.mixer.forward_chunk(
            self.mixer.rows(features, frames), state.mixer)
        trunk = []
        for block, block_state in zip(self.trunk, state.trunk, strict=True):
            hidden, block_state = block.forward_chunk(hidden, block_state)
            trunk.append(block_state)
        speech, lookahead, turn, vad, turn_state, prev = self._heads(
            self.final_norm(hidden), state.vad, state.turn, state.prev_speech)
        if lookahead is not None:
            lookahead = _mask_lookahead(lookahead, state.frames,
                                        self.cfg.lookahead_frames)
        state = StreamState(frontend=frontend,
                            mixer=mixer,
                            trunk=trunk,
                            vad=vad,
                            turn=turn_state,
                            prev_speech=prev,
                            frames=state.frames + frames.shape[1])
        return Outputs(speech=speech, turn=turn, lookahead=lookahead), state


def _mask_lookahead(logits: torch.Tensor, first_frame: int | torch.Tensor,
                    lookahead: int) -> torch.Tensor:
    """NO_SPEECH_LOGIT on the stream's first ``lookahead`` frames."""
    early = first_frame + torch.arange(logits.shape[1],
                                       device=logits.device) < lookahead
    return torch.where(early[None], torch.full_like(logits, NO_SPEECH_LOGIT),
                       logits)


def parameter_count(model: SavaModel) -> int:
    return sum(p.numel() for p in model.parameters())
