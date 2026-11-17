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
"""Configuration of the SAVA model.

Audio is 16 kHz, framed every 40 ms (640 samples). A causal log-mel front end
feeds a frame mixer, a trunk of Mamba-2 and sliding-window attention blocks,
and two heads: voice activity and end of turn. A checkpoint carries its own
configuration; the defaults are the released model's.
"""

import dataclasses
import json
import typing

SAMPLE_RATE = 16000
FRAME_SAMPLES = 640
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
MEL_BINS = 128
MEL_FFT = 400
MEL_HOP = 160
MEL_HOPS_PER_FRAME = FRAME_SAMPLES // MEL_HOP
LAYER_KINDS = ("mamba", "attention")


@dataclasses.dataclass
class Mamba2Config:
    expand: int = 2
    head_dim: int = 64
    state_size: int = 128
    num_groups: int = 1
    conv_kernel: int = 4
    chunk_size: int = 64


@dataclasses.dataclass
class AttentionConfig:
    """Configures causal sliding-window grouped-query attention.

    Causal sliding-window grouped-query attention with rotary positions,
    RMS-normed queries and keys and, optionally, exclusive self attention
    (the frame's own value is removed from each head's output)."""
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    sliding_window: int = 512
    rope_theta: float = 10000.0
    xsa: bool = True
    xsa_alpha: float = 1.0


@dataclasses.dataclass
class BranchConfig:
    """A head: a down-projection, one attention block and a linear readout."""
    hidden_size: int = 512
    ffn_size: int = 1536
    attention: AttentionConfig = dataclasses.field(
        default_factory=AttentionConfig)


@dataclasses.dataclass
class SavaConfig:
    """The model's topology.

    Attributes:
        layers: Trunk block kinds, bottom up.
        turn_horizons: Horizons of the turn head's columns, in frames; column
            zero answers "has the turn ended by now".
        vad_embed_size: Width of the embedding of the previous frame's speech
            probability, which conditions the turn head.
        lookahead_frames: Frames the speech output lags the audio: the value
            emitted at frame t judges frame t - lookahead_frames. Zero means an
            on-time speech output.
        mixer_context_frames: Frames the mixer concatenates for each step.
    """
    hidden_size: int = 512
    ffn_size: int = 2048
    rms_eps: float = 1e-6
    layers: tuple[str, ...] = ("mamba", "mamba", "mamba", "attention")
    mamba: Mamba2Config = dataclasses.field(default_factory=Mamba2Config)
    attention: AttentionConfig = dataclasses.field(
        default_factory=AttentionConfig)
    vad_branch: BranchConfig = dataclasses.field(default_factory=BranchConfig)
    turn_branch: BranchConfig = dataclasses.field(default_factory=BranchConfig)
    turn_horizons: tuple[int, ...] = (0, 5, 10, 20)
    vad_embed_size: int = 64
    lookahead_frames: int = 3
    mixer_context_frames: int = 2
    log_energy_floor: float = 1e-5

    def __post_init__(self):
        self.layers = tuple(self.layers)
        self.turn_horizons = tuple(self.turn_horizons)
        unknown = set(self.layers) - set(LAYER_KINDS)
        if unknown:
            raise ValueError(f"Unknown trunk layer kinds {sorted(unknown)}.")
        if not self.layers:
            raise ValueError("The trunk needs at least one layer.")
        if not self.turn_horizons or self.turn_horizons[0] != 0:
            raise ValueError("turn_horizons must start with horizon 0.")
        if self.lookahead_frames < 0:
            raise ValueError("lookahead_frames must be >= 0.")
        if self.mixer_context_frames < 1:
            raise ValueError("mixer_context_frames must be >= 1.")

    @property
    def mel_width(self) -> int:
        """Width of one frame's mel features: its hops, each of MEL_BINS."""
        return MEL_BINS * MEL_HOPS_PER_FRAME


_NESTED = {
    "mamba": Mamba2Config,
    "attention": AttentionConfig,
    "vad_branch": BranchConfig,
    "turn_branch": BranchConfig,
}


def _branch(raw: dict[str, typing.Any]) -> BranchConfig:
    raw = dict(raw)
    if "attention" in raw:
        raw["attention"] = AttentionConfig(**raw["attention"])
    return BranchConfig(**raw)


def from_dict(raw: dict[str, typing.Any]) -> SavaConfig:
    """A config from a nested dict, partial or complete.

    Raises:
        TypeError: On an unknown key.
        ValueError: On an invalid value.
    """
    raw = dict(raw)
    for key, cls in _NESTED.items():
        if key in raw:
            raw[key] = _branch(raw[key]) if cls is BranchConfig else cls(
                **raw[key])
    return SavaConfig(**raw)


def to_dict(config: SavaConfig) -> dict[str, typing.Any]:
    """Returns the configuration as nested JSON-ready values."""
    return dataclasses.asdict(config)


def from_json(text: str) -> SavaConfig:
    """Builds a configuration from its JSON representation."""
    return from_dict(json.loads(text))


def to_json(config: SavaConfig) -> str:
    """Returns the configuration's JSON representation."""
    return json.dumps(to_dict(config), indent=1)
