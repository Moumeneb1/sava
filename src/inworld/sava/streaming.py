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
"""Running the model over a stream, and over whole recordings as a stream.

A ``Session`` takes 16 kHz mono samples in pieces of any size and returns,
for every whole 40 ms frame it could step, the model's two probabilities at
the step they come out. The speech probability emitted at frame ``t`` judges
frame ``t - lag`` (``lag`` is the model's lookahead; 0 for an on-time model),
so a frame's speech verdict arrives ``lag`` frames after its audio and the
first ``lag`` values of a stream are the no-speech floor. ``flush()`` ends a
stream: it pads the last partial frame and steps ``lag`` frames of silence so
the last real frames get their verdicts. The turn probability is on time.

``run()`` gives the same curves for a whole recording, lined up per frame.
"""

import dataclasses
import math

import numpy as np
import torch

from inworld.sava import config
from inworld.sava import model as model_lib


@dataclasses.dataclass
class Frames:
    """One push's outputs for the frames it stepped.

    Attributes:
        first: Index of the first stepped frame in the stream.
        speech: ``[n]`` P(speech) as emitted: value ``i`` judges frame
            ``first + i - lag``.
        turn: ``[n]`` P(the turn has ended by frame ``first + i``).
        turns: ``[n, K]`` the same for every horizon of the model.
    """
    first: int
    speech: np.ndarray
    turn: np.ndarray
    turns: np.ndarray

    def __len__(self) -> int:
        """Returns the number of frames in this push's output."""
        return self.speech.size


@dataclasses.dataclass
class Curves:
    """Per-frame probabilities of a whole recording.

    The curves have shape ``[T]`` each, frame
    ``t`` covering samples ``[640 t, 640 (t + 1))``: ``speech`` is the
    lookahead verdict on frame ``t`` (read ``lag`` frames later), ``turn`` the
    end-of-turn probability at ``t`` (horizon 0) and ``turns [T, K]`` all
    horizons; ``speech_on_time`` is the on-time speech output.
    ``flush_turns [lag, K]`` holds the end-of-turn outputs of the ``lag``
    silent steps a flushed session takes after the recording, while the last
    frames' speech verdicts come out.
    """
    speech: np.ndarray
    turn: np.ndarray
    turns: np.ndarray
    speech_on_time: np.ndarray
    flush_turns: np.ndarray


def _probabilities(out: model_lib.Outputs) -> tuple[np.ndarray, np.ndarray]:
    speech = out.speech if out.lookahead is None else out.lookahead
    return (torch.sigmoid(speech[0].float()).cpu().numpy(),
            torch.sigmoid(out.turn[0].float()).cpu().numpy())


def _mono_float(samples: np.ndarray | torch.Tensor) -> np.ndarray:
    """Validates mono floating-point samples and returns a float32 array."""
    if isinstance(samples, torch.Tensor):
        if samples.ndim != 1 or not samples.is_floating_point():
            raise ValueError("Expected mono floating-point samples [samples].")
        return samples.detach().to(device="cpu", dtype=torch.float32).numpy()
    samples = np.asarray(samples)
    if samples.ndim != 1 or not np.issubdtype(samples.dtype, np.floating):
        raise ValueError("Expected mono floating-point samples [samples].")
    return samples.astype(np.float32, copy=False)


class Session:
    """A stream through one model; not safe for concurrent pushes."""

    def __init__(self, model: model_lib.SavaModel) -> None:
        self.model = model
        self.lag = model.lookahead_frames
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self._state = model.init_state(1, self.device, torch.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._frames = 0
        self._closed = False

    @property
    def frames(self) -> int:
        """Frames stepped so far."""
        return self._frames

    @property
    def closed(self) -> bool:
        """Whether the session has been flushed."""
        return self._closed

    def push(self, samples: np.ndarray | torch.Tensor) -> Frames:
        """Appends samples; steps every whole frame now available.

        Raises:
            RuntimeError: After ``flush()``.
            ValueError: If samples are not a one-dimensional float array.
        """
        if self._closed:
            raise RuntimeError("The session is flushed; start a new one.")
        samples = _mono_float(samples)
        self._pending = np.concatenate([self._pending, samples])
        whole = self._pending.size - self._pending.size % config.FRAME_SAMPLES
        chunk, self._pending = self._pending[:whole], self._pending[whole:]
        return self._step(chunk)

    def flush(self) -> Frames:
        """Ends the stream and returns every remaining frame's verdict.

        Pads the last partial frame, then steps ``lag`` frames of silence.
        """
        if self._closed:
            raise RuntimeError("The session is already flushed.")
        tail = (-self._pending.size) % config.FRAME_SAMPLES
        if self._pending.size == 0:
            tail = 0
        chunk = np.concatenate([
            self._pending,
            np.zeros(tail + self.lag * config.FRAME_SAMPLES, dtype=np.float32)
        ])
        self._pending = np.zeros(0, dtype=np.float32)
        self._closed = True
        return self._step(chunk)

    def _step(self, chunk: np.ndarray) -> Frames:
        first = self._frames
        if chunk.size == 0:
            empty = np.zeros(0, dtype=np.float32)
            return Frames(
                first, empty, empty,
                np.zeros((0, len(self.model.cfg.turn_horizons)),
                         dtype=np.float32))
        pcm = torch.from_numpy(chunk)[None].to(self.device)
        with torch.inference_mode():
            out, self._state = self.model.step(pcm, self._state)
        speech, turns = _probabilities(out)
        self._frames += speech.size
        return Frames(first, speech, turns[:, 0], turns)


def num_frames(samples: int) -> int:
    """Returns the number of frames needed to cover the samples."""
    return math.ceil(samples / config.FRAME_SAMPLES)


def run(model: model_lib.SavaModel, audio: np.ndarray | torch.Tensor) -> Curves:
    """Returns whole-recording curves, as if a session were pushed and flushed.

    Audio is a one-dimensional NumPy array or tensor of float samples at 16 kHz.
    """
    audio = _mono_float(audio)
    frames = num_frames(audio.size)
    if frames == 0:
        empty = np.zeros(0, dtype=np.float32)
        no_turns = np.zeros((0, len(model.cfg.turn_horizons)), dtype=np.float32)
        return Curves(speech=empty,
                      turn=empty,
                      turns=no_turns,
                      speech_on_time=empty,
                      flush_turns=no_turns)
    lag = model.lookahead_frames
    padded = np.zeros((frames + lag) * config.FRAME_SAMPLES, dtype=np.float32)
    padded[:audio.size] = audio
    device = next(model.parameters()).device
    with torch.inference_mode():
        out = model(torch.from_numpy(padded)[None].to(device))
    speech, turns = _probabilities(out)
    on_time = torch.sigmoid(out.speech[0].float()).cpu().numpy()
    return Curves(speech=speech[lag:lag + frames],
                  turn=turns[:frames, 0],
                  turns=turns[:frames],
                  speech_on_time=on_time[:frames],
                  flush_turns=turns[frames:])
