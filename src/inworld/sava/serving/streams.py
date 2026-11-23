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
"""Many streams through one model, stepped together.

A server with many live calls would step a ``Session`` per call, one small
step per call per 40 ms frame. ``Streams`` steps every call with a whole
frame ready as one batch. A stream may open, push or flush at any time, so
the streams in a step are at different frames; each stream's outputs equal
those of a ``Session`` given the same audio, to float32 rounding.
"""

from collections.abc import Iterable, Mapping
import dataclasses
import itertools

import numpy as np
import torch

from inworld.sava import config
from inworld.sava import model as model_lib
from inworld.sava import streaming
from inworld.sava.serving import graphs
from inworld.sava.serving import step


@dataclasses.dataclass
class _Stream:
    slot: int
    pending: np.ndarray
    frames: int = 0


class Streams:
    """Up to ``capacity`` streams of one model, stepped one frame at a time.

    ``open`` starts a stream and returns its id; ``push`` takes samples for
    any streams and returns, per stream, the frames it stepped, as
    ``Session.push`` would; ``flush`` ends streams as ``Session.flush`` does
    and frees their slots. ``cuda_graphs`` (CUDA only) replays each step of
    every open stream as a CUDA graph (``graphs``). Not safe for concurrent
    calls.

    Raises:
        ValueError: For a capacity below one, or ``cuda_graphs`` off CUDA.
    """

    def __init__(self,
                 model: model_lib.SavaModel,
                 capacity: int,
                 cuda_graphs: bool = False) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")
        self.model = model
        self.capacity = capacity
        self.lag = model.lookahead_frames
        self.device = next(model.parameters()).device
        if cuda_graphs and self.device.type != "cuda":
            raise ValueError("cuda_graphs needs the model on a CUDA device.")
        self._state = step.init_state(model, capacity, self.device)
        self._graphs = (graphs.Graphs(model, self._state, capacity)
                        if cuda_graphs else None)
        # open streams hold slots 0 to n - 1, so a step of all of them works
        # on their state in place and can replay a graph
        self._streams: dict[int, _Stream] = {}
        self._ids = itertools.count()

    @property
    def open_streams(self) -> list[int]:
        """The ids of the streams not yet flushed."""
        return list(self._streams)

    def open(self) -> int:
        """Starts a stream and returns its id.

        Raises:
            RuntimeError: When every slot holds a stream.
        """
        slot = len(self._streams)
        if slot == self.capacity:
            raise RuntimeError(
                f"All {self.capacity} slots hold a stream; flush one first.")
        for buffer in step.buffers(self._state):
            buffer[slot].zero_()
        stream = next(self._ids)
        self._streams[stream] = _Stream(slot, np.zeros(0, dtype=np.float32))
        return stream

    def push(
        self, pieces: Mapping[int, np.ndarray | torch.Tensor]
    ) -> dict[int, streaming.Frames]:
        """Appends samples to streams and steps every whole frame now ready.

        Args:
            pieces: Mono floating-point 16 kHz samples by stream id.

        Returns:
            The stepped frames of each stream in ``pieces``, as its
            ``Session.push`` would return them.

        Raises:
            KeyError: For an id that is not open.
            ValueError: If samples are not a one-dimensional float array.
        """
        # every id and piece is checked before any stream takes its piece
        streams = {stream_id: self._streams[stream_id] for stream_id in pieces}
        # pylint: disable-next=protected-access
        checked = {s: streaming._mono_float(p) for s, p in pieces.items()}
        chunks = {}
        for stream_id, samples in checked.items():
            stream = streams[stream_id]
            pending = np.concatenate([stream.pending, samples])
            whole = pending.size - pending.size % config.FRAME_SAMPLES
            chunks[stream_id], stream.pending = pending[:whole], pending[whole:]
        return self._run(chunks)

    def flush(self, stream_ids: Iterable[int]) -> dict[int, streaming.Frames]:
        """Ends streams as ``Session.flush`` does and frees their slots.

        Pads each stream's last partial frame and steps ``lag`` frames of
        silence, so its last frames get their verdicts.

        Raises:
            KeyError: For an id that is not open.
        """
        # every id is checked before any stream is ended
        streams = {
            stream_id: self._streams[stream_id]
            for stream_id in dict.fromkeys(stream_ids)
        }
        chunks = {}
        for stream_id, stream in streams.items():
            silence = ((-stream.pending.size) % config.FRAME_SAMPLES +
                       self.lag * config.FRAME_SAMPLES)
            chunks[stream_id] = np.concatenate(
                [stream.pending,
                 np.zeros(silence, dtype=np.float32)])
            stream.pending = np.zeros(0, dtype=np.float32)
        frames = self._run(chunks)
        for stream_id in chunks:
            self._release(self._streams.pop(stream_id).slot)
        return frames

    def _release(self, slot: int) -> None:
        """Frees a slot, moving the last open stream into it."""
        last = len(self._streams)
        if slot == last:
            return
        for buffer in step.buffers(self._state):
            buffer[slot].copy_(buffer[last])
        mover = next(s for s in self._streams.values() if s.slot == last)
        mover.slot = slot

    def _run(self, chunks: Mapping[int,
                                   np.ndarray]) -> dict[int, streaming.Frames]:
        """Steps every chunk's frames, a frame of every chunk per step."""
        counts = {s: c.size // config.FRAME_SAMPLES for s, c in chunks.items()}
        horizons = len(self.model.cfg.turn_horizons)
        speech = {s: np.zeros(n, dtype=np.float32) for s, n in counts.items()}
        turns = {
            s: np.zeros((n, horizons), dtype=np.float32)
            for s, n in counts.items()
        }
        for frame in range(max(counts.values(), default=0)):
            ids = sorted((s for s in chunks if counts[s] > frame),
                         key=lambda s: self._streams[s].slot)
            samples = slice(frame * config.FRAME_SAMPLES,
                            (frame + 1) * config.FRAME_SAMPLES)
            pcm = torch.from_numpy(np.stack([chunks[s][samples] for s in ids
                                            ])).to(self.device)
            with torch.inference_mode():
                frame_speech, frame_turns = self._step(
                    pcm, [self._streams[s].slot for s in ids])
            frame_speech = frame_speech.cpu().numpy()
            frame_turns = frame_turns.cpu().numpy()
            for row, stream_id in enumerate(ids):
                speech[stream_id][frame] = frame_speech[row]
                turns[stream_id][frame] = frame_turns[row]
        frames = {}
        for stream_id, count in counts.items():
            stream = self._streams[stream_id]
            frames[stream_id] = streaming.Frames(stream.frames,
                                                 speech[stream_id],
                                                 turns[stream_id][:, 0],
                                                 turns[stream_id])
            stream.frames += count
        return frames

    def _step(self, pcm: torch.Tensor,
              slots: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if self._graphs is not None and slots == list(range(len(
                self._streams))):
            return self._graphs.step(pcm)
        return step.step(self.model, self._state, pcm,
                         step.Rows(slots, self.device))
