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
"""Steps of slots 0 to n - 1 replayed as CUDA graphs.

A step launches several hundred small kernels; replayed from a graph, it is
one launch. A graph is recorded per batch size, as an LLM server records its
decode steps, the first time a step needs it; a step of n rows replays the
smallest recorded size of at least n, its extra rows stepping free slots.
With Triton 3 the recorded steps use the ``kernels``: launched one by one,
outside a graph, they cost more than they save.
"""

import dataclasses

import torch

from inworld.sava import config
from inworld.sava import model as model_lib
from inworld.sava.serving import step

SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def _kernels():
    try:
        # pylint: disable-next=import-outside-toplevel
        from inworld.sava.serving import kernels
    except ImportError:
        return None
    return kernels


@dataclasses.dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    pcm: torch.Tensor
    outputs: tuple[torch.Tensor, torch.Tensor]
    # the graph reads the slot indices where they were when it was recorded
    rows: step.Rows


class Graphs:
    """The recorded steps of one model's slots, on its CUDA device."""

    def __init__(self, model: model_lib.SavaModel, state: model_lib.StreamState,
                 capacity: int) -> None:
        self.model = model
        self.state = state
        self.capacity = capacity
        self.device = state.frames.device
        self._kernels = _kernels()
        self._graphs: dict[int, _Graph] = {}

    def step(self, pcm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``step.step`` of ``pcm [n, FRAME_SAMPLES]`` in slots 0 to n - 1."""
        batch = pcm.shape[0]
        size = min(next((s for s in SIZES if s >= batch), self.capacity),
                   self.capacity)
        # the model's device, which need not be the current one
        with torch.cuda.device(self.device):
            if size not in self._graphs:
                self._graphs[size] = self._record(size)
            graph = self._graphs[size]
            graph.pcm[:batch].copy_(pcm)
            graph.pcm[batch:].zero_()
            graph.graph.replay()
        return graph.outputs[0][:batch], graph.outputs[1][:batch]

    def _record(self, size: int) -> _Graph:
        """Records the step of slots 0 to ``size`` - 1. The warm-up before it
        steps them, so their state is restored after."""
        rows = step.Rows(list(range(size)), self.device)
        pcm = torch.zeros(size, config.FRAME_SAMPLES, device=self.device)
        saved = [buffer[:size].clone() for buffer in step.buffers(self.state)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                step.step(self.model, self.state, pcm, rows, self._kernels)
        torch.cuda.current_stream().wait_stream(side)
        for buffer, value in zip(step.buffers(self.state), saved):
            buffer[:size].copy_(value)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = step.step(self.model, self.state, pcm, rows,
                                self._kernels)
        return _Graph(graph, pcm, outputs, rows)
