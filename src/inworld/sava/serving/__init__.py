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
"""Serving many streams from one model.

``Streams`` steps every live stream as one batch per frame (``streams``).
Its parts: ``step``, one frame of the streams in a set of slots; ``graphs``,
that step replayed as CUDA graphs; ``kernels``, the Triton kernels the
recorded steps use for the Mamba-2 state and the attention ring.
"""

from inworld.sava.serving import streams

Streams = streams.Streams

__all__ = ["Streams"]
