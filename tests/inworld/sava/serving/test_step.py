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
"""A step of streams in their slots against the model's own streaming step."""

import torch

import conftest
from inworld.sava import config
from inworld.sava.serving import step

FRAME = config.FRAME_SAMPLES


def test_a_step_of_scattered_slots_equals_the_model_step(model):
    frames = 12  # past both attention windows
    audio = conftest.pcm(batch=2, frames=frames)
    state = step.init_state(model, 3, torch.device("cpu"))
    rows = step.Rows([2, 0], torch.device("cpu"))
    untouched = [buffer[1].clone() for buffer in step.buffers(state)]
    want_state = model.init_state(2, torch.device("cpu"), torch.float32)

    with torch.inference_mode():
        got = [
            step.step(model, state, audio[:, t * FRAME:(t + 1) * FRAME], rows)
            for t in range(frames)
        ]

    with torch.inference_mode():
        for t, (speech, turn) in enumerate(got):
            want, want_state = model.step(audio[:, t * FRAME:(t + 1) * FRAME],
                                          want_state)
            torch.testing.assert_close(speech,
                                       torch.sigmoid(want.lookahead[:, 0]),
                                       atol=1e-6,
                                       rtol=0)
            torch.testing.assert_close(turn,
                                       torch.sigmoid(want.turn[:, 0]),
                                       atol=1e-6,
                                       rtol=0)
    for buffer, before in zip(step.buffers(state), untouched):
        assert torch.equal(buffer[1], before)
