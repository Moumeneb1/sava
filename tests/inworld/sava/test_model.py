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
"""Tests for the SAVA model module."""

import pytest
import torch

import conftest
from inworld.sava import config
from inworld.sava import model as model_lib

_ATOL = 1e-4


def _stream(model, pcm, chunk_frames):
    state = model.init_state(pcm.shape[0], pcm.device, pcm.dtype)
    speech, lookahead, turn = [], [], []
    step = chunk_frames * config.FRAME_SAMPLES
    for start in range(0, pcm.shape[1], step):
        out, state = model.step(pcm[:, start:start + step], state)
        speech.append(out.speech)
        lookahead.append(
            out.lookahead if out.lookahead is not None else out.speech)
        turn.append(out.turn)
    return (torch.cat(speech, 1), torch.cat(lookahead, 1), torch.cat(turn,
                                                                     1)), state


def test_output_shapes(model):
    out = model(
        conftest.pcm(frames=24)[:, :-100])  # A partial last frame pads up.

    assert out.speech.shape == (2, 24)
    assert out.lookahead.shape == (2, 24)
    assert out.turn.shape == (2, 24, 4)


@pytest.mark.parametrize("chunk_frames", [1, 5, 24])
def test_streaming_matches_the_full_forward(model, chunk_frames):
    pcm = conftest.pcm()
    with torch.no_grad():
        full = model(pcm)

        (speech, lookahead, turn), _ = _stream(model, pcm, chunk_frames)

    torch.testing.assert_close(speech, full.speech, atol=_ATOL, rtol=_ATOL)
    torch.testing.assert_close(lookahead,
                               full.lookahead,
                               atol=_ATOL,
                               rtol=_ATOL)
    torch.testing.assert_close(turn, full.turn, atol=_ATOL, rtol=_ATOL)


def test_is_causal(model):
    pcm = conftest.pcm(batch=1)
    changed = pcm.clone()
    changed[:, 15 * config.FRAME_SAMPLES:] += 0.5

    with torch.no_grad():
        a, b = model(pcm), model(changed)

    torch.testing.assert_close(a.speech[:, :15], b.speech[:, :15])
    torch.testing.assert_close(a.lookahead[:, :15], b.lookahead[:, :15])
    torch.testing.assert_close(a.turn[:, :15], b.turn[:, :15])


def test_the_first_lookahead_frames_of_a_stream_are_no_speech(model):
    pcm = conftest.pcm(batch=1, frames=8)
    k = model.lookahead_frames

    with torch.no_grad():
        out = model(pcm)
        (_, streamed, _), state = _stream(model, pcm, 1)
        later, _ = model.step(pcm[:, :3 * config.FRAME_SAMPLES], state)

    assert k == 2
    assert (out.lookahead[:, :k] == model_lib.NO_SPEECH_LOGIT).all()
    assert (out.lookahead[:, k:] != model_lib.NO_SPEECH_LOGIT).all()
    torch.testing.assert_close(streamed, out.lookahead, atol=_ATOL, rtol=_ATOL)
    assert (later.lookahead != model_lib.NO_SPEECH_LOGIT).all()


def test_without_lookahead_the_output_is_none():
    model = conftest.tiny_model(lookahead_frames=0)

    with torch.no_grad():
        out = model(conftest.pcm(frames=8))
        (speech, _, _), _ = _stream(model, conftest.pcm(frames=8), 1)

    assert model.vad_la_readout is None
    assert out.lookahead is None
    torch.testing.assert_close(speech, out.speech, atol=_ATOL, rtol=_ATOL)


def test_the_turn_head_reads_the_previous_frames_speech(model):
    pcm = conftest.pcm(batch=1)
    state = model.init_state(1, pcm.device, pcm.dtype)

    with torch.no_grad():
        model.step(pcm[:, :10 * config.FRAME_SAMPLES], state)
        _, state = model.step(pcm[:, :10 * config.FRAME_SAMPLES], state)
        tenth, _ = model.step(
            pcm[:, 10 * config.FRAME_SAMPLES:11 * config.FRAME_SAMPLES], state)
        state.prev_speech = 1.0 - state.prev_speech
        flipped, _ = model.step(
            pcm[:, 10 * config.FRAME_SAMPLES:11 * config.FRAME_SAMPLES], state)

    torch.testing.assert_close(flipped.speech, tenth.speech)
    assert not torch.allclose(flipped.turn, tenth.turn)


def test_step_rejects_partial_frames(model):
    state = model.init_state(1, torch.device("cpu"), torch.float32)

    with pytest.raises(ValueError):
        model.step(torch.zeros(1, 700), state)


def test_streaming_state_keeps_its_shapes_from_the_first_frame(model):
    pcm = conftest.pcm(batch=1, frames=3)
    state = model.init_state(1, pcm.device, pcm.dtype)
    shapes = _state_shapes(state)

    with torch.no_grad():
        for start in range(0, pcm.shape[1], config.FRAME_SAMPLES):
            _, state = model.step(pcm[:, start:start + config.FRAME_SAMPLES],
                                  state)

    assert _state_shapes(state) == shapes
    assert state.frames == 3


def test_the_streaming_step_compiles_once_to_one_graph(model):
    pcm = conftest.pcm(batch=1, frames=4)
    graphs = []

    def count_graphs(graph, example_inputs):
        del example_inputs
        graphs.append(graph)
        return graph.forward

    try:
        step = torch.compile(model.step, backend=count_graphs, fullgraph=True)
    except RuntimeError as error:  # torch.compile unsupported on this Python.
        pytest.skip(str(error))
    state = model.init_state(1, pcm.device, pcm.dtype)

    with torch.no_grad():
        for start in range(0, pcm.shape[1], config.FRAME_SAMPLES):
            _, state = step(pcm[:, start:start + config.FRAME_SAMPLES], state)

    assert len(graphs) == 1


def _state_shapes(state):
    tensors = [
        state.frontend, state.mixer.tail, state.prev_speech, state.frames
    ]
    for cache in [state.vad, state.turn, *state.trunk]:
        tensors.extend(vars(cache).values())
    return [tuple(t.shape) for t in tensors]


def test_bf16_forward_is_finite(model):
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        out = model(conftest.pcm())

    assert torch.isfinite(out.speech.float()).all()
    assert torch.isfinite(out.turn.float()).all()


def test_released_topology_has_25_million_parameters():
    with torch.device("meta"):
        model = model_lib.SavaModel(config.SavaConfig())

    assert model_lib.parameter_count(model) == 25_521_558
