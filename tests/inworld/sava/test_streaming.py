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
"""Stream framing, input validation and whole-recording parity."""

import numpy as np
import pytest
import torch

import conftest
from inworld.sava import config
from inworld.sava import streaming


@pytest.fixture(name="audio")
def mono_samples():
    return conftest.pcm(batch=1, frames=12)[0].numpy()


def test_run_lines_the_curves_up_with_the_frames(model, audio):
    lag = model.lookahead_frames
    padded = np.concatenate(
        [audio, np.zeros(lag * config.FRAME_SAMPLES, np.float32)])
    with torch.no_grad():
        out = model(torch.from_numpy(padded)[None])

    curves = streaming.run(model, audio)

    assert curves.speech.shape == curves.turn.shape == (12,)
    assert curves.turns.shape == (12, 4)
    np.testing.assert_allclose(curves.speech,
                               torch.sigmoid(out.lookahead[0, lag:lag +
                                                           12]).numpy(),
                               atol=1e-6)
    np.testing.assert_allclose(curves.turn,
                               torch.sigmoid(out.turn[0, :12, 0]).numpy(),
                               atol=1e-6)
    np.testing.assert_allclose(curves.speech_on_time,
                               torch.sigmoid(out.speech[0, :12]).numpy(),
                               atol=1e-6)


def test_run_pads_a_partial_last_frame(model, audio):
    curves = streaming.run(model, audio[:-100])

    assert curves.speech.shape == (12,)


def test_run_without_lookahead_returns_the_on_time_output(audio):
    model = conftest.tiny_model(lookahead_frames=0)

    curves = streaming.run(model, audio)

    np.testing.assert_allclose(curves.speech, curves.speech_on_time)


@pytest.mark.parametrize("piece", [640, 1000, 7 * 640 + 13, 12 * 640])
def test_a_session_pushed_in_pieces_equals_the_run(model, audio, piece):
    expected = streaming.run(model, audio)
    session = streaming.Session(model)

    outputs = [
        session.push(audio[i:i + piece]) for i in range(0, audio.size, piece)
    ]
    outputs.append(session.flush())

    speech = np.concatenate([o.speech for o in outputs])
    turn = np.concatenate([o.turn for o in outputs])
    assert [o.first for o in outputs
           ] == list(np.cumsum([0] + [len(o) for o in outputs[:-1]]))
    assert session.frames == 12 + model.lookahead_frames
    np.testing.assert_allclose(speech[model.lookahead_frames:],
                               expected.speech,
                               atol=1e-5)
    np.testing.assert_allclose(turn[:12], expected.turn, atol=1e-5)
    np.testing.assert_allclose(turn[12:], expected.flush_turns[:, 0], atol=1e-5)


def test_a_session_emits_the_no_speech_floor_before_its_first_verdict(
        model, audio):
    session = streaming.Session(model)

    first = session.push(audio[:4 * config.FRAME_SAMPLES])

    assert first.first == 0
    np.testing.assert_allclose(first.speech[:model.lookahead_frames],
                               0.0,
                               atol=1e-12)
    assert first.speech[model.lookahead_frames:].min() > 1e-6


def test_a_short_push_returns_nothing_until_a_frame_is_complete(model, audio):
    session = streaming.Session(model)

    partial = session.push(audio[:100])
    rest = session.push(audio[100:config.FRAME_SAMPLES])

    assert len(partial) == 0
    assert len(rest) == 1
    assert rest.first == 0


def test_flush_closes_the_session(model, audio):
    session = streaming.Session(model)
    session.push(audio)

    session.flush()

    assert session.closed
    with pytest.raises(RuntimeError):
        session.push(audio)
    with pytest.raises(RuntimeError):
        session.flush()


def test_push_accepts_tensors(model, audio):
    session = streaming.Session(model)

    out = session.push(torch.from_numpy(audio))

    assert len(out) == 12


def test_num_frames_rounds_up():
    assert streaming.num_frames(0) == 0
    assert streaming.num_frames(640) == 1
    assert streaming.num_frames(641) == 2


@pytest.mark.parametrize("samples", [
    np.zeros((2, 640), dtype=np.float32),
    np.zeros((640, 2), dtype=np.float32),
    np.zeros(640, dtype=np.int16),
    np.zeros(640, dtype=np.complex64),
    torch.zeros(640, dtype=torch.int16),
    torch.zeros((1, 640)),
    np.array(0.0),
])
def test_invalid_push_is_rejected_without_consuming_pending_audio(
        model, audio, samples):
    session = streaming.Session(model)
    session.push(audio[:100])

    with pytest.raises(ValueError, match="mono floating-point"):
        session.push(samples)
    result = session.push(audio[100:640])

    expected = streaming.Session(model).push(audio[:640])
    assert result.first == 0
    assert session.frames == 1
    np.testing.assert_array_equal(result.speech, expected.speech)
    np.testing.assert_array_equal(result.turns, expected.turns)


@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float64, torch.bfloat16])
def test_push_accepts_floating_tensors_with_gradients(model, audio, dtype):
    samples = torch.tensor(audio, dtype=dtype, requires_grad=True)

    actual = streaming.Session(model).push(samples)
    expected = streaming.Session(model).push(samples.detach().float().numpy())

    np.testing.assert_array_equal(actual.speech, expected.speech)
    np.testing.assert_array_equal(actual.turns, expected.turns)


@pytest.mark.parametrize("samples", [
    np.zeros((640, 2), dtype=np.float32),
    np.zeros(640, dtype=np.int16),
])
def test_run_rejects_non_mono_or_integer_audio(model, samples):
    with pytest.raises(ValueError, match="mono floating-point"):
        streaming.run(model, samples)


@pytest.mark.parametrize("lookahead", [0, 2])
def test_run_on_empty_audio_returns_empty_curves(lookahead):
    model = conftest.tiny_model(lookahead_frames=lookahead)

    curves = streaming.run(model, np.zeros(0, dtype=np.float32))

    assert curves.speech.shape == curves.turn.shape == (0,)
    assert curves.speech_on_time.shape == (0,)
    assert curves.turns.shape == (0, len(model.cfg.turn_horizons))
    assert curves.flush_turns.shape == (0, len(model.cfg.turn_horizons))


def test_partial_frame_flush_matches_run(model, audio):
    samples = audio[:-100]
    session = streaming.Session(model)
    session.push(np.zeros(0, dtype=np.float32))

    outputs = [session.push(samples), session.flush()]
    expected = streaming.run(model, samples)

    speech = np.concatenate([output.speech for output in outputs])
    turns = np.concatenate([output.turns for output in outputs])
    np.testing.assert_allclose(speech[model.lookahead_frames:],
                               expected.speech,
                               atol=1e-5)
    np.testing.assert_allclose(turns[:len(expected.turn)],
                               expected.turns,
                               atol=1e-5)
