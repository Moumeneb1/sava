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
"""Tests for the SAVA segmenter module."""

import numpy as np
import pytest

import conftest
from inworld.sava import segmenter
from inworld.sava import streaming

_LAG = 3
_POLICY = segmenter.Policy(pre_roll_ms=200,
                           max_silence_ms=400,
                           min_speech_ms=120,
                           min_confident_speech_ms=80)


def _frames(speech, turn, first=_LAG):
    speech, turn = np.asarray(speech, np.float32), np.asarray(turn, np.float32)
    return streaming.Frames(first=first,
                            speech=speech,
                            turn=turn,
                            turns=turn[:, None])


def _curves(speech, turn):
    speech, turn = np.asarray(speech, np.float32), np.asarray(turn, np.float32)
    # Frame t's verdict comes out with step t + _LAG's end-of-turn output.
    steps = np.concatenate([np.zeros(_LAG, np.float32), turn])[:, None]
    return streaming.Curves(speech=speech,
                            turn=steps[:speech.size, 0],
                            turns=steps[:speech.size],
                            speech_on_time=speech,
                            flush_turns=steps[speech.size:])


def test_a_turn_closes_when_the_model_calls_the_end():
    speech = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    turn = [0, 0, 0, 0, 0, 0, 0.5, 0.95, 0, 0]

    turns = segmenter.segment(_curves(speech, turn), _POLICY, _LAG)

    assert turns == [
        segmenter.Turn(start=0,
                       end=8 + _LAG,
                       speech_ms=160,
                       reason="turn",
                       kept=True)
    ]


def test_the_pre_roll_starts_the_turn_earlier():
    speech = [0] * 10 + [1] * 5 + [0] * 5
    turn = [0] * 15 + [0.95] + [0] * 4

    turns = segmenter.segment(_curves(speech, turn), _POLICY, _LAG)

    assert turns[0].start == 10 - 5
    assert turns[0].start_seconds == pytest.approx(0.2)


def test_silence_closes_a_turn_the_model_never_ends():
    speech = [1] * 4 + [0] * 12
    turn = [0.0] * 16

    turns = segmenter.segment(_curves(speech, turn), _POLICY, _LAG)

    assert len(turns) == 1
    assert turns[0].reason == "silence"
    assert turns[0].end == 4 + 10 + _LAG


def test_the_end_of_the_stream_closes_an_open_turn():
    speech = [0, 1, 1, 1]
    turn = [0.0] * 4

    turns = segmenter.segment(_curves(speech, turn), _POLICY, _LAG)

    assert turns[0].reason == "end"
    assert turns[0].end == 4


def test_short_turns_are_dropped_unless_confident():
    speech = [0.6, 0.6, 0, 0, 0, 0, 0, 0, 0, 0]
    turn = [0, 0, 0.95, 0, 0, 0, 0, 0, 0, 0]
    confident = [0.95, 0.95, 0, 0, 0, 0, 0, 0, 0, 0]

    dropped = segmenter.segment(_curves(speech, turn), _POLICY, _LAG)
    kept = segmenter.segment(_curves(confident, turn), _POLICY, _LAG)

    assert [t.kept for t in dropped] == [False]
    assert [t.kept for t in kept] == [True]


def test_a_minimum_silence_delays_the_check():
    policy = segmenter.Policy(pre_roll_ms=0,
                              min_silence_before_check_ms=120,
                              max_silence_ms=1600,
                              min_speech_ms=0)
    speech = [1, 1, 1, 0, 0, 0, 0, 0]
    turn = [0.95] * 8

    turns = segmenter.segment(_curves(speech, turn), policy, _LAG)

    assert turns[0].end == 3 + 3 + _LAG


def test_feeding_a_stream_piece_by_piece_equals_the_whole():
    speech = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    turn = [0, 0, 0, 0, 0, 0, 0, 0.95, 0, 0, 0, 0, 0, 0, 0, 0.95]
    whole = segmenter.segment(_curves(speech, turn), _POLICY, _LAG)
    cutter = segmenter.TurnSegmenter(_POLICY, _LAG)
    pieces = [(0, 5), (5, 6), (6, 16)]

    turns = []
    for start, end in pieces:
        turns += cutter.feed(
            _frames(speech[start:end], turn[start:end], _LAG + start))
    turns += cutter.finish()

    assert turns == whole
    assert len(turns) == 2


def test_a_frame_is_checked_with_the_turn_of_the_step_that_emits_its_verdict():
    speech = np.array([1, 1, 1, 0, 0, 0, 0, 0], np.float32)
    steps = np.array([0, 0, 0, 0.95, 0, 0, 0, 0, 0, 0, 0], np.float32)[:, None]
    curves = streaming.Curves(speech=speech,
                              turn=steps[:8, 0],
                              turns=steps[:8],
                              speech_on_time=speech,
                              flush_turns=steps[8:])

    turns = segmenter.segment(curves, _POLICY, _LAG)

    assert [t.reason for t in turns] == ["end"]


def test_frames_before_the_stream_are_skipped():
    cutter = segmenter.TurnSegmenter(_POLICY, _LAG)

    turns = cutter.feed(_frames([0, 0, 0, 1], [0, 0, 0, 0], first=0))

    assert not turns
    assert cutter.open


@pytest.mark.parametrize("piece", [640, 1000, 48 * 640])
def test_a_recording_is_cut_as_its_stream_is(model, piece):
    audio = conftest.pcm(batch=1, frames=48, seed=3)[0].numpy()
    curves = streaming.run(model, audio)
    # Thresholds from the tiny model's own outputs, so that turns open and
    # the model closes them.
    policy = segmenter.Policy(speech_threshold=float(np.median(curves.speech)),
                              turn_threshold=float(np.median(curves.turn)),
                              pre_roll_ms=0,
                              min_speech_ms=0,
                              min_confident_speech_ms=0,
                              max_silence_ms=10000)
    session = streaming.Session(model)
    cutter = segmenter.TurnSegmenter(policy, model.lookahead_frames)
    streamed = []
    for start in range(0, audio.size, piece):
        streamed += cutter.feed(session.push(audio[start:start + piece]))
    streamed += cutter.feed(session.flush()) + cutter.finish()

    whole = segmenter.segment(curves, policy, model.lookahead_frames)

    assert whole == streamed
    assert "turn" in [turn.reason for turn in whole]
