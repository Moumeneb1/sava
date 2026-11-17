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
"""Cuts a stream into turns from its speech and end-of-turn probabilities.

A reference policy that cuts a stream into turns from the model's two
probabilities. The benchmarks score the raw probabilities; this is the demo's
and the command line's way of turning them into segments.

A turn opens on the first frame judged speech and starts ``pre_roll_ms``
earlier, so the onset is kept. While it is open, silent frames count a pause.
The turn closes when the pause has lasted ``min_silence_before_check_ms`` and
the end-of-turn probability crosses ``turn_threshold``, or when the pause
reaches ``max_silence_ms`` whatever the model says. A closed turn is kept when
it held ``min_speech_ms`` of speech, or ``min_confident_speech_ms`` of speech
above ``confident_speech_threshold``; shorter ones are reported as dropped.

The speech verdicts arrive ``lag`` frames after their audio (the model's
lookahead); the policy indexes everything by the audio frame they judge, and
reads the end-of-turn probability of the frame being emitted, which is on
time. With a lookahead model ``min_silence_before_check_ms`` 0 checks the turn
one lag after the last speech frame.
"""

import dataclasses

import numpy as np

from inworld.sava import config
from inworld.sava import streaming


@dataclasses.dataclass(frozen=True)
class Policy:
    """Sets the speech gates and timing rules used to open and close turns."""

    speech_threshold: float = 0.5
    turn_threshold: float = 0.9
    pre_roll_ms: int = 800
    min_silence_before_check_ms: int = 0
    max_silence_ms: int = 1600
    min_speech_ms: int = 400
    min_confident_speech_ms: int = 200
    confident_speech_threshold: float = 0.9


@dataclasses.dataclass
class Turn:
    """Describes a closed turn with its frame span and decision.

    A closed turn: frames ``[start, end)`` of the stream's audio, pre-roll
    included; ``speech_ms`` the speech it held; ``reason`` how it closed
    ("turn", "silence" or "end"); ``kept`` whether it passed the gate.
    """
    start: int
    end: int
    speech_ms: int
    reason: str
    kept: bool

    @property
    def start_seconds(self) -> float:
        """Returns the inclusive start of the turn in seconds."""
        return self.start * config.FRAME_SECONDS

    @property
    def end_seconds(self) -> float:
        """Returns the exclusive end of the turn in seconds."""
        return self.end * config.FRAME_SECONDS


def _frames(ms: int) -> int:
    return round(ms / (1000 * config.FRAME_SECONDS))


class TurnSegmenter:
    """Feeds ``streaming.Frames`` in order and returns the turns they close."""

    def __init__(self, policy: Policy, lag: int) -> None:
        self.policy = policy
        self.lag = lag
        self._start: int | None = None
        self._speech_ms = 0
        self._confident_ms = 0
        self._pause_frames = 0

    @property
    def open(self) -> bool:
        """Whether a turn is waiting for its closing decision."""
        return self._start is not None

    def feed(self, frames: streaming.Frames) -> list[Turn]:
        """Consumes the next frames and returns turns closed by their scores."""
        closed = []
        for i in range(len(frames)):
            judged = frames.first + i - self.lag
            if judged < 0:
                continue
            turn = self._frame(judged, frames.first + i,
                               float(frames.speech[i]), float(frames.turn[i]))
            if turn is not None:
                closed.append(turn)
        return closed

    def finish(self) -> list[Turn]:
        """Closes an open turn at the end of the stream."""
        if self._start is None:
            return []
        return [self._close(self._last_judged + 1, "end")]

    def _frame(self, judged: int, now: int, speech: float,
               turn: float) -> Turn | None:
        policy = self.policy
        frame_ms = round(1000 * config.FRAME_SECONDS)
        self._last_judged = judged
        is_speech = speech >= policy.speech_threshold
        if self._start is None:
            if not is_speech:
                return None
            self._start = max(0, judged - _frames(policy.pre_roll_ms))
            self._speech_ms = self._confident_ms = 0
            self._pause_frames = 0
        if is_speech:
            self._pause_frames = 0
            self._speech_ms += frame_ms
            if speech >= policy.confident_speech_threshold:
                self._confident_ms += frame_ms
            return None
        self._pause_frames += 1
        pause_ms = self._pause_frames * frame_ms
        if pause_ms >= policy.max_silence_ms:
            return self._close(now + 1, "silence")
        checkable = pause_ms >= policy.min_silence_before_check_ms
        if checkable and turn >= policy.turn_threshold:
            return self._close(now + 1, "turn")
        return None

    def _close(self, end: int, reason: str) -> Turn:
        policy = self.policy
        kept = (self._speech_ms >= policy.min_speech_ms or
                self._confident_ms >= policy.min_confident_speech_ms)
        turn = Turn(start=self._start,
                    end=end,
                    speech_ms=self._speech_ms,
                    reason=reason,
                    kept=kept)
        self._start = None
        return turn


def segment(curves: streaming.Curves, policy: Policy, lag: int) -> list[Turn]:
    """The turns of a whole recording's curves, as the stream would cut them."""
    segmenter = TurnSegmenter(policy, lag)
    # The end-of-turn outputs of the steps that emit the verdicts: frame t's
    # comes out at step t + lag, the last frames' during the flush.
    steps = np.concatenate([curves.turns, curves.flush_turns])
    steps = steps[lag:lag + len(curves.speech)]
    frames = streaming.Frames(first=lag,
                              speech=np.asarray(curves.speech),
                              turn=steps[:, 0],
                              turns=steps)
    return segmenter.feed(frames) + segmenter.finish()
