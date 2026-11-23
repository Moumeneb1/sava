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
"""Graphed streams against eager streams and each stream's own session, on
CUDA."""

import numpy as np
import pytest
import torch

import conftest
from inworld.sava import config
from inworld.sava import streaming
from inworld.sava.serving import streams

FRAME = config.FRAME_SAMPLES


def _run_plan(engine, audio, plan, skip=frozenset()):
    """Opens per ``plan`` and pushes each tick the frames due so far; a
    stream whose (index, tick) is in ``skip`` pushes nothing that tick, so
    its next push carries several frames. Flushes once all is pushed."""
    ids, got, sent = {}, {i: [] for i in range(len(plan))}, {}
    for tick in range(max(start + n for start, n in plan) + 4):
        for i, (start, _) in enumerate(plan):
            if tick == start:
                ids[i], sent[i] = engine.open(), 0
        pieces = {}
        for i in ids:
            due = min(tick - plan[i][0] + 1, plan[i][1])
            if (i, tick) not in skip and sent[i] < due:
                pieces[i] = audio[i][sent[i] * FRAME:due * FRAME]
                sent[i] = due
        pushed = engine.push({ids[i]: a for i, a in pieces.items()})
        for i in pieces:
            got[i].append(pushed[ids[i]])
        done = [i for i in ids if sent[i] == plan[i][1]]
        flushed = engine.flush([ids[i] for i in done])
        for i in done:
            got[i].append(flushed[ids.pop(i)])
    return {i: _joined(frames) for i, frames in got.items()}


def _joined(frames):
    return (np.concatenate([f.speech for f in frames]),
            np.concatenate([f.turns for f in frames]))


def _session(model, audio):
    session = streaming.Session(model)
    return _joined([session.push(audio), session.flush()])


def _audio(plan):
    return [
        conftest.pcm(batch=1, frames=n, seed=i)[0].numpy()
        for i, (_, n) in enumerate(plan)
    ]


def test_graphed_streams_equal_eager_streams_and_sessions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs run on CUDA")
    model = conftest.tiny_model().cuda()
    plan = [(0, 30), (2, 12), (5, 25), (9, 4)]
    audio = _audio(plan)
    # stream 2 skips ticks, so those ticks leave an open stream out
    skip = {(2, 7), (2, 8), (2, 15)}

    graphed = _run_plan(streams.Streams(model, capacity=4, cuda_graphs=True),
                        audio, plan, skip)

    eager = _run_plan(streams.Streams(model, capacity=4), audio, plan, skip)
    for i, a in enumerate(audio):
        np.testing.assert_array_equal(graphed[i][0], eager[i][0])
        np.testing.assert_array_equal(graphed[i][1], eager[i][1])
        np.testing.assert_allclose(graphed[i][0],
                                   _session(model, a)[0],
                                   atol=1e-5)


def test_graphed_streams_with_the_attention_kernel_equal_their_sessions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs run on CUDA")
    pytest.importorskip("triton.language.extra.libdevice")
    # a window the attention kernel takes, which the streams wrap around
    attention = config.AttentionConfig(num_heads=4,
                                       num_kv_heads=2,
                                       head_dim=16,
                                       sliding_window=64)
    branch = config.BranchConfig(hidden_size=32,
                                 ffn_size=64,
                                 attention=attention)
    model = conftest.tiny_model(attention=attention,
                                vad_branch=branch,
                                turn_branch=branch).cuda()
    plan = [(0, 150), (7, 40), (30, 120), (31, 70)]
    audio = _audio(plan)

    got = _run_plan(streams.Streams(model, capacity=4, cuda_graphs=True), audio,
                    plan)

    for i, a in enumerate(audio):
        speech, turns = _session(model, a)
        np.testing.assert_allclose(got[i][0], speech, atol=1e-5)
        np.testing.assert_allclose(got[i][1], turns, atol=1e-5)


def test_graphed_streams_record_on_the_model_device_not_the_current_one():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    model = conftest.tiny_model().to("cuda:1")
    plan = [(0, 8)]
    audio = _audio(plan)

    with torch.cuda.device(0):
        got = _run_plan(streams.Streams(model, capacity=1, cuda_graphs=True),
                        audio, plan)

    np.testing.assert_allclose(got[0][0],
                               _session(model, audio[0])[0],
                               atol=1e-5)
