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
"""Released checkpoint identity and parity with the reference implementation.

The speech clip's logits were computed by the reference implementation's
frame-by-frame path in exact fp32. The fixture arrays stay unchanged.
The existing release registry identifies the weights; an override must match
that checksum. Skips without ``models/sava.safetensors`` or
``SAVA_GOLDEN_MODEL``.
"""

import os

import numpy as np
import torch

from inworld.sava import audio
from inworld.sava import checkpoint
from inworld.sava import config
from inworld.sava import registry
from inworld.sava import streaming

_ATOL = 2e-4


def test_the_released_weights_match_the_training_reference(
        released_weights, fixtures_dir):
    digest = registry.sha256_file(released_weights)
    assert digest == registry.MODELS["inworld/sava"].sha256, (
        "Golden parity requires the registered release weights.")
    model = checkpoint.load(released_weights)
    clip = audio.load(os.path.join(fixtures_dir, "speech.wav"))
    with np.load(os.path.join(fixtures_dir, "speech_expected.npz")) as data:
        expected = {name: data[name] for name in data.files}
    lag = model.lookahead_frames

    with torch.inference_mode():
        out = model(torch.from_numpy(clip)[None])

    np.testing.assert_allclose(out.speech[0].numpy(),
                               expected["speech"],
                               atol=_ATOL)
    np.testing.assert_allclose(out.lookahead[0].numpy()[lag:],
                               expected["lookahead"][lag:],
                               atol=_ATOL)
    np.testing.assert_allclose(out.turn[0].numpy(),
                               expected["turn"],
                               atol=_ATOL)
    probabilities = {
        name: torch.sigmoid(torch.from_numpy(logits)).numpy()
        for name, logits in expected.items()
    }
    curves = streaming.run(model, clip)
    frames = len(expected["speech"])
    for sizes in ((config.FRAME_SAMPLES,), (1000,), (13, 1280, 777)):
        session = streaming.Session(model)
        outputs = []
        offset, index = 0, 0
        while offset < clip.size:
            size = sizes[index % len(sizes)]
            outputs.append(session.push(clip[offset:offset + size]))
            offset += size
            index += 1
        outputs.append(session.flush())
        speech = np.concatenate([output.speech for output in outputs])
        turns = np.concatenate([output.turns for output in outputs])

        assert session.frames == frames + lag
        np.testing.assert_allclose(speech[lag:frames],
                                   probabilities["lookahead"][lag:],
                                   atol=_ATOL)
        np.testing.assert_allclose(turns[:frames],
                                   probabilities["turn"],
                                   atol=_ATOL)
        np.testing.assert_allclose(speech[lag:], curves.speech, atol=1e-5)
        np.testing.assert_allclose(turns[:frames], curves.turns, atol=1e-5)
