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
"""An adapter for LiveKit's eot-bench harness.

https://github.com/livekit/eot-bench, Apache-2.0; data
``livekit/eot-bench-data``, CC BY 4.0.

One causal pass per turn gives the end-of-turn probability per 40 ms frame
(frame k covers samples [640 k, 640 (k + 1))). The score at a harness
timestamp t is the last frame that ends at or before t, so it uses only
audio before t. Grid, span filter and labels are the harness's own.

From a checkout of eot-bench with this repository's ``bench/scripts`` on
``PYTHONPATH`` and ``SAVA_MODEL`` naming the weights:

    eot-harness run --adapter eotbench:SavaAdapter ...
"""

# The adapter deliberately follows the harness's private timing helpers.
# pylint: disable=protected-access

import math
import os

from eot_harness import io
from eot_harness import languages
import numpy as np

import common
from inworld.sava import audio
from inworld.sava import config


def causal_frame_index(timestamp: float, num_samples: int) -> int:
    """The last frame that ends at or before ``timestamp`` (-1 if none)."""
    end = min(int(math.floor(float(timestamp) * config.SAMPLE_RATE + io._EPS)),
              num_samples)
    return end // config.FRAME_SAMPLES - 1


class SavaAdapter:
    """Exposes causal SAVA probabilities through the harness adapter."""
    display_name = "SAVA"
    concurrency = 1

    def __init__(self, source: str | None = None) -> None:
        self.system = common.SavaSystem(
            source or os.environ.get("SAVA_MODEL", "inworld/sava"),
            os.environ.get("SAVA_DEVICE", "cpu"))
        self.adapter_id = "inworld/sava"

    def supports_language(self, lang_code: str) -> bool:
        return languages.supports_any_benchmark_language(lang_code)

    def predict_turn(self, row, *, inference_interval):
        samples, sample_rate = io.decode_audio(row["audio"])
        samples = audio.to_model_rate(np.asarray(samples), int(sample_rate))
        recording = common.Recording(row["id"], samples, [])
        turn = self.system.turn(recording)
        speech = self.system.speech(recording)
        rows = []
        spans = row["silence_spans"]
        for span_index, span in enumerate(spans):
            start, end = float(span["start"]), float(span["end"])
            if end - start < io.DEFAULT_MIN_SILENCE - io._EPS:
                continue
            label = "eot" if span_index == len(spans) - 1 else "hold"
            for timestamp in io._time_grid(start, end, inference_interval):
                k = causal_frame_index(timestamp, samples.size)
                rows.append({
                    "id": row["id"],
                    "language": str(row["language"]).strip().lower(),
                    "span_index": span_index,
                    "timestamp": timestamp,
                    "silence_dur": round(timestamp - start, 6),
                    "p_eot": float(turn[k]) if k >= 0 else 0.0,
                    "label": label,
                    "p_speech": float(speech[k]) if k >= 0 else 0.0,
                })
        return {
            "id": row["id"],
            "audio_sec": samples.size / config.SAMPLE_RATE,
            "events": [],
            "prediction_rows": rows
        }
