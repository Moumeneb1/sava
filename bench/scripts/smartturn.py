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
"""pipecat smart-turn v3.2, the end-of-turn baseline, run as its own
``inference.py`` runs it: the clip's last 8 s zero-padded at the front,
Whisper log-mel features, the ONNX model's sigmoid output.

The ONNX file is pipecat's release (``smart-turn-v3.2-gpu.onnx``, fp32, or
``smart-turn-v3.2-cpu.onnx``, int8) from https://huggingface.co/pipecat-ai,
BSD-2-Clause; pass its path. Needs the ``eval`` extra.
"""

import numpy as np

from inworld.sava import config

WINDOW_SECONDS = 8


class SmartTurn:
    """Runs the released smart-turn baseline on its native feature window."""

    def __init__(self, onnx_path: str, threads: int = 4):
        try:
            # Both dependencies belong to the optional eval extra.
            import onnxruntime  # pylint: disable=import-outside-toplevel
            import transformers  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                'smart-turn requires pip install "inworld-sava[eval]".'
            ) from exc

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            onnx_path, sess_options=options, providers=["CPUExecutionProvider"])
        self.features = transformers.WhisperFeatureExtractor(
            chunk_length=WINDOW_SECONDS)
        self.name = "smart_turn"

    def window(self, samples: np.ndarray) -> np.ndarray:
        length = WINDOW_SECONDS * config.SAMPLE_RATE
        if samples.size >= length:
            return samples[-length:]
        return np.pad(samples, (length - samples.size, 0))

    def probability(self, samples: np.ndarray) -> float:
        features = self.features([self.window(samples)],
                                 sampling_rate=config.SAMPLE_RATE,
                                 return_tensors="np",
                                 padding="max_length",
                                 max_length=WINDOW_SECONDS * config.SAMPLE_RATE,
                                 truncation=False,
                                 do_normalize=True)["input_features"]
        out = self.session.run(None,
                               {self.session.get_inputs()[0].name: features})[0]
        return float(np.asarray(out).reshape(-1)[0])
