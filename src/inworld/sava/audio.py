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
"""Audio in and out: the model takes mono float32 at 16 kHz."""

import os
import tempfile

import numpy as np
import soundfile
import soxr

from inworld.sava import config


def to_model_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Converts ``[S]`` or ``[S, channels]`` audio to mono float32 at 16 kHz.

    Integer PCM is scaled by its dtype's range.
    """
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio / float(np.iinfo(audio.dtype).max)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError(
            f"Expected [S] or [S, channels] audio, got shape {audio.shape}.")
    if sample_rate != config.SAMPLE_RATE:
        audio = soxr.resample(audio, sample_rate,
                              config.SAMPLE_RATE).astype(np.float32)
    return audio


def load(path: str) -> np.ndarray:
    """A file's audio as mono float32 at the model's rate."""
    audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    return to_model_rate(audio, sample_rate)


def save(path: str,
         audio: np.ndarray,
         sample_rate: int = config.SAMPLE_RATE) -> None:
    """Writes PCM16 audio atomically, preserving any file if writing fails."""
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary = tempfile.mkstemp(dir=directory,
                                             prefix=".partial-",
                                             suffix=os.path.splitext(path)[1])
    os.close(descriptor)
    try:
        soundfile.write(temporary, audio, sample_rate, subtype="PCM_16")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
