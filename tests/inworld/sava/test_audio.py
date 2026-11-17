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
"""Audio conversion, resampling and atomic output contracts."""

import os
from unittest import mock

import numpy as np
import pytest
import soundfile

from inworld.sava import audio
from inworld.sava import config


def test_stereo_is_downmixed():
    stereo = np.stack([np.ones(100, np.float32), -np.ones(100, np.float32)],
                      axis=1)

    mono = audio.to_model_rate(stereo, config.SAMPLE_RATE)

    assert mono.shape == (100,)
    np.testing.assert_allclose(mono, 0.0)


def test_integer_pcm_is_scaled_by_its_range():
    pcm = np.array([0, 16384, -32768], dtype=np.int16)

    scaled = audio.to_model_rate(pcm, config.SAMPLE_RATE)

    assert scaled.dtype == np.float32
    np.testing.assert_allclose(scaled, [0.0, 16384 / 32767, -32768 / 32767],
                               atol=1e-6)


def test_other_rates_are_resampled():
    seconds = 0.5
    t = np.arange(int(48000 * seconds)) / 48000
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    resampled = audio.to_model_rate(tone, 48000)

    assert resampled.size == int(config.SAMPLE_RATE * seconds)
    assert np.abs(resampled[2000:6000]).max() == pytest.approx(1.0, abs=0.02)


def test_three_dimensional_audio_is_refused():
    with pytest.raises(ValueError):
        audio.to_model_rate(np.zeros((2, 2, 2), np.float32), config.SAMPLE_RATE)


def test_save_then_load_roundtrips_at_the_model_rate(tmp_path):
    path = os.path.join(tmp_path, "tone.wav")
    samples = (0.5 * np.sin(np.arange(1600) / 7)).astype(np.float32)

    audio.save(path, samples)
    loaded = audio.load(path)

    assert soundfile.info(path).samplerate == config.SAMPLE_RATE
    np.testing.assert_allclose(loaded, samples, atol=1e-4)


@pytest.mark.parametrize("existing", [False, True])
def test_failed_save_leaves_no_partial_file_and_preserves_existing_output(
        tmp_path, existing):
    path = tmp_path / "recording.wav"
    original = b"previous recording"
    if existing:
        path.write_bytes(original)
    write = soundfile.write

    def interrupted(destination, samples, *args, **kwargs):
        write(destination, samples[:10], *args, **kwargs)
        raise OSError("disk full")

    with mock.patch.object(soundfile, "write", autospec=True,
                           side_effect=interrupted), \
            pytest.raises(OSError, match="disk full"):
        audio.save(str(path), np.zeros(100, dtype=np.float32))

    assert list(tmp_path.iterdir()) == ([path] if existing else [])
    if existing:
        assert path.read_bytes() == original
