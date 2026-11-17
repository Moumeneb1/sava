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
"""Tests for build_vadsuite."""

import json
import os
import subprocess
from unittest import mock

import numpy as np
import soundfile

import build_vadsuite

TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 5
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "SPK1"
        xmin = 0
        xmax = 5
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 1.5
            text = "hello"
        intervals [2]:
            xmin = 1.5
            xmax = 5
            text = ""
    item [2]:
        class = "IntervalTier"
        name = "SPK2"
        xmin = 0
        xmax = 5
        intervals: size = 1
        intervals [1]:
            xmin = 2
            xmax = 3
            text = "yes"
"""


def _write_wav(path, seconds, rate=16000, value=0.1, channels=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    samples = np.full((round(seconds * rate), channels),
                      value,
                      dtype=np.float32)
    soundfile.write(path, samples, rate, subtype="FLOAT")


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_merge_sorts_joins_touching_segments_and_drops_empty_ones():
    segments = [(3.0, 4.0), (0.0, 1.0), (1.0, 2.0), (2.5, 2.5), (0.5, 0.8)]

    merged = build_vadsuite.merge(segments)

    assert merged == [[0.0, 2.0], [3.0, 4.0]]


def test_clip_cuts_at_the_end_and_drops_segments_past_it():
    segments = [[0.0, 1.0], [9.0, 11.0], [12.0, 13.0]]

    clipped = build_vadsuite.clip(segments, 10.0)

    assert clipped == [[0.0, 1.0], [9.0, 10.0]]


def test_parse_textgrid_reads_every_interval_tier(tmp_path):
    path = os.path.join(tmp_path, "meeting.TextGrid")
    _write_text(path, TEXTGRID)

    tiers = build_vadsuite.parse_textgrid(path)

    assert tiers == {
        "SPK1": [(0.0, 1.5, "hello"), (1.5, 5.0, "")],
        "SPK2": [(2.0, 3.0, "yes")],
    }


def test_parse_textgrid_reads_utf16(tmp_path):
    path = os.path.join(tmp_path, "meeting.TextGrid")
    with open(path, "wb") as f:
        f.write(TEXTGRID.encode("utf-16"))

    tiers = build_vadsuite.parse_textgrid(path)

    assert list(tiers) == ["SPK1", "SPK2"]


def test_spoken_keeps_the_intervals_with_text():
    tiers = {
        "SPK1": [(0.0, 1.5, "hello"), (1.5, 5.0, " ")],
        "SPK2": [(2.0, 3.0, "yes")]
    }

    spans = build_vadsuite.spoken(tiers)

    assert spans == [(0.0, 1.5), (2.0, 3.0)]


def test_parse_rttm_groups_speaker_turns_by_file_and_speaker(tmp_path):
    path = os.path.join(tmp_path, "all.rttm")
    _write_text(
        path, "SPEAKER a 1 0.50 1.00 <NA> <NA> spk1 <NA> <NA>\n"
        "SPEAKER a 1 2.00 0.50 <NA> <NA> spk2 <NA> <NA>\n"
        "NON-SPEECH a 1 3.00 1.00 <NA> <NA> <NA> <NA> <NA>\n"
        "SPEAKER b 1 0.00 1.00 <NA> <NA> spk1 <NA> <NA>\n")

    turns = build_vadsuite.parse_rttm(path)

    assert turns == {
        "a": {
            "spk1": [[0.5, 1.5]],
            "spk2": [[2.0, 2.5]]
        },
        "b": {
            "spk1": [[0.0, 1.0]]
        },
    }


def test_ava_segments_join_touching_speech_and_end_at_the_audio():
    conditions = [[10.0, 20.0, "NO_SPEECH"], [0.0, 5.0, "CLEAN_SPEECH"],
                  [5.0, 8.0, "SPEECH_WITH_MUSIC"],
                  [20.0, 905.0, "SPEECH_WITH_NOISE"]]

    segments = build_vadsuite.ava_segments(conditions, 900.0)

    assert segments == [[0.0, 8.0], [20.0, 900.0]]


def test_manifest_row_keeps_the_public_fields_to_the_millisecond():
    row = {
        "id": "set/test/a",
        "name": "a",
        "split": "test",
        "wav": "audio16k/test/a.wav",
        "seconds": 2.0,
        "segments": [[0.12345, 1.99999]],
        "source": "/downloads/a.wav",
    }

    public = build_vadsuite.manifest_row(row)

    assert json.dumps(public) == json.dumps({
        "id": "set/test/a",
        "name": "a",
        "split": "test",
        "wav": "audio16k/test/a.wav",
        "seconds": 2.0,
        "segments": [[0.123, 2.0]],
    })


def test_convert_writes_the_chosen_channel_as_mono_16_khz_float(tmp_path):
    source = os.path.join(tmp_path, "array.wav")
    samples = np.zeros((48000, 2), dtype=np.float32)
    samples[:, 0] = 0.2
    soundfile.write(source, samples, 48000, subtype="FLOAT")
    out = os.path.join(tmp_path, "audio16k", "test", "array.wav")

    seconds = build_vadsuite.convert((source, 1, out))

    written, rate = soundfile.read(out, dtype="float32")
    assert seconds == 1.0
    assert rate == 16000
    assert written.shape == (16000,)
    assert soundfile.info(out).subtype == "FLOAT"
    np.testing.assert_allclose(written, 0.0, atol=1e-6)


def test_convert_averages_the_channels_without_a_channel(tmp_path):
    source = os.path.join(tmp_path, "stereo.wav")
    samples = np.zeros((16000, 2), dtype=np.float32)
    samples[:, 0] = 0.2
    soundfile.write(source, samples, 16000, subtype="FLOAT")
    out = os.path.join(tmp_path, "stereo16k.wav")

    build_vadsuite.convert((source, None, out))

    written, _ = soundfile.read(out, dtype="float32")
    np.testing.assert_allclose(written, 0.1, atol=1e-6)


def test_convert_keeps_audio_already_written(tmp_path):
    out = os.path.join(tmp_path, "audio16k", "test", "a.wav")
    _write_wav(out, 0.5)

    seconds = build_vadsuite.convert((os.path.join(tmp_path,
                                                   "missing.wav"), None, out))

    assert seconds == 0.5


def test_build_writes_the_msdwild_audio_and_manifest(tmp_path):
    downloads = os.path.join(tmp_path, "msdwild")
    _write_wav(os.path.join(downloads, "raw_wav", "00001.wav"), 2.0)
    _write_wav(os.path.join(downloads, "raw_wav", "00002.wav"), 1.0)
    _write_text(
        os.path.join(downloads, "all.rttm"),
        "SPEAKER 00001 1 0.1234 0.5 <NA> <NA> a <NA> <NA>\n"
        "SPEAKER 00001 1 0.5 2.0 <NA> <NA> b <NA> <NA>\n"
        "SPEAKER 00002 1 0.25 0.25 <NA> <NA> a <NA> <NA>\n")
    out = os.path.join(tmp_path, "out")

    rows = build_vadsuite.build("msdwild", downloads, out, workers=1)

    with open(os.path.join(out, "msdwild", "manifest.jsonl"),
              encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines == [
        json.dumps({
            "id": "msdwild/test/00001",
            "name": "00001",
            "split": "test",
            "wav": "audio16k/test/00001.wav",
            "seconds": 2.0,
            "segments": [[0.123, 2.0]],
        }),
        json.dumps({
            "id": "msdwild/test/00002",
            "name": "00002",
            "split": "test",
            "wav": "audio16k/test/00002.wav",
            "seconds": 1.0,
            "segments": [[0.25, 0.5]],
        }),
    ]
    assert rows == [json.loads(line) for line in lines]
    assert soundfile.info(
        os.path.join(out, "msdwild", "audio16k", "test",
                     "00002.wav")).frames == 16000


def test_build_lists_only_the_libriparty_eval_sessions(tmp_path):
    downloads = os.path.join(tmp_path, "libriparty")
    meta = {
        "1089": [{
            "start": 0.2,
            "stop": 0.6
        }],
        "noises": [{
            "start": 0.0,
            "stop": 1.0
        }],
        "background": [{
            "start": 0.0,
            "stop": 1.0
        }],
    }
    for folder, session in (("eval", "session_0"), ("dev", "session_1"),
                            ("dev", "session_0")):
        base = os.path.join(downloads, "LibriParty", "dataset", folder, session)
        _write_wav(os.path.join(base, f"{session}_mixture.wav"), 1.0)
        _write_text(os.path.join(base, f"{session}.json"), json.dumps(meta))

    rows = build_vadsuite.build("libriparty",
                                downloads,
                                os.path.join(tmp_path, "out"),
                                workers=1)

    assert [row["id"] for row in rows] == ["libriparty/test/session_0"]
    assert rows[0]["segments"] == [[0.2, 0.6]]


def test_build_labels_alimeeting_arrays_by_all_speakers_and_headsets_by_wearer(
        tmp_path):
    downloads = os.path.join(tmp_path, "alimeeting")
    far = os.path.join(downloads, "Test_Ali", "Test_Ali_far")
    near = os.path.join(downloads, "Test_Ali", "Test_Ali_near")
    _write_wav(os.path.join(far, "audio_dir", "R1_M1_MS1.wav"), 5.0, channels=2)
    _write_text(os.path.join(far, "textgrid_dir", "R1_M1.TextGrid"), TEXTGRID)
    _write_wav(os.path.join(near, "audio_dir", "R1_M1_N_SPK2.wav"), 5.0)
    _write_text(os.path.join(near, "textgrid_dir", "R1_M1_N_SPK2.TextGrid"),
                TEXTGRID.replace('text = "hello"', 'text = ""'))

    rows = build_vadsuite.build("alimeeting",
                                downloads,
                                os.path.join(tmp_path, "out"),
                                workers=1)

    segments = {row["id"]: row["segments"] for row in rows}
    assert segments["alimeeting/test/R1_M1_MS1"] == [[0.0, 1.5], [2.0, 3.0]]
    assert segments["alimeeting/test/R1_M1_N_SPK2"] == [[2.0, 3.0]]
    assert len(rows) == 2


def test_build_lists_every_esc50_clip_without_segments(tmp_path):
    downloads = os.path.join(tmp_path, "esc50")
    root = os.path.join(downloads, "ESC-50-master")
    _write_text(
        os.path.join(root, "meta", "esc50.csv"),
        "filename,fold,target,category\n"
        "1-100-A-0.wav,1,0,dog\n"
        "2-200-A-1.wav,2,1,rooster\n")
    for name in ("1-100-A-0.wav", "2-200-A-1.wav"):
        _write_wav(os.path.join(root, "audio", name), 0.5, rate=44100)

    rows = build_vadsuite.build("esc50",
                                downloads,
                                os.path.join(tmp_path, "out"),
                                workers=1)

    assert [(row["name"], row["seconds"], row["segments"]) for row in rows] == [
        ("1-100-A-0", 0.5, []),
        ("2-200-A-1", 0.5, []),
    ]


def test_decode_mp3_reads_the_channels_and_rate_that_ffprobe_reports():
    pcm = np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32)
    probe = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='{"streams": [{"sample_rate": "44100", "channels": 2}]}')
    decoded = subprocess.CompletedProcess(args=[],
                                          returncode=0,
                                          stdout=pcm.tobytes())

    with mock.patch.object(subprocess,
                           "run",
                           autospec=True,
                           spec_set=True,
                           side_effect=[probe, decoded]) as run:
        samples, rate = build_vadsuite.decode_mp3("talk.mp3")

    assert rate == 44100
    np.testing.assert_array_equal(samples, pcm)
    assert run.call_args_list[0].args[0][0] == "ffprobe"
    assert run.call_args_list[1].args[0][0] == "ffmpeg"


def test_fetch_ava_writes_seconds_900_to_1800_at_16_khz_mono(tmp_path):
    out = os.path.join(tmp_path, "audio16k", "video.wav")

    def ffmpeg(command, check):
        del check
        _write_wav(command[-1], 1.0)
        return subprocess.CompletedProcess(args=command, returncode=0)

    with mock.patch.object(subprocess,
                           "run",
                           autospec=True,
                           spec_set=True,
                           side_effect=ffmpeg) as run:
        written = build_vadsuite.fetch_ava(("video.mkv", out))

    command = run.call_args.args[0]
    assert written
    assert os.path.exists(out)
    assert command[command.index("-i") + 1] == (
        "https://s3.amazonaws.com/ava-dataset/trainval/video.mkv")
    assert command[command.index("-ss") + 1] == "900"
    assert command[command.index("-t") + 1] == "900"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"


def test_fetch_ava_gives_up_after_three_failed_attempts(tmp_path):
    failed = subprocess.CompletedProcess(args=[], returncode=1)

    with mock.patch.object(subprocess,
                           "run",
                           autospec=True,
                           spec_set=True,
                           return_value=failed) as run:
        written = build_vadsuite.fetch_ava(
            ("video.mkv", os.path.join(tmp_path, "video.wav")))

    assert not written
    assert run.call_count == 3
