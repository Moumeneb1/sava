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
"""Tests the recording command and its explicit policy options."""

import builtins
import json
import os
from unittest import mock

import numpy as np
import pytest

from inworld.sava import audio
from inworld.sava import checkpoint
from inworld.sava import cli
from inworld.sava import segmenter


@pytest.fixture(name="recording")
def example_recording(tmp_path):
    path = os.path.join(tmp_path, "recording.wav")
    audio.save(path, (0.1 * np.sin(np.arange(16000) / 5)).astype(np.float32))
    return path


def test_run_prints_turns_and_writes_curves_and_wavs(model, recording, tmp_path,
                                                     capsys):
    curves_path = os.path.join(tmp_path, "curves.json")
    turns_dir = os.path.join(tmp_path, "turns")
    policy = segmenter.Policy(min_speech_ms=0)
    turn = segmenter.Turn(start=2,
                          end=10,
                          speech_ms=200,
                          reason="turn",
                          kept=True)

    with (mock.patch.object(checkpoint,
                            "load",
                            autospec=True,
                            return_value=model),
          mock.patch.object(segmenter,
                            "segment",
                            autospec=True,
                            return_value=[turn]) as segment):
        turns = cli.run_file(recording,
                             policy=policy,
                             curves_path=curves_path,
                             turns_dir=turns_dir)

    out = capsys.readouterr().out
    assert turns == [turn]
    assert "closed by turn    kept" in out
    assert "1 turns, 1 kept" in out
    with open(curves_path, encoding="utf-8") as f:
        curves = json.load(f)
    assert len(curves["speech"]) == len(curves["turn"]) == 25
    assert audio.load(os.path.join(turns_dir, "turn_000.wav")).size == 8 * 640
    assert segment.call_args.args[1] == policy


def test_policy_options_do_not_leak_between_invocations():
    changed = cli.parse_args(
        ["mic", "--turn_threshold", "0.7", "--pre-roll-ms", "400"])
    default = cli.parse_args(["mic"])

    assert cli.policy_from_args(changed) == segmenter.Policy(turn_threshold=0.7,
                                                             pre_roll_ms=400)
    assert cli.policy_from_args(default) == segmenter.Policy()


@pytest.mark.parametrize("argv", [["fly"], ["run"], ["mic", "recording.wav"]])
def test_main_rejects_invalid_commands(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)

    assert exc.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_help_succeeds_without_loading_weights(capsys):
    with mock.patch.object(checkpoint, "load", autospec=True) as load:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--help"])

    assert exc.value.code == 0
    assert "sava run" in capsys.readouterr().out
    load.assert_not_called()


def test_main_passes_file_and_options_to_recording_command():
    with mock.patch.object(cli, "run_file", autospec=True) as run_file:
        cli.main([
            "--device", "cpu", "run", "--model", "weights.safetensors",
            "recording.wav", "--curves", "curves.json"
        ])

    run_file.assert_called_once_with("recording.wav",
                                     model_source="weights.safetensors",
                                     device="cpu",
                                     policy=segmenter.Policy(),
                                     curves_path="curves.json",
                                     turns_dir=None)


def test_main_passes_microphone_options():
    with mock.patch.object(cli, "run_mic", autospec=True) as run_mic:
        cli.main(["mic", "--seconds", "2.5", "--turns_dir", "turns"])

    run_mic.assert_called_once_with(model_source="inworld/sava",
                                    device="cpu",
                                    policy=segmenter.Policy(),
                                    seconds=2.5,
                                    turns_dir="turns")


def test_missing_microphone_extra_explains_installation():
    original_import = builtins.__import__

    def import_without_microphone(name, *args, **kwargs):
        if name == "sounddevice":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    with mock.patch.object(builtins,
                           "__import__",
                           autospec=True,
                           side_effect=import_without_microphone):
        with pytest.raises(ImportError, match=r"inworld-sava\[mic\]"):
            cli.run_mic()


def test_format_turn_marks_dropped_turns():
    line = cli.format_turn(
        segmenter.Turn(start=0,
                       end=25,
                       speech_ms=80,
                       reason="silence",
                       kept=False))

    assert line.endswith("closed by silence dropped")
    assert "1.00" in line
