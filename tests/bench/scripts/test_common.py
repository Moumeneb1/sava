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
"""Tests for common."""

import argparse
import importlib

import numpy as np
import pytest
from sklearn import metrics

import common
from inworld.sava import config


@pytest.fixture(name="recording")
def example_recording():
    samples = np.zeros(10 * config.FRAME_SAMPLES, np.float32)
    return common.Recording("r",
                            samples, [(0.1, 0.25), (0.3, 0.33)],
                            scored=[(0.0, 0.2)])


def test_frame_labels_follow_the_centre_rule(recording):
    labels = common.frame_labels(recording)

    # Centres at 20, 60, 100, ... ms: 100 to 220 ms inside the first segment;
    # the 30 ms segment covers the centre at 300 ms.
    assert labels.tolist() == [
        False, False, True, True, True, True, False, True, False, False
    ]


def test_frame_mask_follows_the_scored_spans(recording):
    mask = common.frame_mask(recording)

    assert mask.tolist() == [True] * 5 + [False] * 5
    assert common.frame_mask(common.Recording("r", recording.samples, [])).all()


def test_binned_auc_matches_sklearn_without_ties():
    rng = np.random.default_rng(0)
    labels = rng.random(500) < 0.4
    scores = rng.random(500) + 0.3 * labels
    exact = metrics.roc_auc_score(labels, scores)

    positives, negatives = common.binned_histograms([labels], [scores],
                                                    bins=100000)

    assert common.binned_auc(positives[0],
                             negatives[0]) == pytest.approx(exact, abs=1e-6)


def test_bootstrap_sees_a_clear_winner():
    rng = np.random.default_rng(1)
    labels = [rng.random(300) < 0.5 for _ in range(20)]
    strong = [lab + 0.2 * rng.random(300) for lab in labels]
    weak = [rng.random(300) + 0.1 * lab for lab in labels]

    result = common.bootstrap_auc_difference(labels,
                                             strong,
                                             weak,
                                             resamples=200)

    assert result["low"] > 0.0
    assert result["wins"] == 1.0


def test_frame_scores_report_the_threshold_metrics():
    labels = np.array([True, True, False, False])
    probs = np.array([0.9, 0.4, 0.6, 0.1])

    scores = common.frame_scores(labels, probs)

    assert scores["accuracy"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["auc"] == 0.75
    assert scores["best_accuracy"] >= 0.5


class _ScoredElsewhere:
    """A system whose curves are given, so its own pass must not run."""
    name = "sava"

    def speech(self, recording):
        raise AssertionError("the pass ran although curves were given")


def test_score_recordings_reuses_given_curves():
    samples = np.zeros(4 * config.FRAME_SAMPLES, np.float32)
    recordings = [
        common.Recording("a", samples, [(0.0, 0.08)]),
        common.Recording("b", samples, [(0.08, 0.16)])
    ]
    curves = {
        "sava": [
            np.array([0.9, 0.8, 0.1, 0.2]),
            np.array([0.1, 0.2, 0.9, 0.8])
        ]
    }

    result = common.score_recordings(recordings, [_ScoredElsewhere()], curves)

    assert result["systems"]["sava"]["auc"] == 1.0
    assert result["systems"]["sava"]["frames"] == 8


@pytest.mark.parametrize("option, expected", [([], True), (["--silero"], True),
                                              (["--nosilero"], False),
                                              (["--no-silero"], False),
                                              (["--silero=false"], False),
                                              (["--silero=0"], False),
                                              (["--silero=yes"], True)])
def test_boolean_options_keep_original_and_hyphenated_spellings(
        option, expected):
    parser = argparse.ArgumentParser()
    common.add_boolean_argument(parser, "silero", default=True, description="")

    args = parser.parse_args(option)

    assert args.silero is expected


def test_boolean_options_reject_unknown_values():
    parser = argparse.ArgumentParser()
    common.add_boolean_argument(parser, "silero", default=True, description="")

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--silero=perhaps"])

    assert exc.value.code == 2


@pytest.mark.parametrize("module_name, argv", [
    ("ami", ["--root", "ami"]),
    ("build_vadsuite", ["--set", "aishell4", "--downloads", "downloads"]),
    ("eotsummary", ["--output_dir", "metrics"]),
    ("pipecat", ["--root", "pipecat"]),
    ("tables", []),
    ("tenvad", ["--root", "tenvad"]),
    ("turnsets", ["--root", "turnsets"]),
    ("vadsuite", ["--set", "esc50", "--root", "vadsuite"]),
    ("voxconverse", ["--root", "voxconverse"]),
])
def test_benchmark_parsers_import_together_and_have_independent_options(
        module_name, argv, capsys):
    module = importlib.import_module(module_name)
    first = module.parse_args(argv)
    first.root = "changed"

    second = module.parse_args(argv)
    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--help"])

    assert getattr(second, "root", None) != "changed"
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out
