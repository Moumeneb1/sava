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
"""Checks that published figures follow the recorded scores and timing."""

import json
import re
import xml.etree.ElementTree as ET

import pytest

import build_figures

SVG = "{http://www.w3.org/2000/svg}"


@pytest.mark.parametrize("task", ["vad", "turn"])
def test_rows_read_each_recorded_auc_in_dataset_order(tmp_path, task):
    sets = build_figures.VAD_SETS if task == "vad" else build_figures.TURN_SETS
    baseline = "silero" if task == "vad" else "smart_turn"
    for name, _ in sets:
        ours = {"all": 0.61} if task == "turn" and name != "pipecat" else 0.61
        other = {"all": 0.72} if isinstance(ours, dict) else 0.72
        result = {"systems": {"sava": {"auc": ours}, baseline: {"auc": other}}}
        (tmp_path / f"{name}.json").write_text(json.dumps(result),
                                               encoding="utf-8")

    rows = build_figures.benchmark_rows(tmp_path, task)

    assert rows == [(label, 0.61, 0.72) for _, label in sets]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_comparison_labels_values_and_caveats_are_preserved(theme):
    rows = [("A < B & C", 0.9234, 0.9876)]

    svg = build_figures.benchmark_svg(rows, "vad", theme)
    root = ET.fromstring(svg)

    assert "0.9234" in svg and "0.9876" in svg
    assert "A &lt; B &amp; C" in svg
    assert "Inworld SAVA" in svg
    assert "120 ms lookahead" in svg
    assert root.attrib["role"] == "img"
    assert build_figures.PALETTES[theme]["surface"] in svg


def test_voice_activity_bars_start_at_the_chart_floor_of_0_9():
    rows = [("A", 0.95, 1.0)]

    svg = build_figures.benchmark_svg(rows, "vad", "light")
    root = ET.fromstring(svg)
    ours, other = (float(re.search(r" h([\d.]+) ", bar.attrib["d"]).group(1))
                   for bar in root.iter(f"{SVG}path"))
    ticks = [node.text for node in root.iter(f"{SVG}text")]

    assert (ours + 4) / (other + 4) == pytest.approx(0.5)
    assert "0.90" in ticks and "1.00" in ticks


def test_a_value_below_the_chart_floor_is_refused():
    rows = [("A", 0.85, 0.95)]

    with pytest.raises(ValueError, match=r"below the chart's floor of 0\.9"):
        build_figures.benchmark_svg(rows, "vad", "light")


def test_missing_results_fail_instead_of_publishing_a_partial_board(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_figures.benchmark_rows(tmp_path, "vad")


def test_architecture_only_needs_no_results(tmp_path):
    build_figures.main(["--architecture-only", "--out", str(tmp_path)])

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "architecture-dark.svg", "architecture-light.svg"
    ]
    for path in tmp_path.iterdir():
        svg = path.read_text(encoding="utf-8")
        ET.fromstring(svg)
        assert "16 kHz" in svg and "40 ms" in svg
        assert "120 ms" in svg and "on time" in svg


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_architecture_is_live_text_and_describes_the_actual_signal_paths(theme):
    svg = build_figures.architecture_svg(theme)
    root = ET.fromstring(svg)

    assert root.attrib["role"] == "img"
    words = "".join(
        "".join(node.itertext()) for node in root.iter(f"{SVG}text"))
    for stage in ("Causal log-mel", "Frame mixer", "Shared trunk",
                  "Speech branch", "Speech output", "On-time speech",
                  "Turn branch", "End of turn", "Inside each trunk block",
                  "Timing at emission step"):
        assert stage in words
    description = root.find(f"{SVG}desc")
    for contract in ("16 kHz", "40 ms", "1026 to 512", "three Mamba-2",
                     "frame t-3", "120 ms", "64-dimensional",
                     "previous frame's on-time speech", "0, 5, 10 and 20"):
        assert contract in description.text
    assert build_figures.ARCH_PALETTES[theme]["bg"] in svg


def test_architecture_is_deterministic_and_stays_within_its_canvas():
    first = build_figures.architecture_svg("light")
    second = build_figures.architecture_svg("light")
    dark = build_figures.architecture_svg("dark")

    assert first == second
    assert first != dark
    for svg in (first, dark):
        root = ET.fromstring(svg)
        width = float(root.attrib["width"])
        assert width == build_figures.CHART_WIDTH
        assert all(0 <= float(node.attrib["x"]) <= width
                   for node in root.iter(f"{SVG}text"))
