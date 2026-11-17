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
"""Tests for turnsets."""

import numpy as np

import turnsets


def test_threshold_list_and_original_spelling_are_retained():
    args = turnsets.parse_args([
        "--root", "data", "--anchor_ms", "200", "--tail_dbfs", "-70",
        "--thresholds", "0.4,0.5,0.6"
    ])

    assert args.anchor_ms == 200
    assert args.tail_dbfs == -70.0
    assert args.thresholds == ["0.4", "0.5", "0.6"]


def test_preparation_options_do_not_leak_between_clips():
    samples = np.full(1600, 0.2, dtype=np.float32)

    anchored = turnsets.prepare(samples, "one", anchor_ms=200)
    default = turnsets.prepare(samples, "two")

    np.testing.assert_array_equal(default, samples)
    assert anchored.size == samples.size + 3200
    np.testing.assert_array_equal(anchored[:samples.size], samples)
    np.testing.assert_array_equal(
        turnsets.prepare(samples, "one", anchor_ms=200), anchored)
    assert not np.array_equal(turnsets.prepare(samples, "two", anchor_ms=200),
                              anchored)
