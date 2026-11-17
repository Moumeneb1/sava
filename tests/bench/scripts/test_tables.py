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
"""Tests for tables."""

import json

import tables


def test_explicit_results_directories_are_independent(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    result = {"hours": 1.5, "systems": {"sava": {"auc": 0.7, "f1": 0.6}}}
    (first / "tenvad.json").write_text(json.dumps(result), encoding="utf-8")

    with_results = tables.render(str(first))
    empty = tables.render(str(second))

    assert "TEN-VAD test | 1.5 | 0.7000" in with_results
    assert empty == ""
    assert tables.render(str(first)) == with_results
