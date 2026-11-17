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
"""Tests for ami."""

import ami


def test_original_negative_boolean_spelling_and_list_are_retained():
    args = ami.parse_args(
        ["--root", "data", "--notimed_only", "--thresholds", "0.3, 0.7"])

    assert args.timed_only is False
    assert args.thresholds == ["0.3", "0.7"]
    assert ami.parse_args(["--root", "data"]).timed_only is True
