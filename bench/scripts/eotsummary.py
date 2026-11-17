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
"""The eot-bench result file from the harness's own metrics.

After ``eot-harness predict-streaming`` (or ``predict``) per language and
``compute-metrics`` on each run and on the pooled predictions, this reads
every metrics directory (a ``tradeoff.parquet`` with its ``summary.json``)
under ``--output_dir`` and writes ``bench/results/eotbench.json``: per system,
per language and pooled, the AUC, the average precision, the harness's
headline operating points (lowest false-cutoff rate within a 300 ms and a
600 ms mean-latency budget, lowest mean latency within a 5% and a 10% cutoff
budget) and the point of a fixed policy (threshold 0.5, 200 ms action delay,
3 s timeout).

A run directory names its adapter (``sava_adapter__…``,
``smart_turn_audio_adapter__…``). Each metrics directory under a run is one
reading of its predictions, named by the summary's own ``score_point_s``: the
full score trace, or the score read at 0.2 s of silence (``_at200ms``), which
is LiveKit's protocol for smart-turn and what a run's manifest asks for by
default. Pooled runs live under ``pooled`` (SAVA) or ``pooled_<system>``.

    python bench/scripts/eotsummary.py --output_dir <harness output dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import typing

import numpy as np

import common

if typing.TYPE_CHECKING:
    import pandas


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses this benchmark's command line without global flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir",
                        "--output-dir",
                        required=True,
                        help="The harness's --output-dir.")
    return parser.parse_args(argv)


HEADLINE = (("cutoff_at_300ms", "mean_latency", 0.3), ("cutoff_at_600ms",
                                                       "mean_latency", 0.6),
            ("latency_at_5pct", "cutoff_rate", 0.05), ("latency_at_10pct",
                                                       "cutoff_rate", 0.10))
POLICY = (0.5, 0.2, 3.0)
FIELDS = ("cutoff_rate", "mean_latency", "detect_rate", "timeout_rate")
ADAPTERS = {"sava_adapter": "sava", "smart_turn_audio_adapter": "smart_turn"}


def _number(value) -> float | None:
    value = float(value)
    return None if not math.isfinite(value) else value


def headline(tradeoff: pandas.DataFrame) -> dict:
    try:
        # eot-bench is a separately installed evaluation harness.
        # pylint: disable-next=import-outside-toplevel
        from eot_harness import metrics
    except ImportError as exc:
        raise ImportError(
            "Install the eot-bench harness as described in bench/README.md."
        ) from exc
    model = tradeoff[tradeoff["policy_type"] == "model"]
    out = {}
    for key, column, budget in HEADLINE:
        row = metrics.get_pareto_operating_points(model, column,
                                                  [budget]).iloc[0]
        out[key] = None if not bool(row["feasible"]) else {
            "value": _number(row["value"]),
            "threshold": _number(row["threshold"]),
            "action_delay": _number(row["action_delay"]),
            "timeout": _number(row["timeout"]),
        }
    return out


def policy_point(tradeoff: pandas.DataFrame) -> dict | None:
    threshold, delay, timeout = POLICY
    rows = tradeoff[(tradeoff["policy_type"] == "model") &
                    np.isclose(tradeoff["threshold"], threshold) &
                    np.isclose(tradeoff["action_delay"], delay) &
                    np.isclose(tradeoff["timeout"], timeout)]
    if len(rows) != 1:
        return None
    return {key: _number(rows.iloc[0][key]) for key in FIELDS}


def score(metrics_dir: str) -> tuple[str, dict]:
    """The reading's suffix ("" for the full trace, "_at200ms" for the 0.2 s
    point) and its scores."""
    try:
        # pandas is supplied by the separately installed eot-bench harness.
        import pandas  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "Install the eot-bench harness as described in bench/README.md."
        ) from exc
    with open(os.path.join(metrics_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    tradeoff = pandas.read_parquet(os.path.join(metrics_dir,
                                                "tradeoff.parquet"))
    point = summary.get("score_point_s")
    reading = "" if point is None else f"_at{int(round(1000 * point))}ms"
    return reading, {
        "auc": _number(summary["auc"]),
        "ap": _number(summary["ap"]),
        "score_point_s": point,
        "eot_spans": summary["n_eot_spans"],
        "hold_spans": summary["n_hold_spans"],
        "headline": headline(tradeoff),
        "policy_0.5": policy_point(tradeoff),
    }


def adapter_name(run_dir: str) -> str | None:
    """``<adapter>__<model>`` run -> the system's name, or None for an adapter
    this summary does not know."""
    return ADAPTERS.get(os.path.basename(run_dir).split("__")[0])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    systems: dict[str, dict] = {}
    pattern = os.path.join(args.output_dir, "**", "tradeoff.parquet")
    for path in sorted(glob.glob(pattern, recursive=True)):
        metrics_dir = os.path.dirname(path)
        top = os.path.relpath(metrics_dir, args.output_dir).split(os.sep)[0]
        reading, scores = score(metrics_dir)
        if top.startswith("pooled"):
            base = "sava" if top == "pooled" else top[len("pooled_"):]
            systems.setdefault(base + reading,
                               {"languages": {}})["pooled"] = scores
            continue
        run_dir = metrics_dir
        while adapter_name(run_dir) is None and len(run_dir) > len(
                args.output_dir):
            run_dir = os.path.dirname(run_dir)
        name = adapter_name(run_dir)
        if name is None:
            continue
        language = os.path.basename(os.path.dirname(run_dir))
        systems.setdefault(name + reading,
                           {"languages": {}})["languages"][language] = scores
    result = {
        "systems": systems,
        "policy": {
            "threshold": POLICY[0],
            "action_delay": POLICY[1],
            "timeout": POLICY[2]
        }
    }
    for name, scores in systems.items():
        pooled = scores.get("pooled")
        if pooled is None:
            continue
        cut = pooled["headline"]["cutoff_at_300ms"]
        lat = pooled["headline"]["latency_at_5pct"]
        print(f"{name:14s} pooled auc {pooled['auc']:.4f} cut@300 "
              f"{100 * cut['value'] if cut else float('nan'):5.1f}% lat@5% "
              f"{1000 * lat['value'] if lat else float('nan'):5.0f} ms "
              f"languages {len(scores['languages'])}")
    common.write_result("eotbench", result)


if __name__ == "__main__":
    main()
