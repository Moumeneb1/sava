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
"""The pipecat smart-turn test set: is the turn complete at the clip's end?

Data: ``pipecat-ai/smart-turn-data-v3.2-test`` (CC BY 4.0) downloaded with
``huggingface-cli download --repo-type dataset``: 31.5k clips, 16 kHz, one
label ``endpoint_bool`` each, 23 languages, real and synthetic sources.

The clip's score is the end-of-turn probability of its last frame (the
stream as given, nothing appended), smart-turn's its own output on the clip.
Metrics follow smart-turn's ``train.py``: accuracy, precision, recall and F1
with "complete" positive at threshold 0.5, overall and per language, source
and real / synthetic; plus a threshold sweep.

    python bench/scripts/pipecat.py --root <download> --model <weights>
        [--smart_turn smart-turn-v3.2-gpu.onnx]
"""

import argparse
import glob
import io
import os
import time

import numpy as np
from sklearn import metrics
import soundfile

import common
from inworld.sava import audio
import smartturn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses this benchmark's command line without global flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="The dataset download.")
    parser.add_argument("--model",
                        default="inworld/sava",
                        help="Checkpoint source.")
    parser.add_argument("--device", default="cpu", help="")
    parser.add_argument("--smart_turn",
                        "--smart-turn",
                        default=None,
                        help="smart-turn v3.2 ONNX file; scores it too.")
    parser.add_argument("--limit",
                        default=None,
                        type=int,
                        help="Clips to score (all by default).")
    return parser.parse_args(argv)


COLUMNS = ("id", "language", "endpoint_bool", "synthetic", "dataset", "audio")


def clips(root: str, limit: int | None):
    """Yields (metadata, samples) in shard order."""
    try:
        # Dataset readers belong to the optional eval extra.
        # pylint: disable-next=import-outside-toplevel
        from pyarrow import parquet as parquet_lib
    except ImportError as exc:
        raise ImportError(
            'Parquet input requires pip install "inworld-sava[eval]".') from exc
    paths = sorted(
        glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No parquet shards under [{root}].")
    count = 0
    for path in paths:
        parquet = parquet_lib.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            for row in parquet.read_row_group(
                    group, columns=list(COLUMNS)).to_pylist():
                samples, rate = soundfile.read(io.BytesIO(
                    row["audio"]["bytes"]),
                                               dtype="float32",
                                               always_2d=True)
                meta = {key: row[key] for key in COLUMNS if key != "audio"}
                yield meta, audio.to_model_rate(samples, rate)
                count += 1
                if limit is not None and count >= limit:
                    return


def binary(labels: np.ndarray, decided: np.ndarray) -> dict[str, float]:
    return {
        "n":
            int(labels.size),
        "accuracy":
            float(metrics.accuracy_score(labels, decided)),
        "precision":
            float(metrics.precision_score(labels, decided, zero_division=0)),
        "recall":
            float(metrics.recall_score(labels, decided, zero_division=0)),
        "f1":
            float(metrics.f1_score(labels, decided, zero_division=0)),
    }


def summarize(rows: list[dict], key: str, threshold: float = 0.5) -> dict:
    labels = np.array([r["endpoint_bool"] for r in rows], dtype=bool)
    probs = np.array([r[key] for r in rows])
    out = {
        "threshold": threshold,
        "overall": binary(labels, probs > threshold),
        "auc": float(metrics.roc_auc_score(labels, probs)),
        "groups": {}
    }
    for field in ("language", "dataset", "synthetic"):
        keys = np.array([str(r[field]) for r in rows])
        out["groups"][field] = {
            k: binary(labels[keys == k], probs[keys == k] > threshold)
            for k in sorted(set(keys))
        }
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    accuracies = [((probs > t) == labels).mean() for t in grid]
    best = int(np.argmax(accuracies))
    out["sweep"] = {
        "best_threshold": float(grid[best]),
        "best_accuracy": float(accuracies[best])
    }
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    system = common.SavaSystem(args.model, args.device)
    baseline = smartturn.SmartTurn(args.smart_turn) if args.smart_turn else None
    rows, started = [], time.time()
    for meta, samples in clips(args.root, args.limit):
        turn = system.turn(common.Recording(meta["id"], samples, []))
        meta["sava"] = float(turn[-1])
        if baseline is not None:
            meta["smart_turn"] = baseline.probability(samples)
        rows.append(meta)
        if len(rows) % 1000 == 0:
            print(f"{len(rows)} clips, {time.time() - started:.0f} s",
                  flush=True)
    result = {"clips": len(rows), "systems": {"sava": summarize(rows, "sava")}}
    if baseline is not None:
        result["systems"]["smart_turn"] = summarize(rows, "smart_turn")
    for name, scores in result["systems"].items():
        print(f"{name:10s} accuracy {scores['overall']['accuracy']:.4f} "
              f"auc {scores['auc']:.4f} "
              f"best {scores['sweep']['best_accuracy']:.4f} "
              f"at {scores['sweep']['best_threshold']:.2f}")
    common.write_result("pipecat", result)


if __name__ == "__main__":
    main()
