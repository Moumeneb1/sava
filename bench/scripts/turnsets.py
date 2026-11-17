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
"""Easy Turn and FastTurn test sets: four turn states, one clip each.

Data (both Apache-2.0, on Hugging Face):

- ``1xg/Easy-Turn-Testset``: ``<state>/<state>_test.list`` (or
  ``incomplete/incomplete_real_test.list``) rows with ``key``, ``wav``,
  ``txt`` ending in the state's tag; ``--format easyturn``.
- ``ASLP-lab/FastTurn-Testset``: ``data.jsonl`` rows with ``key``, ``wav``,
  ``state``; ``--format fastturn``.

The clip's score is the end-of-turn probability of its last frame. ``--pad_ms``
appends digital silence to every clip; ``--anchor_ms`` cuts each clip at its
energy speech end and appends that many ms of low-level white noise, so every
clip ends the same way after its speech. Reported: the share of each state
called ended at 0.5, and on complete vs incomplete the accuracy, the paper's
ACC_cp / ACC_incp, the miss and false-alarm rates and the AUC.

    python bench/scripts/turnsets.py --root <testset> --format easyturn \\
        --model <weights>
"""

import argparse
import json
import os
import zlib

import numpy as np
from sklearn import metrics

import common
from inworld.sava import audio
from inworld.sava import config
import smartturn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses this benchmark's command line without global flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="The test set directory.")
    parser.add_argument("--format",
                        default="easyturn",
                        choices=["easyturn", "fastturn"],
                        help="")
    parser.add_argument("--model",
                        default="inworld/sava",
                        help="Checkpoint source.")
    parser.add_argument("--device", default="cpu", help="")
    parser.add_argument("--smart_turn",
                        "--smart-turn",
                        default=None,
                        help="smart-turn v3.2 ONNX file; scores it too.")
    parser.add_argument("--pad_ms",
                        "--pad-ms",
                        default=0,
                        type=int,
                        help="Silence appended to every clip.")
    parser.add_argument("--anchor_ms",
                        "--anchor-ms",
                        default=None,
                        type=int,
                        help="Anchored read: noise tail after the speech end.")
    parser.add_argument("--tail_dbfs",
                        "--tail-dbfs",
                        default=-70.0,
                        type=float,
                        help="RMS of the anchored tail.")
    parser.add_argument("--thresholds",
                        default=["0.5"],
                        type=common.comma_separated,
                        help="")
    parser.add_argument("--limit", default=None, type=int, help="")
    return parser.parse_args(argv)


STATES = ("complete", "incomplete", "backchannel", "wait")
EASY_TURN_LISTS = (("complete", "complete/complete_test.list"),
                   ("incomplete", "incomplete/incomplete_real_test.list"),
                   ("backchannel", "backchannel/backchannel_test.list"),
                   ("wait", "wait/wait_test.list"))


def load_easyturn(root: str) -> list[dict]:
    rows = []
    for state, rel in EASY_TURN_LISTS:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                wav = os.path.normpath(os.path.join(root, r["wav"]))
                rows.append({
                    "key": r["key"],
                    "state": state,
                    "source": "real" if "/real/" in wav else "synthetic",
                    "wav": wav
                })
    return rows


def load_fastturn(root: str) -> list[dict]:
    rows = []
    with open(os.path.join(root, "data.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({
                    "key": r["key"],
                    "state": r["state"],
                    "source": "synthetic" if r["state"] == "wait" else "real",
                    "wav": os.path.normpath(os.path.join(root, r["wav"]))
                })
    return rows


def speech_end(samples: np.ndarray) -> int:
    """Sample index after the last 10 ms frame within 35 dB of the loudest
    and above -55 dBFS."""
    hop = config.SAMPLE_RATE // 100
    n = samples.size // hop
    power = np.mean(samples[:n * hop].reshape(n, hop).astype(np.float64)**2,
                    axis=1)
    db = 10 * np.log10(power + 1e-12)
    active = np.flatnonzero(db >= max(db.max() - 35.0, -55.0))
    return int(active[-1] + 1) * hop if active.size else samples.size


def prepare(samples: np.ndarray,
            key: str,
            *,
            anchor_ms: int | None = None,
            tail_dbfs: float = -70.0,
            pad_ms: int = 0) -> np.ndarray:
    if anchor_ms is not None:
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        tail = rng.standard_normal(anchor_ms * config.SAMPLE_RATE // 1000)
        tail = (tail * 10**(tail_dbfs / 20)).astype(np.float32)
        samples = np.concatenate([samples[:speech_end(samples)], tail])
    if pad_ms:
        samples = np.concatenate([
            samples,
            np.zeros(pad_ms * config.SAMPLE_RATE // 1000, np.float32)
        ])
    return samples


def summarize(rows: list[dict], key: str, thresholds: list[float]) -> dict:
    states = np.array([r["state"] for r in rows])
    sources = np.array([r["source"] for r in rows])
    probs = np.array([r[key] for r in rows])
    binary = np.isin(states, ["complete", "incomplete"])
    complete = states == "complete"
    out = {"share_ended": {}, "complete_vs_incomplete": {}, "auc": {}}
    for threshold in thresholds:
        ended = probs > threshold
        shares = {}
        for state in STATES:
            for source in ("all", "real", "synthetic"):
                mask = (states == state) & (
                    (sources == source) if source != "all" else True)
                shares[f"{state}/{source}"] = float(
                    ended[mask].mean()) if mask.any() else None
        out["share_ended"][str(threshold)] = shares
        scores = {}
        for source in ("all", "real", "synthetic"):
            mask = binary & ((sources == source) if source != "all" else True)
            y, d = complete[mask], ended[mask]
            if not (y.any() and (~y).any()):
                continue
            scores[source] = {
                "n": int(mask.sum()),
                "accuracy": float((d == y).mean()),
                "acc_complete": float(d[y].mean()),
                "acc_incomplete": float((~d[~y]).mean()),
                "miss": float((~d[y]).mean()),
                "false_alarm": float(d[~y].mean()),
            }
        out["complete_vs_incomplete"][str(threshold)] = scores
    for source in ("all", "real", "synthetic"):
        mask = binary & ((sources == source) if source != "all" else True)
        y = complete[mask]
        if y.any() and (~y).any():
            out["auc"][source] = float(metrics.roc_auc_score(y, probs[mask]))
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    thresholds = [float(t) for t in args.thresholds]
    rows = (load_easyturn if args.format == "easyturn" else load_fastturn)(
        args.root)
    rows = rows[:args.limit]
    system = common.SavaSystem(args.model, args.device)
    baseline = smartturn.SmartTurn(args.smart_turn) if args.smart_turn else None
    for i, row in enumerate(rows):
        samples = prepare(audio.load(row["wav"]),
                          row["key"],
                          anchor_ms=args.anchor_ms,
                          tail_dbfs=args.tail_dbfs,
                          pad_ms=args.pad_ms)
        row["sava"] = float(
            system.turn(common.Recording(row["key"], samples, []))[-1])
        if baseline is not None:
            row["smart_turn"] = baseline.probability(samples)
        if (i + 1) % 1000 == 0:
            print(f"{i + 1} clips", flush=True)
    result = {
        "clips": len(rows),
        "format": args.format,
        "pad_ms": args.pad_ms,
        "anchor_ms": args.anchor_ms,
        "systems": {
            "sava": summarize(rows, "sava", thresholds)
        }
    }
    if baseline is not None:
        result["systems"]["smart_turn"] = summarize(rows, "smart_turn",
                                                    thresholds)
    for name, scores in result["systems"].items():
        for threshold, shares in scores["share_ended"].items():
            ended = " ".join(
                f"{s} {100 * shares[f'{s}/all']:5.1f}%" for s in STATES)
            print(f"{name:10s} @{threshold}: ended {ended} "
                  f"| auc {scores['auc']['all']:.4f}")
    suffix = f"_anchor{args.anchor_ms}" if args.anchor_ms is not None else (
        f"_pad{args.pad_ms}" if args.pad_ms else "")
    common.write_result(f"{args.format}{suffix}", result)


if __name__ == "__main__":
    main()
