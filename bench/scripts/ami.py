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
"""AMI meetings, policy-free end-of-turn recall and latency on real turn ends.

Each participant's headset channel is one stream from its first sample. A
turn end is a moment T where the channel's owner stops speaking and the floor
passes to another speaker; ``bench/data/ami_turn_ends.jsonl`` lists them per
channel with ``floor_free``, the seconds before anyone else is audible in the
channel. The protocol asks one thing: does the model's end-of-turn
probability cross a fixed threshold by T + d, and how soon. A turn end counts
at delay d only when its floor stays free for d, and the first crossing is
looked for before min(T + floor_free, T + 1.6 s). Within-turn pauses are not
labelled, so there are no negatives and nothing to tune.

SAVA's probability is its per-frame curve over the whole channel. smart-turn
is read the way a pipeline would call it: on the last 8 s of the channel at
the end of every frame after T, until it crosses or the window closes.

Data: the AMI corpus (CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/):
``<root>/<meeting>/audio/<meeting>.Headset-<headset>.wav``, 16 kHz mono.

    python bench/scripts/ami.py --root <amicorpus> --model <weights>
        [--smart_turn smart-turn-v3.2-gpu.onnx]
"""

import argparse
from collections import abc
import json
import math
import os

import numpy as np

import common
from inworld.sava import config
import smartturn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses this benchmark's command line without global flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",
                        required=True,
                        help="The AMI corpus directory.")
    parser.add_argument("--manifest",
                        default=os.path.join(common.DATA,
                                             "ami_turn_ends.jsonl"),
                        help="")
    parser.add_argument("--model",
                        default="inworld/sava",
                        help="Checkpoint source.")
    parser.add_argument("--device", default="cpu", help="")
    parser.add_argument("--smart_turn",
                        "--smart-turn",
                        default=None,
                        help="smart-turn v3.2 ONNX file; scores it too.")
    parser.add_argument("--threads",
                        default=4,
                        type=int,
                        help="ONNX Runtime threads for smart-turn.")
    parser.add_argument("--thresholds",
                        default=["0.5"],
                        type=common.comma_separated,
                        help="End-of-turn thresholds to report.")
    common.add_boolean_argument(
        parser,
        "timed_only",
        default=True,
        description="Score only turn ends whose words carry timestamps.")
    return parser.parse_args(argv)


DELAYS = (0.2, 0.4, 0.6, 1.0)
HORIZON = 1.6
WINDOW = smartturn.WINDOW_SECONDS * config.SAMPLE_RATE


def wav_path(root: str, channel: dict) -> str:
    return os.path.join(
        root, channel["meeting"], "audio",
        f"{channel['meeting']}.Headset-{channel['headset']}.wav")


def frame_span(t: float, floor_free: float, frames: int) -> tuple[int, int]:
    """The frames read after T: the first one ending after T, up to the floor
    being taken or 1.6 s."""
    hop = config.FRAME_SECONDS
    k0 = math.ceil(t / hop)
    k1 = min(frames, math.ceil((t + min(floor_free, HORIZON)) / hop))
    return k0, k1


def crossings(probabilities: np.ndarray, k0: int, t: float,
              thresholds: list[float]) -> list[float | None]:
    """Seconds after T of the first frame, among ``probabilities`` read from
    frame ``k0`` on, that reaches each threshold."""
    out = []
    for threshold in thresholds:
        hits = np.flatnonzero(probabilities >= threshold)
        out.append((k0 + hits[0] + 1) * config.FRAME_SECONDS -
                   t if hits.size else None)
    return out


def curve_reader(turn: np.ndarray) -> abc.Callable[[int, int], np.ndarray]:
    return lambda k0, k1: turn[k0:k1]


def smart_turn_reader(model: smartturn.SmartTurn, samples: np.ndarray,
                      stop_at: float) -> abc.Callable[[int, int], np.ndarray]:
    """Reads smart-turn on the last 8 s at the end of each frame, stopping
    once ``stop_at`` is reached (no higher threshold is asked for)."""

    def read(k0: int, k1: int) -> np.ndarray:
        probabilities = []
        for k in range(k0, k1):
            end = (k + 1) * config.FRAME_SAMPLES
            probabilities.append(
                model.probability(samples[max(0, end - WINDOW):end]))
            if probabilities[-1] >= stop_at:
                break
        return np.array(probabilities)

    return read


def score(ends: list[tuple[float, float, list[float | None]]],
          thresholds: list[float]) -> dict:
    """``ends``: (t, floor_free, first crossing per threshold) -> shares fired
    by each delay and the latency quantiles of the crossings."""
    out = {}
    for i, threshold in enumerate(thresholds):
        fired = {}
        for d in DELAYS:
            eligible = [
                first[i] for _, floor_free, first in ends if floor_free >= d
            ]
            fired[str(d)] = float(
                np.mean([f is not None and f <= d + 1e-9 for f in eligible
                        ])) if eligible else None
        latencies = np.array(
            [first[i] for _, _, first in ends if first[i] is not None])
        out[str(threshold)] = {
            "fired_by":
                fired,
            "latency_p50_ms":
                float(1000 *
                      np.quantile(latencies, 0.5)) if latencies.size else None,
            "latency_p90_ms":
                float(1000 *
                      np.quantile(latencies, 0.9)) if latencies.size else None,
            "crossed":
                int(latencies.size),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    thresholds = [float(t) for t in args.thresholds]
    system = common.SavaSystem(args.model, args.device)
    baseline = smartturn.SmartTurn(args.smart_turn,
                                   args.threads) if args.smart_turn else None
    ends = {"sava": []}
    if baseline is not None:
        ends["smart_turn"] = []
    with open(args.manifest, encoding="utf-8") as f:
        channels = [json.loads(line) for line in f]
    for channel in channels:
        samples = common.load_wav(wav_path(args.root, channel))
        turn = system.turn(common.Recording(channel["id"], samples, []))
        readers = {"sava": curve_reader(turn)}
        if baseline is not None:
            readers["smart_turn"] = smart_turn_reader(baseline, samples,
                                                      max(thresholds))
        for end in channel["turn_ends"]:
            if args.timed_only and not end["timed"]:
                continue
            k0, k1 = frame_span(end["t"], end["floor_free"], turn.size)
            for name, read in readers.items():
                first = crossings(read(k0, k1), k0, end["t"], thresholds)
                ends[name].append((end["t"], end["floor_free"], first))
        print(f"{channel['id']}: {len(channel['turn_ends'])} turn ends",
              flush=True)
    counted = {
        str(d): int(sum(floor_free >= d for _, floor_free, _ in ends["sava"]))
        for d in DELAYS
    }
    result = {
        "turn_ends": len(ends["sava"]),
        "counted": counted,
        "systems": {
            name: {
                "thresholds": score(e, thresholds)
            } for name, e in ends.items()
        }
    }
    for name, scores in result["systems"].items():
        for threshold, s in scores["thresholds"].items():
            fired = " ".join(f"+{int(1000 * float(d))} ms {100 * v:5.1f}%"
                             for d, v in s["fired_by"].items())
            print(f"{name:10s} @{threshold}: {fired} | latency p50 / p90 "
                  f"{s['latency_p50_ms']:.0f} / {s['latency_p90_ms']:.0f} ms")
    common.write_result("ami", result)


if __name__ == "__main__":
    main()
