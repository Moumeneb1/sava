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
"""What every benchmark script shares: the frame grid and its labels, the
scores, the bootstrap, the systems under test and the result files.

Frames are the model's 40 ms frames. A frame is speech when a labelled
segment covers its centre. A system's probability for a frame is the one
it emitted for that frame: SAVA's speech verdict on the frame (read
``lookahead_frames`` later), Silero's probability of the 32 ms chunk holding
the frame's centre. Every curve is produced by the system's own streaming
path from the start of the recording.
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np
from sklearn import metrics
import torch

from inworld.sava import audio
from inworld.sava import checkpoint
from inworld.sava import config
from inworld.sava import streaming

RESULTS = os.environ.get(
    "BENCH_RESULTS",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"))
DATA = os.environ.get(
    "BENCH_DATA",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
THRESHOLD = 0.5
BOOTSTRAP = 2000
BINS = 8192


def boolean_value(value: str) -> bool:
    """Reads the boolean spellings accepted by the benchmark commands."""
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}.")


def add_boolean_argument(parser: argparse.ArgumentParser, name: str, *,
                         default: bool, description: str) -> None:
    """Adds a boolean option, including its original --no<name> spelling."""
    positive = ["--" + name]
    negative = ["--no" + name, "--no-" + name.replace("_", "-")]
    if "_" in name:
        positive.append("--" + name.replace("_", "-"))
    parser.add_argument(*positive,
                        type=boolean_value,
                        nargs="?",
                        const=True,
                        default=default,
                        help=description)
    parser.add_argument(*negative,
                        dest=name,
                        action="store_false",
                        help="Disable " + name.replace("_", " ") + ".")


def comma_separated(value: str) -> list[str]:
    """Reads a comma-separated list, retaining the benchmark flag format."""
    return [item.strip() for item in value.split(",")]


@dataclasses.dataclass
class Recording:
    name: str
    samples: np.ndarray  # Mono float32 at 16 kHz.
    segments: list[tuple[float, float]]  # Speech, in seconds.
    scored: list[tuple[float, float]] | None = None  # Spans scored; None: all.

    @property
    def frames(self) -> int:
        return streaming.num_frames(self.samples.size)


def frame_labels(recording: Recording) -> np.ndarray:
    """``[T]`` bool: a segment covers the frame's centre."""
    centres = (np.arange(recording.frames) + 0.5) * config.FRAME_SECONDS
    labels = np.zeros(recording.frames, dtype=bool)
    for start, end in recording.segments:
        labels |= (centres >= start) & (centres < end)
    return labels


def frame_mask(recording: Recording) -> np.ndarray:
    """``[T]`` bool: the frames the recording's scored spans cover."""
    if recording.scored is None:
        return np.ones(recording.frames, dtype=bool)
    centres = (np.arange(recording.frames) + 0.5) * config.FRAME_SECONDS
    mask = np.zeros(recording.frames, dtype=bool)
    for start, end in recording.scored:
        mask |= (centres >= start) & (centres < end)
    return mask


class SavaSystem:
    """The model under test: curves lined up per frame (``streaming.run``)."""

    def __init__(self, source: str, device: str = "cpu"):
        self.model = checkpoint.load(source, device)
        self.name = "sava"
        self.lag = self.model.lookahead_frames

    def speech(self, recording: Recording) -> np.ndarray:
        return streaming.run(self.model, recording.samples).speech

    def turn(self, recording: Recording) -> np.ndarray:
        return streaming.run(self.model, recording.samples).turn


class SileroSystem:
    """silero-vad (the pip package) on its native 512-sample chunks, states
    reset per recording; each frame takes the chunk holding its centre."""

    CHUNK = 512

    def __init__(self):
        try:
            # Silero is installed only with the eval extra.
            import silero_vad  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                'Silero evaluation requires pip install "inworld-sava[eval]".'
            ) from exc

        self.model = silero_vad.load_silero_vad()
        self.name = "silero"
        self.lag = 0

    @torch.no_grad()
    def chunk_probabilities(self, samples: np.ndarray) -> np.ndarray:
        self.model.reset_states()
        wav = torch.from_numpy(samples)
        wav = torch.nn.functional.pad(wav, (0, (-wav.numel()) % self.CHUNK))
        return np.array([
            float(self.model(wav[i:i + self.CHUNK], config.SAMPLE_RATE))
            for i in range(0, wav.numel(), self.CHUNK)
        ])

    def speech(self, recording: Recording) -> np.ndarray:
        probs = self.chunk_probabilities(recording.samples)
        centres = (np.arange(recording.frames) + 0.5) * config.FRAME_SAMPLES
        return probs[np.minimum((centres // self.CHUNK).astype(int),
                                probs.size - 1)]


def systems(source: str, device: str, silero: bool) -> list:
    out = [SavaSystem(source, device)]
    if silero:
        out.append(SileroSystem())
    return out


def frame_scores(labels: np.ndarray,
                 probs: np.ndarray,
                 threshold: float = THRESHOLD) -> dict[str, float]:
    """ROC-AUC, accuracy and speech F1 at ``threshold``, and the
    accuracy-best threshold on a 0.05 grid."""
    grid = np.linspace(0.05, 0.95, 19)
    best = max(grid, key=lambda t: ((probs > t) == labels).mean())
    return {
        "frames":
            int(labels.size),
        "speech_fraction":
            float(labels.mean()),
        "auc":
            float(metrics.roc_auc_score(labels, probs)),
        "accuracy":
            float(((probs > threshold) == labels).mean()),
        "f1":
            float(metrics.f1_score(labels, probs > threshold)),
        "recall":
            float(metrics.recall_score(labels, probs > threshold)),
        "precision":
            float(
                metrics.precision_score(labels,
                                        probs > threshold,
                                        zero_division=0)),
        "best_accuracy":
            float(((probs > best) == labels).mean()),
        "best_threshold":
            float(best),
    }


def binned_histograms(labels_per_recording: list[np.ndarray],
                      scores_per_recording: list[np.ndarray],
                      bins: int = BINS) -> tuple[np.ndarray, np.ndarray]:
    """Per recording, the counts of positive and negative frames in ``bins``
    quantile bins of the pooled scores: ``(positives, negatives)``, each
    ``[recordings, bins]``."""
    pooled = np.concatenate(scores_per_recording)
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    positives = np.zeros((len(labels_per_recording), bins), dtype=np.int64)
    negatives = np.zeros_like(positives)
    pairs = zip(labels_per_recording, scores_per_recording, strict=True)
    for i, (labels, scores) in enumerate(pairs):
        index = np.searchsorted(edges, scores, side="right")
        positives[i] = np.bincount(index[labels], minlength=bins)
        negatives[i] = np.bincount(index[~labels], minlength=bins)
    return positives, negatives


def binned_auc(positives: np.ndarray, negatives: np.ndarray) -> float:
    """The AUC of binned scores ``[bins]`` (bins ascending), ties inside a
    bin counting one half."""
    higher = positives.sum() - np.cumsum(positives)
    pairs = (negatives * higher).sum() + 0.5 * (negatives * positives).sum()
    return float(pairs / (positives.sum() * negatives.sum()))


def bootstrap_auc_difference(labels_per_recording: list[np.ndarray],
                             a_per_recording: list[np.ndarray],
                             b_per_recording: list[np.ndarray],
                             resamples: int = BOOTSTRAP,
                             seed: int = 0) -> dict[str, float]:
    """Paired bootstrap over recordings of the pooled AUC of ``a`` minus
    ``b``: the 95% interval and the share of resamples where ``a`` wins. The
    AUCs of a resample come from the recordings' score histograms."""
    rng = np.random.default_rng(seed)
    a_pos, a_neg = binned_histograms(labels_per_recording, a_per_recording)
    b_pos, b_neg = binned_histograms(labels_per_recording, b_per_recording)
    n = len(labels_per_recording)
    diffs = []
    for _ in range(resamples):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        if (counts @ a_pos).sum() == 0 or (counts @ a_neg).sum() == 0:
            continue
        diffs.append(
            binned_auc(counts @ a_pos, counts @ a_neg) -
            binned_auc(counts @ b_pos, counts @ b_neg))
    diffs = np.array(diffs)
    return {
        "low": float(np.percentile(diffs, 2.5)),
        "high": float(np.percentile(diffs, 97.5)),
        "wins": float((diffs > 0).mean())
    }


def curves_of(recordings: list[Recording],
              systems_under_test: list) -> dict[str, list]:
    """Each system's speech curve per recording, keyed by system name."""
    curves = {}
    for system in systems_under_test:
        started = time.time()
        curves[system.name] = [system.speech(r) for r in recordings]
        seconds = time.time() - started
        print(f"{system.name}: {len(recordings)} recordings in {seconds:.0f} s",
              flush=True)
    return curves


def score_recordings(recordings: list[Recording],
                     systems_under_test: list,
                     curves: dict[str, list] | None = None) -> dict:
    """Pooled frame scores per system over the scored frames of every
    recording, plus the paired bootstrap of SAVA's AUC against each other
    system. ``curves`` (``curves_of``) lets several readings of the same
    audio share one pass per system."""
    labels, masks = [], []
    for recording in recordings:
        labels.append(frame_labels(recording))
        masks.append(frame_mask(recording))
    if curves is None:
        curves = curves_of(recordings, systems_under_test)
    kept_labels = [lab[m] for lab, m in zip(labels, masks, strict=True)]
    out = {
        "recordings":
            len(recordings),
        "hours":
            sum(r.samples.size for r in recordings) / config.SAMPLE_RATE / 3600,
        "systems": {},
    }
    pooled_labels = np.concatenate(kept_labels)
    kept = {
        name: [c[m] for c, m in zip(cs, masks, strict=True)]
        for name, cs in curves.items()
    }
    for name, cs in kept.items():
        out["systems"][name] = frame_scores(pooled_labels, np.concatenate(cs))
    for name in kept:
        if name != "sava" and "sava" in kept:
            out["systems"][name][
                "sava_minus_this_auc"] = bootstrap_auc_difference(
                    kept_labels, kept["sava"], kept[name])
    return out


def write_result(name: str, result: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {path}", flush=True)
    return path


def load_wav(path: str) -> np.ndarray:
    return audio.load(path)
