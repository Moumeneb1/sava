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
"""VoxConverse test: broadcast and debate recordings, frame scores.

Data: the Hugging Face dataset ``diarizers-community/voxconverse`` downloaded
with ``huggingface-cli download --repo-type dataset`` (parquet shards named
``test-*``); speech is the union of its speaker segments.

Two readings of the same audio:

- ``diarization``: the dataset's own labels, every frame scored;
- ``curated``: ``bench/data/voxconverse_curated.jsonl``, word-level speech
  spans on the spans of each recording that passed a quality bar, only those
  spans scored. See bench/README.md for what the two disagree on.

    python bench/scripts/voxconverse.py --root <download> --model <weights>
"""

import argparse
import glob
import io
import json
import os

import soundfile

import common
from inworld.sava import audio


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses this benchmark's command line without global flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="The dataset download.")
    parser.add_argument("--model",
                        default="inworld/sava",
                        help="Checkpoint source.")
    parser.add_argument("--device", default="cpu", help="")
    common.add_boolean_argument(parser,
                                "silero",
                                default=True,
                                description="Score Silero VAD too.")
    parser.add_argument(
        "--curated",
        default=os.path.join(common.DATA, "voxconverse_curated.jsonl"),
        help="The curated manifest; empty skips the curated reading.")
    parser.add_argument("--limit",
                        default=None,
                        type=int,
                        help="Recordings to score (all by default).")
    return parser.parse_args(argv)


def load(root: str, limit: int | None) -> list[common.Recording]:
    try:
        # Dataset readers belong to the optional eval extra.
        from pyarrow import parquet  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            'Parquet input requires pip install "inworld-sava[eval]".') from exc
    recordings = []
    for path in sorted(
            glob.glob(os.path.join(root, "**", "test-*.parquet"),
                      recursive=True)):
        for row in parquet.read_table(path).to_pylist():
            samples, sample_rate = soundfile.read(io.BytesIO(
                row["audio"]["bytes"]),
                                                  dtype="float32",
                                                  always_2d=True)
            name = os.path.splitext(os.path.basename(row["audio"]["path"] or
                                                     ""))[0]
            recordings.append(
                common.Recording(
                    name or str(len(recordings)),
                    audio.to_model_rate(samples, sample_rate),
                    list(
                        zip(row["timestamps_start"],
                            row["timestamps_end"],
                            strict=True))))
            if limit is not None and len(recordings) >= limit:
                return recordings
    if not recordings:
        raise FileNotFoundError(f"No test-*.parquet under [{root}].")
    return recordings


def curated(recordings: list[common.Recording],
            manifest: str) -> list[common.Recording]:
    """The recordings with the manifest's speech spans and scored spans."""
    with open(manifest, encoding="utf-8") as f:
        rows = {row["recording"]: row for row in map(json.loads, f)}
    out = []
    for recording in recordings:
        row = rows[recording.name]
        if row["frames"] != recording.frames:
            raise ValueError(
                f"{recording.name}: {recording.frames} frames, the manifest "
                f"has {row['frames']}.")
        out.append(
            common.Recording(recording.name,
                             recording.samples,
                             [tuple(s) for s in row["speech"]],
                             scored=[tuple(s) for s in row["scored"]]))
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    recordings = load(args.root, args.limit)
    under_test = common.systems(args.model, args.device, args.silero)
    curves = common.curves_of(recordings, under_test)
    common.write_result("voxconverse",
                        common.score_recordings(recordings, under_test, curves))
    if args.curated:
        result = common.score_recordings(curated(recordings, args.curated),
                                         under_test, curves)
        common.write_result("voxconverse_curated", result)


if __name__ == "__main__":
    main()
