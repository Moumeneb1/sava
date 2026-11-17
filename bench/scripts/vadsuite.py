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
"""Public voice-activity sets with speaker annotations, and ESC-50 noise.

Each set is the folder ``build_vadsuite.py`` writes: 16 kHz mono wavs and
``manifest.jsonl`` (one row per recording: ``wav`` relative to the folder,
``segments`` the union of its annotated speech, in seconds). ESC-50 has no
speech: it is scored as the share of clips with no run of 100 ms (three
frames) above 0.5, the share of clips with no frame above 0.5, and the share
of frames above 0.5.

    python bench/scripts/vadsuite.py --set aishell4 \
        --root data/vadsuite/aishell4 --model <weights>
"""

import argparse
import json
import os

import numpy as np

import common


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses this benchmark's command line without global flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True, help="The set's name.")
    parser.add_argument("--root",
                        required=True,
                        help="The set's folder written by build_vadsuite.py.")
    parser.add_argument("--model",
                        default="inworld/sava",
                        help="Checkpoint source.")
    parser.add_argument("--device", default="cpu", help="")
    common.add_boolean_argument(parser,
                                "silero",
                                default=True,
                                description="Score Silero VAD too.")
    parser.add_argument("--limit",
                        default=None,
                        type=int,
                        help="Recordings to score (all by default).")
    return parser.parse_args(argv)


RUN_FRAMES = 3  # 100 ms of 40 ms frames, rounded up.


def load(unused_name: str, root: str,
         limit: int | None) -> list[common.Recording]:
    manifest = os.path.join(root, "manifest.jsonl")
    recordings = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            recordings.append(
                common.Recording(
                    row["id"], common.load_wav(os.path.join(root, row["wav"])),
                    [tuple(s) for s in row["segments"]]))
            if limit is not None and len(recordings) >= limit:
                break
    return recordings


def noise_scores(recordings: list[common.Recording],
                 systems_under_test: list) -> dict:
    out = {"clips": len(recordings), "systems": {}}
    for system in systems_under_test:
        curves = [system.speech(r) for r in recordings]
        clean, never_above = [], []
        for probs in curves:
            above = (probs > common.THRESHOLD).astype(int)
            runs = np.convolve(above,
                               np.ones(RUN_FRAMES, dtype=int),
                               mode="valid")
            clean.append(not (runs >= RUN_FRAMES).any())
            never_above.append(not above.any())
        pooled = np.concatenate(curves)
        out["systems"][system.name] = {
            "clips_without_a_false_trigger":
                float(np.mean(clean)),
            "clips_without_a_frame_above_threshold":
                float(np.mean(never_above)),
            "frames_above_threshold":
                float((pooled > common.THRESHOLD).mean()),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    recordings = load(args.set, args.root, args.limit)
    under_test = common.systems(args.model, args.device, args.silero)
    if args.set == "esc50":
        result = noise_scores(recordings, under_test)
        for name, scores in result["systems"].items():
            print(f"{name:8s} clean clips "
                  f"{100 * scores['clips_without_a_false_trigger']:.1f}%")
    else:
        result = common.score_recordings(recordings, under_test)
        for name, scores in result["systems"].items():
            print(f"{name:8s} auc {scores['auc']:.4f} f1 {scores['f1']:.4f}")
    common.write_result(f"vadsuite_{args.set}", result)


if __name__ == "__main__":
    main()
