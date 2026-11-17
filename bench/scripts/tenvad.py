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
"""The TEN-VAD test set: 30 hand-labelled recordings, frame scores.

Data: the ``testset/`` directory of https://github.com/TEN-framework/ten-vad
(``<name>.wav`` and ``<name>.scv`` with rows ``name,start,end,label,...``).

    python bench/scripts/tenvad.py --root <ten-vad>/testset --model <weights>
"""

import argparse
import glob
import os

import common


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses this benchmark's command line without global flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",
                        required=True,
                        help="The TEN-VAD testset directory.")
    parser.add_argument("--model",
                        default="inworld/sava",
                        help="Checkpoint source.")
    parser.add_argument("--device", default="cpu", help="")
    common.add_boolean_argument(parser,
                                "silero",
                                default=True,
                                description="Score Silero VAD too.")
    return parser.parse_args(argv)


def load(root: str) -> list[common.Recording]:
    recordings = []
    for label_path in sorted(glob.glob(os.path.join(root, "*.scv"))):
        with open(label_path, encoding="utf-8") as f:
            fields = f.read().strip().split(",")
        triples = [fields[i:i + 3] for i in range(1, len(fields) - 2, 3)]
        segments = [(float(s), float(e))
                    for s, e, label in triples
                    if label.strip() == "1"]
        name = os.path.basename(label_path)[:-4]
        recordings.append(
            common.Recording(name, common.load_wav(label_path[:-4] + ".wav"),
                             segments))
    if not recordings:
        raise FileNotFoundError(f"No .scv files under [{root}].")
    return recordings


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    recordings = load(args.root)
    result = common.score_recordings(
        recordings, common.systems(args.model, args.device, args.silero))
    for name, scores in result["systems"].items():
        print(f"{name:8s} auc {scores['auc']:.4f} f1 {scores['f1']:.4f} "
              f"recall {scores['recall']:.4f}")
    common.write_result("tenvad", result)


if __name__ == "__main__":
    main()
