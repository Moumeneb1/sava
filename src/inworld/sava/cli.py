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
"""The command line.

    sava run recording.wav [--curves out.json] [--turns_dir dir]
        Cuts a recording into turns with the reference policy, prints them and
        optionally writes the per-frame curves and one wav per kept turn.
    sava mic [--seconds N] [--turns_dir dir]
        The same on the microphone, live (needs the ``mic`` extra).

The model is ``--model`` (a registered Hub repository such as inworld/sava,
an hf:// location or a file) and
the policy's dials are flags of the same names as ``segmenter.Policy``.
"""

import argparse
import json
import os
import queue
import time

import numpy as np

from inworld.sava import audio
from inworld.sava import checkpoint
from inworld.sava import config
from inworld.sava import segmenter
from inworld.sava import streaming


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses a recording or microphone command and its turn policy."""
    parser = argparse.ArgumentParser(
        prog="sava",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "mic"))
    parser.add_argument("path", nargs="?", help="run: input audio file.")
    parser.add_argument(
        "--model",
        default="inworld/sava",
        help="Registered Hub repository, hf:// location or checkpoint file.")
    parser.add_argument("--device", default="cpu", help="torch device.")
    parser.add_argument("--curves", help="run: per-frame curves JSON file.")
    parser.add_argument("--turns_dir",
                        "--turns-dir",
                        help="Write one wav per kept turn into this directory.")
    parser.add_argument("--seconds",
                        type=float,
                        default=60.0,
                        help="mic: how long to listen.")
    for name, kind in (("speech_threshold", float), ("turn_threshold", float),
                       ("pre_roll_ms", int), ("min_silence_before_check_ms",
                                              int), ("max_silence_ms", int),
                       ("min_speech_ms", int), ("min_confident_speech_ms", int),
                       ("confident_speech_threshold", float)):
        parser.add_argument("--" + name,
                            "--" + name.replace("_", "-"),
                            type=kind,
                            default=getattr(segmenter.Policy, name))
    args = parser.parse_intermixed_args(argv)
    if args.command == "run" and args.path is None:
        parser.error("run requires an audio file")
    if args.command == "mic" and args.path is not None:
        parser.error("mic does not take an audio file")
    return args


def policy_from_args(args: argparse.Namespace) -> segmenter.Policy:
    """Builds the turn policy from the parsed command-line options."""
    return segmenter.Policy(
        speech_threshold=args.speech_threshold,
        turn_threshold=args.turn_threshold,
        pre_roll_ms=args.pre_roll_ms,
        min_silence_before_check_ms=args.min_silence_before_check_ms,
        max_silence_ms=args.max_silence_ms,
        min_speech_ms=args.min_speech_ms,
        min_confident_speech_ms=args.min_confident_speech_ms,
        confident_speech_threshold=args.confident_speech_threshold)


def format_turn(turn: segmenter.Turn) -> str:
    """Formats one turn for the command-line report."""
    kept = "kept" if turn.kept else "dropped"
    return (f"{turn.start_seconds:8.2f} {turn.end_seconds:8.2f}  "
            f"speech {turn.speech_ms:5d} ms"
            f"  closed by {turn.reason:7s} {kept}")


def write_turns(turns: list[segmenter.Turn], samples: np.ndarray,
                directory: str) -> None:
    """Writes each kept turn to its own wav file."""
    os.makedirs(directory, exist_ok=True)
    for i, turn in enumerate(turn for turn in turns if turn.kept):
        start = turn.start * config.FRAME_SAMPLES
        end = turn.end * config.FRAME_SAMPLES
        audio.save(os.path.join(directory, f"turn_{i:03d}.wav"),
                   samples[start:end])


def run_file(path: str,
             *,
             model_source: str = "inworld/sava",
             device: str = "cpu",
             policy: segmenter.Policy | None = None,
             curves_path: str | None = None,
             turns_dir: str | None = None) -> list[segmenter.Turn]:
    """Segments a recording, reports its turns and writes requested outputs."""
    model = checkpoint.load(model_source, device)
    samples = audio.load(path)
    curves = streaming.run(model, samples)
    turns = segmenter.segment(curves, policy or segmenter.Policy(),
                              model.lookahead_frames)
    for turn in turns:
        print(format_turn(turn))
    print(f"{len(turns)} turns, {sum(t.kept for t in turns)} kept, "
          f"{samples.size / config.SAMPLE_RATE:.1f} s of audio")
    if curves_path:
        with open(curves_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "frame_seconds": config.FRAME_SECONDS,
                    "speech": curves.speech.tolist(),
                    "turn": curves.turn.tolist()
                }, f)
    if turns_dir:
        write_turns(turns, samples, turns_dir)
    return turns


def run_mic(*,
            model_source: str = "inworld/sava",
            device: str = "cpu",
            policy: segmenter.Policy | None = None,
            seconds: float = 60.0,
            turns_dir: str | None = None) -> list[segmenter.Turn]:
    """Segments live microphone audio and optionally saves the kept turns."""
    try:
        # Microphone input is optional, including when importing SAVA.
        import sounddevice  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            'Microphone input requires pip install "inworld-sava[mic]".'
        ) from exc

    model = checkpoint.load(model_source, device)
    session = streaming.Session(model)
    cutter = segmenter.TurnSegmenter(policy or segmenter.Policy(),
                                     model.lookahead_frames)
    pieces: queue.Queue = queue.Queue()
    heard, turns = [], []

    def callback(indata, frames, time_info, status):
        del frames, time_info, status
        pieces.put(indata[:, 0].copy())

    print(f"listening for {seconds:.0f} s (ctrl-c stops early)", flush=True)
    deadline = time.monotonic() + seconds
    try:
        with sounddevice.InputStream(samplerate=config.SAMPLE_RATE,
                                     channels=1,
                                     dtype="float32",
                                     blocksize=config.FRAME_SAMPLES,
                                     callback=callback):
            while time.monotonic() < deadline:
                piece = pieces.get(timeout=1.0)
                heard.append(piece)
                for turn in cutter.feed(session.push(piece)):
                    turns.append(turn)
                    print(format_turn(turn), flush=True)
    except KeyboardInterrupt:
        pass
    for turn in cutter.feed(session.flush()) + cutter.finish():
        turns.append(turn)
        print(format_turn(turn), flush=True)
    if turns_dir and heard:
        write_turns(turns, np.concatenate(heard), turns_dir)
    return turns


def main(argv: list[str] | None = None) -> None:
    """Runs the selected command with an explicit set of options."""
    args = parse_args(argv)
    policy = policy_from_args(args)
    if args.command == "run":
        run_file(args.path,
                 model_source=args.model,
                 device=args.device,
                 policy=policy,
                 curves_path=args.curves,
                 turns_dir=args.turns_dir)
    else:
        run_mic(model_source=args.model,
                device=args.device,
                policy=policy,
                seconds=args.seconds,
                turns_dir=args.turns_dir)


def entry() -> None:
    """Runs the command for installations using the former entry point."""
    main()


if __name__ == "__main__":
    main()
