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
"""Serves many simulated callers from one model with ``sava.Streams``.

Callers join one after another, each playing a recording in 20 ms packets.
Every 40 ms tick, the server pushes what the live callers sent as one batch,
cuts each caller's frames into turns with the reference policy and prints
them as they close. A caller whose recording ends is flushed, freeing its
slot for the next. The ticks run back to back rather than every 40 ms, so the
run ends with how much faster than real time it served.

    python examples/serve_many_streams.py tests/fixtures/speech.wav
    python examples/serve_many_streams.py a.wav b.wav --callers 256 \\
        --device cuda --cuda_graphs
"""

import argparse
import dataclasses
import time

import numpy as np

from inworld import sava

PACKET = 320  # 20 ms at 16 kHz
PACKETS_PER_TICK = 2


@dataclasses.dataclass
class Caller:
    """A caller on the line: its stream, its recording and its turns."""
    number: int
    stream: int
    samples: np.ndarray
    cutter: sava.TurnSegmenter
    sent: int = 0

    def next_packets(self) -> np.ndarray:
        end = min(self.sent + PACKETS_PER_TICK * PACKET, self.samples.size)
        packets, self.sent = self.samples[self.sent:end], end
        return packets

    @property
    def done(self) -> bool:
        return self.sent == self.samples.size

    def report(self, turns: list[sava.Turn]) -> None:
        for turn in turns:
            print(f"caller {self.number:3d}: {turn.start_seconds:6.2f}-"
                  f"{turn.end_seconds:6.2f} s, {turn.speech_ms} ms of speech, "
                  f"closed by {turn.reason}{'' if turn.kept else ', dropped'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("recordings",
                        nargs="+",
                        help="what the callers play, in turn")
    parser.add_argument("--model", default="inworld/sava")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda_graphs", action="store_true")
    parser.add_argument("--callers", type=int, default=16)
    parser.add_argument("--capacity",
                        type=int,
                        default=None,
                        help="open streams at most; all callers by default")
    parser.add_argument("--join_every",
                        type=int,
                        default=5,
                        help="ticks between one caller joining and the next")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model = sava.load(args.model, device=args.device)
    recordings = [sava.load_audio(path) for path in args.recordings]
    streams = sava.Streams(model,
                           capacity=args.capacity or args.callers,
                           cuda_graphs=args.cuda_graphs)
    waiting, live, served, tick = list(range(args.callers)), [], 0, 0
    start = time.perf_counter()
    while waiting or live:
        if (waiting and tick >= waiting[0] * args.join_every and
                len(live) < streams.capacity):
            number = waiting.pop(0)
            live.append(
                Caller(number, streams.open(),
                       recordings[number % len(recordings)],
                       sava.TurnSegmenter(sava.Policy(), streams.lag)))
        frames = streams.push({c.stream: c.next_packets() for c in live})
        for caller in live:
            caller.report(caller.cutter.feed(frames[caller.stream]))
        ended = [caller for caller in live if caller.done]
        frames = streams.flush(caller.stream for caller in ended)
        for caller in ended:
            caller.report(
                caller.cutter.feed(frames[caller.stream]) +
                caller.cutter.finish())
            served += caller.samples.size
            live.remove(caller)
        tick += 1
    elapsed = time.perf_counter() - start
    seconds = served / sava.SAMPLE_RATE
    print(f"{args.callers} callers, {seconds:.0f} s of audio in "
          f"{elapsed:.1f} s: {seconds / elapsed:.0f}x real time")


if __name__ == "__main__":
    main()
