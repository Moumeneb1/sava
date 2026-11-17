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
"""The 16 kHz audio and manifests of the sets ``vadsuite.py`` scores, built
from the public downloads.

``--downloads`` is the set's folder with the publisher's files unpacked:

- ``aishell4``: ``test/`` with ``TextGrid/*.rttm`` and ``wav/*.flac``;
- ``alimeeting``: ``Test_Ali/`` with ``Test_Ali_far/`` and ``Test_Ali_near/``,
  each holding ``audio_dir/`` and ``textgrid_dir/``;
- ``ava``: ``ava_speech_labels_v1.csv`` and
  ``ava_file_names_trainval_v2.1.txt``;
- ``earnings21``: ``media/*.mp3`` and ``rttms/*.rttm``;
- ``libriparty``: ``LibriParty/dataset/eval/``;
- ``msdwild``: ``raw_wav/*.wav`` and ``all.rttm``;
- ``esc50``: ``ESC-50-master/``.

It writes ``<out_dir>/<set>/audio16k/``, mono 16 kHz wavs, and
``<out_dir>/<set>/manifest.jsonl``, one row per recording: ``id``, ``name``,
``split``, ``wav`` (relative to ``<out_dir>/<set>``), ``seconds`` and
``segments``,
the union of the annotated speech in seconds. AVA-Speech audio is seconds 900
to 1800 of each video on the CVDF mirror. Channel 0 of the AISHELL-4
recordings and of the AliMeeting arrays is kept; every other file is averaged
to mono. On the AliMeeting headsets ``segments`` is the wearer's own speech.
Audio already in the output folder is kept.
Earnings-21 and AVA-Speech need ``ffmpeg``.

    python bench/scripts/build_vadsuite.py --set aishell4 --downloads <folder>
"""

import argparse
from collections import abc
from concurrent import futures
import glob
import json
import os
import subprocess
import typing

import numpy as np
import soundfile
import soxr

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETS = ("aishell4", "alimeeting", "ava", "earnings21", "libriparty", "msdwild",
        "esc50")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses this benchmark's command line without global flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set",
                        required=True,
                        choices=list(SETS),
                        help="The set to build.")
    parser.add_argument("--downloads",
                        required=True,
                        help="The set's download folder.")
    parser.add_argument("--out_dir",
                        "--out-dir",
                        default=os.path.join(ROOT, "data", "vadsuite"),
                        help="Where the set's folder is written.")
    parser.add_argument("--workers",
                        default=8,
                        type=int,
                        help="Processes converting audio.")
    return parser.parse_args(argv)


SAMPLE_RATE = 16000
AVA_MIRROR = "https://s3.amazonaws.com/ava-dataset/trainval/"
AVA_START_SECONDS = 900
AVA_SECONDS = 900
AVA_SPEECH = ("CLEAN_SPEECH", "SPEECH_WITH_MUSIC", "SPEECH_WITH_NOISE")
AVA_ATTEMPTS = 3


def merge(segments: abc.Iterable[tuple[float, float]]) -> list[list[float]]:
    """Segments sorted, empty ones dropped, overlapping or touching ones
    joined."""
    out = []
    for start, end in sorted(
        (float(s), float(e)) for s, e in segments if float(e) > float(s)):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def clip(segments: list[list[float]], seconds: float) -> list[list[float]]:
    """``segments`` cut at ``seconds``, those past it dropped."""
    return [[s, min(e, seconds)]
            for s, e in segments
            if s < seconds and min(e, seconds) > s]


def parse_textgrid(path: str) -> dict[str, list[tuple[float, float, str]]]:
    """``{tier: [(start, end, text)]}`` of a long-format TextGrid."""
    # Pylint loses the dictionary type across the interval state resets.
    # Each use below is guarded by ``interval is not None``.
    # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig", "replace")
    tiers = {}
    name = None
    interval: dict[str, float] | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("item ["):
            name = None
            interval = None
        elif line.startswith("name = ") and interval is None:
            name = line[len("name = "):].strip().strip('"')
            tiers.setdefault(name, [])
        elif line.startswith("intervals ["):
            interval = {}
        elif interval is not None and line.startswith("xmin = "):
            interval["start"] = float(line.split("=", 1)[1])
        elif interval is not None and line.startswith("xmax = "):
            interval["end"] = float(line.split("=", 1)[1])
        elif interval is not None and line.startswith("text = "):
            value = line[len("text = "):].strip()
            value = value[1:] if value.startswith('"') else value
            value = value[:-1] if value.endswith('"') else value
            tiers[name].append((interval["start"], interval["end"], value))
            interval = None
    return tiers


def parse_rttm(path: str) -> dict[str, dict[str, list[list[float]]]]:
    """``{file id: {speaker: [[start, end]]}}`` of an RTTM's SPEAKER lines."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0] != "SPEAKER":
                continue
            start, duration = float(fields[3]), float(fields[4])
            out.setdefault(fields[1],
                           {}).setdefault(fields[7],
                                          []).append([start, start + duration])
    return out


def spoken(
    tiers: dict[str, list[tuple[float, float,
                                str]]]) -> list[tuple[float, float]]:
    """The intervals of every tier that carry text."""
    return [(start, end)
            for intervals in tiers.values()
            for start, end, text in intervals
            if text.strip()]


def ava_segments(conditions: list[list[typing.Any]],
                 seconds: float) -> list[list[float]]:
    """AVA-Speech's speech intervals joined where they touch, cut at
    ``seconds``."""
    segments = []
    for start, end, label in sorted(conditions):
        if label not in AVA_SPEECH:
            continue
        if segments and start <= segments[-1][1] + 1e-6:
            segments[-1][1] = max(segments[-1][1], end)
        else:
            segments.append([start, end])
    return [[s, min(e, seconds)] for s, e in segments if s < seconds]


def recording(dataset: str, split: str, name: str, source: str,
              channel: int | None,
              spans: list[tuple[float, float]]) -> dict[str, typing.Any]:
    return {
        "id": f"{dataset}/{split}/{name}",
        "name": name,
        "split": split,
        "wav": f"audio16k/{split}/{name}.wav",
        "source": source,
        "channel": channel,
        "spans": spans,
    }


def speech_of(speakers: dict[str, list[list[float]]]) -> list[list[float]]:
    return [span for spans in speakers.values() for span in spans]


def aishell4(downloads: str) -> list[dict[str, typing.Any]]:
    rows = []
    for rttm in sorted(
            glob.glob(os.path.join(downloads, "test", "TextGrid", "*.rttm"))):
        name = os.path.basename(rttm)[:-len(".rttm")]
        wav = os.path.join(downloads, "test", "wav", f"{name}.flac")
        speakers = parse_rttm(rttm).get(name, {})
        rows.append(
            recording("aishell4", "test", name, wav, 0, speech_of(speakers)))
    return rows


def alimeeting(downloads: str) -> list[dict[str, typing.Any]]:
    far = os.path.join(downloads, "Test_Ali", "Test_Ali_far")
    near = os.path.join(downloads, "Test_Ali", "Test_Ali_near")
    sessions = {
        os.path.basename(path)[:-len(".TextGrid")]: parse_textgrid(path)
        for path in sorted(
            glob.glob(os.path.join(far, "textgrid_dir", "*.TextGrid")))
    }
    rows = []
    for wav in sorted(glob.glob(os.path.join(far, "audio_dir", "*.wav"))):
        name = os.path.basename(wav)[:-len(".wav")]
        session = "_".join(name.split("_")[:2])
        rows.append(
            recording("alimeeting", "test", name, wav, 0,
                      spoken(sessions[session])))
    for wav in sorted(glob.glob(os.path.join(near, "audio_dir", "*.wav"))):
        name = os.path.basename(wav)[:-len(".wav")]
        own = parse_textgrid(
            os.path.join(near, "textgrid_dir", f"{name}.TextGrid"))
        rows.append(
            recording("alimeeting", "test", name, wav, None, spoken(own)))
    return rows


def earnings21(downloads: str) -> list[dict[str, typing.Any]]:
    rows = []
    for rttm in sorted(glob.glob(os.path.join(downloads, "rttms", "*.rttm"))):
        name = os.path.basename(rttm)[:-len(".rttm")]
        media = os.path.join(downloads, "media", f"{name}.mp3")
        speakers = parse_rttm(rttm).get(name, {})
        rows.append(
            recording("earnings21", "test", name, media, None,
                      speech_of(speakers)))
    return rows


def libriparty(downloads: str) -> list[dict[str, typing.Any]]:
    pattern = os.path.join(downloads, "LibriParty", "dataset", "eval",
                           "session_*", "session_*.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)[:-len(".json")]
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        spans = [(utterance["start"], utterance["stop"])
                 for key, utterances in meta.items()
                 if key not in ("noises", "background")
                 for utterance in utterances]
        wav = os.path.join(os.path.dirname(path), f"{name}_mixture.wav")
        rows.append(recording("libriparty", "test", name, wav, None, spans))
    return rows


def msdwild(downloads: str) -> list[dict[str, typing.Any]]:
    speakers = parse_rttm(os.path.join(downloads, "all.rttm"))
    wavs = sorted(glob.glob(os.path.join(downloads, "raw_wav", "*.wav")))
    if len(wavs) != len(speakers):
        raise ValueError(
            f"{len(wavs)} wavs but {len(speakers)} recordings in all.rttm.")
    rows = []
    for wav in wavs:
        name = os.path.basename(wav)[:-len(".wav")]
        rows.append(
            recording("msdwild", "test", name, wav, None,
                      speech_of(speakers.get(name, {}))))
    return rows


def esc50(downloads: str) -> list[dict[str, typing.Any]]:
    root = os.path.join(downloads, "ESC-50-master")
    with open(os.path.join(root, "meta", "esc50.csv"), encoding="utf-8") as f:
        files = [line.split(",")[0] for line in f.read().splitlines()[1:]]
    return [
        recording("esc50", "test", name[:-len(".wav")],
                  os.path.join(root, "audio", name), None, []) for name in files
    ]


READERS = {
    "aishell4": aishell4,
    "alimeeting": alimeeting,
    "earnings21": earnings21,
    "libriparty": libriparty,
    "msdwild": msdwild,
    "esc50": esc50,
}


def decode_mp3(path: str) -> tuple[np.ndarray, int]:
    """``[samples, channels]`` float32 and the sample rate, through ffmpeg."""
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=sample_rate,channels", "-of", "json", path
    ],
                           capture_output=True,
                           check=True,
                           text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    pcm = subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-i", path, "-f", "f32le",
        "-acodec", "pcm_f32le", "-"
    ],
                         capture_output=True,
                         check=True).stdout
    samples = np.frombuffer(pcm,
                            dtype=np.float32).reshape(-1,
                                                      int(stream["channels"]))
    return samples, int(stream["sample_rate"])


def decode(path: str, channel: int | None) -> np.ndarray:
    """Mono float32 at 16 kHz: ``channel`` of the file, or the mean of its
    channels."""
    if path.endswith(".mp3"):
        samples, rate = decode_mp3(path)
    else:
        samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    samples = samples[:,
                      channel] if channel is not None else samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        samples = soxr.resample(samples, rate, SAMPLE_RATE)
    return np.ascontiguousarray(samples, dtype=np.float32)


def convert(job: tuple[str, int | None, str]) -> float:
    """Writes ``out`` from ``source`` unless it exists; its length in
    seconds."""
    source, channel, out = job
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        soundfile.write(f"{out}.tmp.wav",
                        decode(source, channel),
                        SAMPLE_RATE,
                        subtype="FLOAT")
        os.replace(f"{out}.tmp.wav", out)
    return soundfile.info(out).frames / SAMPLE_RATE


def fetch_ava(job: tuple[str, str]) -> bool:
    """Seconds 900 to 1800 of an AVA video on the mirror as a 16 kHz mono wav;
    whether it was written."""
    file_name, out = job
    os.makedirs(os.path.dirname(out), exist_ok=True)
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-ss",
        str(AVA_START_SECONDS), "-i", AVA_MIRROR + file_name, "-t",
        str(AVA_SECONDS), "-vn", "-ac", "1", "-ar",
        str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-y", f"{out}.tmp.wav"
    ]
    for _ in range(AVA_ATTEMPTS):
        if subprocess.run(command, check=False).returncode == 0:
            os.replace(f"{out}.tmp.wav", out)
            return True
    return False


def run_all(function: abc.Callable[[typing.Any], typing.Any],
            jobs: list[typing.Any], workers: int) -> list[typing.Any]:
    if workers == 1:
        return [function(job) for job in jobs]
    with futures.ProcessPoolExecutor(workers) as pool:
        return list(pool.map(function, jobs, chunksize=4))


def ava(downloads: str, folder: str,
        workers: int) -> list[dict[str, typing.Any]]:
    conditions = {}
    with open(os.path.join(downloads, "ava_speech_labels_v1.csv"),
              encoding="utf-8") as f:
        for line in f:
            video, start, end, label = line.strip().split(",")
            conditions.setdefault(video, []).append([
                float(start) - AVA_START_SECONDS,
                float(end) - AVA_START_SECONDS, label
            ])
    with open(os.path.join(downloads, "ava_file_names_trainval_v2.1.txt"),
              encoding="utf-8") as f:
        files = {os.path.splitext(name)[0]: name for name in f.read().split()}
    wavs = {
        video: os.path.join(folder, "audio16k", f"{video}.wav")
        for video in sorted(conditions)
        if video in files
    }
    run_all(fetch_ava, [(files[video], wav)
                        for video, wav in wavs.items()
                        if not os.path.exists(wav)], workers)
    rows = []
    for video, wav in wavs.items():
        if not os.path.exists(wav):
            continue
        seconds = soundfile.info(wav).frames / SAMPLE_RATE
        rows.append({
            "id": f"ava/test/{video}",
            "name": video,
            "split": "test",
            "wav": f"audio16k/{video}.wav",
            "seconds": seconds,
            "segments": ava_segments(conditions[video], seconds),
        })
    return rows


def manifest_row(row: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """The manifest's fields of a built row, segments to the millisecond."""
    return {
        "id": row["id"],
        "name": row["name"],
        "split": row["split"],
        "wav": row["wav"],
        "seconds": row["seconds"],
        "segments": [[round(s, 3), round(e, 3)] for s, e in row["segments"]],
    }


def build(name: str, downloads: str, out: str,
          workers: int) -> list[dict[str, typing.Any]]:
    """Writes a set's audio16k folder and manifest and returns its rows."""
    folder = os.path.join(out, name)
    if name == "ava":
        rows = ava(downloads, folder, workers)
    else:
        rows = READERS[name](downloads)
        jobs = [(row["source"], row["channel"],
                 os.path.join(folder, row["wav"])) for row in rows]
        for row, seconds in zip(rows,
                                run_all(convert, jobs, workers),
                                strict=True):
            row["seconds"] = seconds
            row["segments"] = clip(merge(row["spans"]), seconds)
    manifest = [manifest_row(row) for row in rows]
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "manifest.jsonl")
    with open(f"{path}.tmp", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in manifest)
    os.replace(f"{path}.tmp", path)
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = build(args.set, args.downloads, args.out_dir, args.workers)
    hours = sum(row["seconds"] for row in rows) / 3600
    print(f"{args.set}: {len(rows)} recordings, {hours:.1f} h in "
          f"{os.path.join(args.out_dir, args.set)}")


if __name__ == "__main__":
    main()
