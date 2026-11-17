# Evaluation data

Nothing here is redistributed beyond the manifests under `bench/data/`. Every
set is downloaded from its publisher under its own license, prepared as
below, and read by the scripts named. Audio for the model is 16 kHz mono; the
scripts resample and downmix whatever they read.

| set | scripts | license | what to download |
|---|---|---|---|
| TEN-VAD test set | `tenvad.py` | Apache-2.0 with conditions (TEN) | the `testset/` directory of https://github.com/TEN-framework/ten-vad |
| VoxConverse test | `voxconverse.py` | CC BY 4.0 | `huggingface-cli download diarizers-community/voxconverse --repo-type dataset` |
| AISHELL-4 test | `build_vadsuite.py`, `vadsuite.py` | CC BY-SA 4.0 | `test` from https://openslr.org/111 |
| AliMeeting test, array and headsets | `build_vadsuite.py`, `vadsuite.py` | CC BY-SA 4.0 | `Test_Ali` from https://openslr.org/119 |
| AVA-Speech | `build_vadsuite.py`, `vadsuite.py` | CC BY 4.0 | `ava_speech_labels_v1.csv` and `ava_file_names_trainval_v2.1.txt` from https://research.google.com/ava/download.html; the builder fetches the audio |
| Earnings-21 | `build_vadsuite.py`, `vadsuite.py` | CC BY-SA 4.0 | `media` and `rttms` of https://github.com/revdotcom/speech-datasets |
| LibriParty eval | `build_vadsuite.py`, `vadsuite.py` | CC BY 4.0 | the eval sessions of https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriParty |
| MSDWild | `build_vadsuite.py`, `vadsuite.py` | see the dataset's terms | `raw_wav` and `all.rttm` of https://github.com/X-LANCE/MSDWILD |
| ESC-50 | `build_vadsuite.py`, `vadsuite.py` | CC BY-NC 3.0 | https://github.com/karolpiczak/ESC-50 |
| AMI corpus, headset audio | `ami.py` | CC BY 4.0 | https://groups.inf.ed.ac.uk/ami/download/ |
| pipecat smart-turn test | `pipecat.py` | CC BY 4.0 | `huggingface-cli download pipecat-ai/smart-turn-data-v3.2-test --repo-type dataset` |
| Easy Turn test set | `turnsets.py --format easyturn` | Apache-2.0 | `huggingface-cli download 1xg/Easy-Turn-Testset --repo-type dataset` |
| FastTurn test set | `turnsets.py --format fastturn` | Apache-2.0 | `huggingface-cli download ASLP-lab/FastTurn-Testset --repo-type dataset` |
| eot-bench | `eotbench.py` | Apache-2.0 (harness), CC BY 4.0 (data) | https://github.com/livekit/eot-bench |

## Baselines

- Silero VAD: the `silero-vad` pip package (MIT), version 6.2.1 in the
  published results. Run on its native 512-sample chunks with its states
  reset per recording.
- smart-turn v3.2: pipecat's released ONNX files (BSD-2-Clause),
  `smart-turn-v3.2-gpu.onnx` (fp32) in the published results, run as its
  own `inference.py` runs it. Download from https://huggingface.co/pipecat-ai/smart-turn-v3.2.

## The manifests

AISHELL-4, AliMeeting, AVA-Speech, Earnings-21, LibriParty, MSDWild and ESC-50
are built from their downloads by `bench/scripts/build_vadsuite.py`, which
writes each set's 16 kHz mono audio and `manifest.jsonl` to
`data/vadsuite/<set>/`; its docstring lists the folder layout each download
needs and how each set is prepared.

`bench/data/ami_turn_ends.jsonl` holds one row per headset channel (`meeting`,
`participant`, `headset`, `split`) with its real turn ends on the meeting
timeline: `t`, the end of the owner's last word; `floor_free`, the seconds
until another speaker is audible in the channel; `timed`, whether the words
around it carry timestamps (the published numbers use timed ends only).
Audio is `<root>/<meeting>/audio/<meeting>.Headset-<headset>.wav` from the AMI
download.

`bench/data/voxconverse_curated.jsonl` holds, per test recording, the spans
that are scored and the speech spans inside them at word level. It exists
because diarization labels are not voice-activity labels: they mark a
speaker's turn as speech from its first word to its last, pauses included,
and they carry the collar of the annotation convention at every boundary. On
the scored spans the diarization labels call 89.5% of frames speech and the
curated ones 77.7%; the two agree on 88% of frames, and a voice-activity
detector is held to the frames in between by the diarization reading alone.
