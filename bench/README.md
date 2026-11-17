# Benchmarks

All results use public datasets and public baselines, one scorer per task, and
treat the model as a black box. Every table below is rendered by
`bench/scripts/tables.py` from the score files in `bench/results/`; nothing in
it is typed by hand. The data comes from the publishers as [DATA.md](DATA.md)
describes; the baselines are Silero VAD 6.2.1 and pipecat smart-turn v3.2
(fp32), installed from their releases.

## How to read it

**Voice activity.** A frame is the model's 40 ms frame; it is speech when a
labeled segment covers its center. Each system's probability for a frame is
the one it emitted for that frame: SAVA's probability for the frame, computed
with 120 ms of audio after it (emitted 120 ms later), Silero's probability of
the 32 ms chunk holding the frame's center. Scores are pooled over all frames
of a set; the confidence interval is a paired bootstrap over recordings (2,000
resamples) of the AUC difference, and a star marks an interval that excludes
zero. F1 is at threshold 0.5 for both. Because SAVA's probability uses 120 ms
of audio past the frame and Silero's does not, the per-frame comparison is not
latency-matched; it is the comparison a pipeline sees when it reads each
system at its own output time.

**ESC-50** holds no speech: the scores are the share of clips on which a
system never stays above 0.5 for 100 ms, the share on which it never goes
above 0.5 at all, and the share of frames above 0.5.

**VoxConverse, curated.** The dataset's diarization labels mark a speaker's
turn as speech from first word to last, pauses included, with a collar at
every boundary. The curated reading scores the spans listed in
`bench/data/voxconverse_curated.jsonl` against word-level speech spans; on
those spans the diarization labels call 89.5% of frames speech, the curated
ones 77.7%, and the two readings agree on 88% of frames. The diarization row
also counts the pauses inside a speaker's turn as speech, so a detector that
marks those pauses as silence loses score there.

**End of turn, clip sets.** On the pipecat test set, Easy Turn and FastTurn
each clip carries one label; the model's score is its end-of-turn
probability at the clip's last frame, smart-turn's its own output on the
clip. Accuracy, precision, recall and F1 take "complete" as positive at
threshold 0.5, as smart-turn's own training script reports them. Easy Turn
and FastTurn add two labels a binary end-of-turn model has no output for,
backchannel and wait; their rows report the share of each state called
ended. The anchored reading cuts every clip at its energy speech end and
appends 200 ms of −70 dBFS noise, so every clip ends the same way after its
speech and the amount of trailing silence in each set's clips does not affect
the score.

**eot-bench** is LiveKit's harness run unmodified with the adapter in
`bench/scripts/eotbench.py`: at every 100 ms grid point of every pause the
model's end-of-turn probability is read from the last frame before the
point, and the harness's own metrics follow. smart-turn runs through the
harness's own `SmartTurnAudioAdapter`. Both models are reported twice: on
the full score trace, where the harness's policy sweep may fire at any
point of a pause, and at the score read 0.2 s into the pause, the harness's
default for smart-turn and how that model is typically run.

**AMI** is policy-free: at each real turn end T in a headset channel (the
owner's last word, the floor passing to another speaker), does the model's
end-of-turn probability cross 0.5 by T + d, and how soon. A turn end counts
at delay d only when nobody else is audible in the channel for d after T.
There are no negatives, so nothing is tuned on it. SAVA's probability is
its per-frame curve; smart-turn is called the way a pipeline would call it,
on the last 8 s of the channel at the end of every 40 ms frame after T.

## Results

<!-- tables start -->
### Voice activity, frame AUC

| set | hours | SAVA AUC | Silero AUC | SAVA − Silero, 95% CI | SAVA F1 | Silero F1 |
|---|---|---|---|---|---|---|
| TEN-VAD test | 0.1 | 0.9769 | 0.9541 | +0.0228* [+0.0092, +0.0362] | 0.956 | 0.938 |
| VoxConverse test, diarization labels | 43.5 | 0.9555 | 0.9520 | +0.0035* [+0.0013, +0.0057] | 0.939 | 0.952 |
| VoxConverse test, curated | 43.5 | 0.9807 | 0.9311 | +0.0495* [+0.0458, +0.0533] | 0.964 | 0.929 |
| AISHELL-4 test | 12.7 | 0.9327 | 0.9350 | -0.0023 [-0.0110, +0.0083] | 0.883 | 0.830 |
| AliMeeting test | 43.1 | 0.9376 | 0.9635 | -0.0259* [-0.0405, -0.0092] | 0.848 | 0.892 |
| AVA-Speech | 37.8 | 0.9188 | 0.9041 | +0.0147* [+0.0048, +0.0243] | 0.820 | 0.808 |
| Earnings-21 | 39.3 | 0.9722 | 0.9656 | +0.0065 [-0.0003, +0.0113] | 0.936 | 0.957 |
| LibriParty eval | 4.0 | 0.9670 | 0.9572 | +0.0098 [-0.0024, +0.0241] | 0.904 | 0.846 |
| MSDWild | 80.0 | 0.9329 | 0.9148 | +0.0180* [+0.0132, +0.0229] | 0.942 | 0.946 |

### Noise

| ESC-50 (no speech) | clips without a false trigger | clips without a frame above 0.5 | frames above 0.5 |
|---|---|---|---|
| SAVA | 97.3% | 93.2% | 0.32% |
| Silero VAD 6.2.1 | 93.2% | 87.1% | 0.57% |

### End of turn, pipecat

| pipecat smart-turn test | accuracy at 0.5 | real | synthetic | AUC | best accuracy (threshold) |
|---|---|---|---|---|---|
| SAVA | 0.9335 | 0.9586 | 0.9278 | 0.9795 | 0.9351 (0.45) |
| smart-turn v3.2 | 0.9371 | 0.9480 | 0.9346 | 0.9825 | 0.9381 (0.36) |

### End of turn, four states

**Easy Turn** (800 clips), share called ended at 0.5 and complete vs incomplete:

| system | complete | incomplete | backchannel | wait | accuracy | miss | false alarm | AUC |
|---|---|---|---|---|---|---|---|---|
| SAVA | 50.0% | 54.0% | 35.0% | 32.0% | 48.0% | 50.0% | 54.0% | 0.5119 |
| smart-turn v3.2 | 87.0% | 32.7% | 71.0% | 90.0% | 77.2% | 13.0% | 32.7% | 0.8266 |

**Easy Turn, anchored +200 ms** (800 clips), share called ended at 0.5 and complete vs incomplete:

| system | complete | incomplete | backchannel | wait | accuracy | miss | false alarm | AUC |
|---|---|---|---|---|---|---|---|---|
| SAVA | 99.0% | 92.3% | 93.0% | 91.0% | 53.3% | 1.0% | 92.3% | 0.7923 |
| smart-turn v3.2 | 88.3% | 34.0% | 73.0% | 85.0% | 77.2% | 11.7% | 34.0% | 0.8396 |

**FastTurn** (22432 clips), share called ended at 0.5 and complete vs incomplete:

| system | complete | incomplete | backchannel | wait | accuracy | miss | false alarm | AUC |
|---|---|---|---|---|---|---|---|---|
| SAVA | 61.9% | 39.2% | 9.6% | 38.6% | 61.7% | 38.1% | 39.2% | 0.6480 |
| smart-turn v3.2 | 50.4% | 33.8% | 50.9% | 90.6% | 53.5% | 49.6% | 33.8% | 0.6211 |

**FastTurn, anchored +200 ms** (22432 clips), share called ended at 0.5 and complete vs incomplete:

| system | complete | incomplete | backchannel | wait | accuracy | miss | false alarm | AUC |
|---|---|---|---|---|---|---|---|---|
| SAVA | 88.9% | 75.8% | 44.0% | 87.3% | 76.1% | 11.1% | 75.8% | 0.6501 |
| smart-turn v3.2 | 54.7% | 35.8% | 45.6% | 90.4% | 56.5% | 45.3% | 35.8% | 0.6329 |


### End of turn, eot-bench

| eot-bench, 14 languages pooled | cut@300 ms | cut@600 ms | latency@5% (ms) | latency@10% (ms) | policy 0.5: cut | policy 0.5: latency (ms) | AUC |
|---|---|---|---|---|---|---|---|
| SAVA, full trace | 38.2% | 14.1% | 936 | 709 | 38.4% | 475 | 0.8157 |
| smart-turn v3.2, full trace | 36.4% | 13.4% | 929 | 693 | 33.9% | 460 | 0.8616 |
| SAVA, 0.2 s point | 39.1% | 14.2% | 927 | 703 | 28.7% | 750 | 0.8109 |
| smart-turn v3.2, 0.2 s point | 40.8% | 14.2% | 935 | 706 | 27.5% | 820 | 0.7834 |

Per language, SAVA / smart-turn v3.2 (full trace):

| language | cut@300 ms | latency@5% (ms) | AUC |
|---|---|---|---|
| ar | 45.7% / 36.4% | 840 / 812 | 0.7677 / 0.8550 |
| de | 31.5% / 31.4% | 937 / 947 | 0.8674 / 0.8708 |
| en | 37.0% / 32.8% | 1121 / 1045 | 0.8520 / 0.8951 |
| es | 37.2% / 37.1% | 944 / 974 | 0.8357 / 0.8702 |
| fr | 31.7% / 31.4% | 812 / 832 | 0.8599 / 0.8987 |
| hi | 37.0% / 36.2% | 605 / 630 | 0.8195 / 0.8480 |
| id | 41.5% / 39.5% | 934 / 928 | 0.7981 / 0.8540 |
| it | 36.3% / 39.0% | 1010 / 964 | 0.7980 / 0.8366 |
| ja | 38.8% / 32.9% | 819 / 777 | 0.8507 / 0.8671 |
| ko | 22.8% / 26.2% | 711 / 730 | 0.8691 / 0.8869 |
| nl | 28.4% / 38.3% | 861 / 912 | 0.8632 / 0.8368 |
| pt | 37.1% / 37.2% | 858 / 819 | 0.8088 / 0.8767 |
| tr | 39.7% / 42.1% | 983 / 1052 | 0.7814 / 0.8449 |
| zh | 49.7% / 40.1% | 1003 / 891 | 0.7399 / 0.8522 |

Per language, SAVA / smart-turn v3.2 (0.2 s point):

| language | cut@300 ms | latency@5% (ms) | AUC |
|---|---|---|---|
| ar | 44.0% / 39.2% | 835 / 814 | 0.7910 / 0.7603 |
| de | 33.6% / 35.5% | 942 / 963 | 0.8579 / 0.8095 |
| en | 37.9% / 35.2% | 1160 / 1051 | 0.8336 / 0.8446 |
| es | 35.2% / 38.7% | 923 / 977 | 0.8462 / 0.7763 |
| fr | 30.8% / 38.9% | 818 / 873 | 0.8637 / 0.8145 |
| hi | 36.2% / 42.6% | 615 / 639 | 0.8231 / 0.7189 |
| id | 45.4% / 46.2% | 924 / 945 | 0.7878 / 0.7631 |
| it | 34.9% / 44.1% | 1005 / 980 | 0.8051 / 0.7634 |
| ja | 42.9% / 34.7% | 839 / 837 | 0.8348 / 0.8120 |
| ko | 23.0% / 30.1% | 710 / 734 | 0.8761 / 0.8420 |
| nl | 28.4% / 42.3% | 838 / 884 | 0.8644 / 0.7859 |
| pt | 36.2% / 42.7% | 816 / 836 | 0.8055 / 0.7832 |
| tr | 38.8% / 42.7% | 983 / 1013 | 0.7916 / 0.7378 |
| zh | 47.7% / 42.9% | 974 / 959 | 0.7464 / 0.7454 |

### End of turn, AMI

| AMI, 5043 real turn ends | fired by +200 ms (n=4956) | fired by +400 ms (n=4766) | fired by +600 ms (n=4571) | fired by +1000 ms (n=4040) | latency p50 / p90 |
|---|---|---|---|---|---|
| SAVA, threshold 0.5 | 85.5% | 91.2% | 92.7% | 93.5% | 71 / 192 ms |
| smart-turn v3.2, threshold 0.5 | 30.8% | 41.8% | 50.5% | 59.6% | 210 / 930 ms |

<!-- tables end -->

## Reproducing the numbers

`--device cuda` runs the model on a GPU; the baselines run on the CPU.

```bash
python -m pip install -e ".[eval]"
export BENCH_RESULTS=staging/results        # anything outside bench/results
python bench/scripts/tenvad.py --root <ten-vad>/testset
python bench/scripts/voxconverse.py --root <voxconverse download>
for set in aishell4 alimeeting ava earnings21 libriparty msdwild esc50; do
  python bench/scripts/build_vadsuite.py --set "$set" --downloads "<$set download>"
  python bench/scripts/vadsuite.py --set "$set" --root "data/vadsuite/$set"
done
python bench/scripts/ami.py --root <amicorpus> --smart_turn smart-turn-v3.2-gpu.onnx
python bench/scripts/pipecat.py --root <smart-turn test download> --smart_turn smart-turn-v3.2-gpu.onnx
python bench/scripts/turnsets.py --root <Easy-Turn-Testset>/testset --format easyturn --smart_turn smart-turn-v3.2-gpu.onnx
python bench/scripts/turnsets.py --root <Easy-Turn-Testset>/testset --format easyturn --anchor_ms 200 --smart_turn smart-turn-v3.2-gpu.onnx
python bench/scripts/turnsets.py --root <FastTurn-Testset> --format fastturn --smart_turn smart-turn-v3.2-gpu.onnx
python bench/scripts/turnsets.py --root <FastTurn-Testset> --format fastturn --anchor_ms 200 --smart_turn smart-turn-v3.2-gpu.onnx
python bench/scripts/tables.py --results staging/results
```

eot-bench runs in LiveKit's harness:

1. Check out https://github.com/livekit/eot-bench, put this repository's
   `src` and `bench/scripts` on `PYTHONPATH`, and set `SAVA_MODEL`.
2. SAVA: run `eot-harness predict-streaming --adapter eotbench:SavaAdapter`
   per language, then `compute-metrics` once as is (the full trace) and once
   with `--score-point 0.2`.
3. smart-turn: run `eot-harness predict --adapter
   eot_harness.smart_turn_adapter:SmartTurnAudioAdapter`, then
   `compute-metrics` once as is (the 0.2 s point its manifest asks for) and
   once on a copy of `predictions.parquet` without the manifest (the full
   trace).
4. Run `bench/scripts/eotsummary.py --output_dir <harness output dir>` to
   collect the result file from the harness's metrics.
