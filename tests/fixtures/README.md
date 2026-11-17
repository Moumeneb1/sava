# Golden parity fixture

`speech.wav` is 23 s of speech on a street: a speech clip provided by Inworld,
mixed into a street field recording about 10 dB under the voice, with the
street alone before and after it. It is 16 kHz mono PCM16.

The street is seconds 22 to 45 of "traffic and birds" (Corso Roma, Serra de'
Conti, Italy), recorded by nicola casetta for radio aporee and marked as public
domain: <https://archive.org/details/aporee_40228_45942>.

`speech_expected.npz` holds the `speech`, `lookahead` and `turn` logits the
training reference implementation computes for that clip in CPU fp32.
`tests/inworld/sava/test_golden.py` checks the released weights against them
to 2e-4, on the whole clip and through streams of several chunk sizes.
