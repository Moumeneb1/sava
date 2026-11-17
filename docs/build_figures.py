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
"""Draws the README's figures as SVG, in a light and a dark variant each.

  docs/assets/vad-{light,dark}.svg           voice-activity frame AUC per
                                             dataset, SAVA beside Silero VAD
  docs/assets/turn-{light,dark}.svg          end-of-turn clip AUC per dataset,
                                             SAVA beside smart-turn
  docs/assets/architecture-{light,dark}.svg  the network, from the waveform to
                                             its two predictions

    python docs/build_figures.py [--results bench/results] [--out docs/assets]
                                 [--architecture-only]

The charts read the published JSONs under bench/results: numbers are
generated, never typed. The architecture has no results dependency. Every
figure is hand-written SVG with live text, so nothing here needs a plotting
library. The README shows each pair through a <picture> element, so GitHub
picks the variant that matches the reader's theme.
"""

import argparse
import html
import json
import math
import pathlib

FONT = "Instrument Sans, Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "Geist Mono, SFMono-Regular, Menlo, Consolas, monospace"
# GitHub's light and dark canvases; the text and bar colours are the tokens of
# inworld.ai: its ink scale, its green accent, its gray scale.
PALETTES = {
    "light": {
        "surface": "#FFFFFF",
        "text": "#151613",
        "body": "#57534B",
        "muted": "#69665E",
        "rule": "#D7D2CC",
        "ours": "#02793D",
        "others": "#9A9089",
    },
    "dark": {
        "surface": "#0D1117",
        "text": "#EEEAE0",
        "body": "#B4AFA3",
        "muted": "#89857C",
        "rule": "#30363D",
        "ours": "#1A9B58",
        "others": "#9A9089",
    },
}
VAD_SETS = (
    ("tenvad", "TEN-VAD"),
    ("voxconverse", "VoxConverse, diarization"),
    ("voxconverse_curated", "VoxConverse, curated"),
    ("vadsuite_aishell4", "AISHELL-4"),
    ("vadsuite_alimeeting", "AliMeeting"),
    ("vadsuite_ava", "AVA-Speech"),
    ("vadsuite_earnings21", "Earnings-21"),
    ("vadsuite_libriparty", "LibriParty"),
    ("vadsuite_msdwild", "MSDWild"),
)
TURN_SETS = (
    ("pipecat", "pipecat smart-turn test"),
    ("easyturn", "Easy Turn"),
    ("easyturn_anchor200", "Easy Turn, anchored +200 ms"),
    ("fastturn", "FastTurn"),
    ("fastturn_anchor200", "FastTurn, anchored +200 ms"),
)
CHARTS = {  # caption, baseline's name, note
    "vad": ("Voice activity: frame AUC, higher is better.", "Silero VAD 6.2.1",
            "Each system at its own output time; SAVA speech has 120 ms"
            " lookahead."),
    "turn": ("End of turn: clip AUC, higher is better.", "smart-turn v3.2",
             "Complete vs incomplete clips; anchored rows use a fixed"
             " speech-end offset."),
}
# Where each chart's bars start, its ticks and their format. The voice-activity
# AUCs all sit between 0.9 and 1, where bars drawn from zero look alike.
AXES = {
    "vad": (0.9, (0.9, 0.92, 0.94, 0.96, 0.98, 1.0), "{:.2f}"),
    "turn": (0.0, (0.0, 0.25, 0.5, 0.75, 1.0), "{:g}"),
}
CHART_SOURCE = ("Full metrics, confidence intervals and caveats:"
                " bench/README.md.")
CHART_WIDTH = 1040
BAR = 20  # px; a dataset's two bars sit 4 px apart, the slot's rest is air
SLOT = 62
ARCH_DESC = (
    "SAVA takes 16 kHz mono audio in 40 ms frames of 640 samples. A causal "
    "log-mel front end uses a 400-sample window and 160-sample hop: four "
    "hops of 128 mel bins per frame. The frame's log RMS is appended to "
    "these 512 features. The current and previous feature rows are "
    "concatenated and projected from 1026 to 512 channels, then "
    "RMS-normalized. "
    "The shared trunk has three Mamba-2 blocks followed by one causal "
    "sliding-window attention block and a final RMSNorm. Each block has a "
    "pre-normalized mixer and SwiGLU with residual connections. Mamba-2 "
    "uses state size 128, head dimension 64, expansion 2 and convolution "
    "width 4. Attention uses 8 query heads, 2 key/value heads, head dimension "
    "64, rotary positions, normalized queries and keys, exclusive self "
    "attention and a 512-frame window. The speech branch projects to 512 "
    "channels, applies causal attention and RMSNorm, then has two linear "
    "sigmoid readouts. At emission step t, the released speech output q_t "
    "judges frame t-3: a fixed three-frame, 120 ms lookahead. The on-time "
    "readout p_t judges frame t. The turn branch receives the shared trunk "
    "features concatenated with a learned 64-dimensional embedding of "
    "p_(t-1), the previous frame's on-time speech probability. It projects "
    "576 channels to 512, applies causal attention and RMSNorm, and has "
    "linear sigmoid readouts for horizons 0, 5, 10 and 20 frames. The "
    "displayed end-of-turn probability r_t uses horizon 0 and is on time; "
    "it does not wait for the delayed speech verdict. The model has "
    "25.5 million parameters and constant state size per stream.")

# The figure's words, in a small markup: *x* italic, **x** bold, ^{x} raised,
# _{x} lowered.
ARCH_INPUT = ("*x*_{*t*}", "16 kHz")
ARCH_STAGES = (  # title, parameter lines, role
    ("Causal log-mel", ("*N* = 400, *H* = 160", "4 × 128 mel bins",
                        "per 40 ms frame"), "dsp"),
    ("Frame mixer", ("2 × (512 + 1) → 512", "current + previous", "RMSNorm"),
     "net"),
    ("Shared trunk", ("3 × Mamba-2", "1 × causal attention", "final RMSNorm"),
     "net"),
)
ARCH_LINKS = ("+ log RMS", "*D* = 512", "*h*_{*t*}")  # over each stage's arrow
ARCH_BRANCHES = (  # title, parameter lines, (output, its timing) each
    ("Speech branch", ("512 → 512", "attention, RMSNorm"),
     (("Speech output", "*q*_{*t*} · frame *t* − 3 · 120 ms later"),
      ("On-time speech", "*p*_{*t*} · frame *t*"))),
    ("Turn branch", ("576 → 512", "attention, RMSNorm"),
     (("End of turn", "*r*_{*t*} · on time · horizon 0 of 4"),)),
)
ARCH_EMBEDDING = ("*e*_{*t*−1} = *E*(*p*_{*t*−1})", "one-frame delay · 64 dims")
ARCH_TIMING = (
    "Timing at emission step *t*",
    "*q*_{*t*}: speech at frame *t* − 3, a fixed 120 ms lookahead.",
    "*r*_{*t*}: turn ended by frame *t*, with no lookahead delay.",
    "Dashed: the previous on-time speech conditions the turn.",
)
ARCH_BLOCK_TYPES = (  # tag, kind, (sub-layer, its role) twice, settings
    ("M", "Mamba-2", (("Mamba-2", "pre-norm mixer"), ("SwiGLU",
                                                      "pre-norm feed-forward")),
     "state 128 · head dimension 64 · expand 2 · conv 4"),
    ("A", "attention", (("causal attention", "pre-norm mixer"),
                        ("SwiGLU", "pre-norm feed-forward")),
     "8 query / 2 KV heads · RoPE · 512-frame window"),
)
ARCH_NOTES = (("Size", "25.5M parameters, 40 ms frames, constant state size"
               " per stream."),)
ARCH_WIDTH = CHART_WIDTH  # so every figure displays at one scale
ARCH_MARGIN = 24
# GitHub's canvases. Fills say a stage's role: fixed signal processing (dsp),
# learned layers (net), the heads. Body text keeps at least 7.5:1 contrast on
# every fill, captions at least 5.5:1.
ARCH_PALETTES = {
    "light": {
        "bg": "#FFFFFF",
        "ink": "#1F2328",
        "ink2": "#3D444D",
        "ink3": "#59636E",
        "line": "#59636E",
        "panel": "#F6F8FA",
        "dsp": "#DCE8F6",
        "net": "#D7EEDF",
        "head": "#F8E5C2",
    },
    "dark": {
        "bg": "#0D1117",
        "ink": "#F0F6FC",
        "ink2": "#D1D7E0",
        "ink3": "#A6AFBA",
        "line": "#A6AFBA",
        "panel": "#161B22",
        "dsp": "#1C3554",
        "net": "#17442F",
        "head": "#4A3511",
    },
}
# Advance widths in 1/1000 em of " " to "~", regular and bold: for each glyph
# the wider of Helvetica and Helvetica Neue (Arial shares Helvetica's
# metrics), so a box sized from them holds its text in whichever font of the
# stack the reader has.
_ADVANCE = {
    False:
        (278, 278, 426, 556, 556, 1000, 667, 278, 333, 333, 389, 600, 278, 389,
         278, 333, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278,
         600, 600, 600, 556, 1015, 667, 685, 722, 722, 667, 611, 778, 722, 278,
         519, 667, 556, 871, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944,
         667, 667, 611, 278, 333, 278, 600, 556, 222, 556, 593, 537, 593, 556,
         296, 574, 556, 222, 222, 519, 222, 853, 556, 574, 593, 593, 333, 500,
         315, 556, 500, 759, 518, 500, 500, 334, 260, 334, 600),
    True: (278, 333, 481, 556, 556, 1000, 722, 278, 333, 333, 407, 600, 278,
           407, 278, 389, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333,
           333, 600, 600, 600, 611, 975, 722, 722, 741, 741, 667, 611, 778, 741,
           296, 556, 722, 611, 907, 741, 778, 667, 778, 722, 667, 611, 741, 667,
           944, 667, 667, 648, 333, 389, 333, 600, 556, 278, 574, 611, 574, 611,
           574, 352, 611, 611, 278, 278, 574, 278, 907, 611, 611, 611, 611, 389,
           556, 370, 611, 556, 815, 556, 556, 519, 389, 280, 389, 600),
}
_ADVANCE_SYMBOLS = {"×": 600, "−": 600, "·": 278, "→": 1000}
_SCRIPT = 0.72  # a raised or lowered run's size relative to its text
_LINE_PITCH = 21


def _text(x, y, label, size, fill, weight="normal", anchor="start", mono=False):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{MONO if mono else FONT}"'
        f' font-size="{size}" font-weight="{weight}" fill="{fill}"'
        f' text-anchor="{anchor}">{html.escape(str(label))}</text>')


def benchmark_rows(results: pathlib.Path, task: str) -> list:
    """Reads paired AUCs in dataset order, refusing missing evidence."""
    sets = VAD_SETS if task == "vad" else TURN_SETS
    baseline = "silero" if task == "vad" else "smart_turn"
    rows = []
    for name, label in sets:
        with (results / f"{name}.json").open(encoding="utf-8") as source:
            systems = json.load(source)["systems"]
        values = []
        for system in ("sava", baseline):
            value = systems[system]["auc"]
            if isinstance(value, dict):
                value = value["all"]
            if not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError(f"Invalid AUC in {name}: {system}")
            values.append(value)
        rows.append((label, *values))
    return rows


def benchmark_svg(rows: list, task: str, theme: str) -> str:
    """Draws each dataset's two bars on the chart's axis, from its floor to
    one, refusing a value below the floor rather than cutting it off."""
    palette = PALETTES[theme]
    caption, baseline, note = CHARTS[task]
    floor, ticks, tick_format = AXES[task]
    if any(value < floor for _, *values in rows for value in values):
        raise ValueError(f"A {task} AUC is below the chart's floor of {floor}")
    left, right, top = 300.0, 84.0, 74.0
    plot = CHART_WIDTH - left - right
    scale = plot / (1.0 - floor)
    bottom = top + SLOT * len(rows)
    height = bottom + 84
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_WIDTH}"'
        f' height="{height:.0f}" viewBox="0 0 {CHART_WIDTH} {height:.0f}"'
        f' role="img" aria-label="{html.escape(caption, quote=True)}">',
        f'<rect width="100%" height="100%" fill="{palette["surface"]}"/>',
        _text(28, 36, caption, 16, palette["body"]),
    ]
    # two bars per dataset, so the legend carries the identity
    x = CHART_WIDTH - 28.0
    for color, label in (("others", baseline), ("ours", "Inworld SAVA")):
        x -= _text_width(label, 14)
        out.append(_text(x, 36, label, 14, palette["text"]))
        x -= 22
        out.append(f'<rect x="{x:.1f}" y="24" width="14" height="14" rx="2"'
                   f' fill="{palette[color]}"/>')
        x -= 24
    for tick in ticks:
        x = left + scale * (tick - floor)
        out.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}"'
                   f' y2="{bottom}" stroke="{palette["rule"]}"'
                   ' stroke-width="1"/>')
        out.append(
            _text(x,
                  bottom + 18,
                  tick_format.format(tick),
                  13.5,
                  palette["muted"],
                  anchor="middle",
                  mono=True))
    for index, (label, ours, other) in enumerate(rows):
        y = top + SLOT * index + (SLOT - 2 * BAR - 4) / 2
        out.append(
            _text(left - 14,
                  y + BAR + 8,
                  label,
                  16,
                  palette["text"],
                  anchor="end"))
        for value, color in ((ours, "ours"), (other, "others")):
            width = max(scale * (value - floor), 4.0)
            # square at the baseline, 4 px rounded at the data end
            out.append(
                f'<path d="M{left:.1f},{y:.1f} h{width - 4:.1f} a4,4 0 0 1 4,4'
                f' v{BAR - 8} a4,4 0 0 1 -4,4 h-{width - 4:.1f} z"'
                f' fill="{palette[color]}"/>')
            out.append(
                _text(left + width + 8,
                      y + BAR - 5,
                      f"{value:.4f}",
                      15,
                      palette["text"],
                      "600",
                      mono=True))
            y += BAR + 4
    out.append(_text(28, height - 40, note, 13.5, palette["muted"]))
    out.append(_text(28, height - 18, CHART_SOURCE, 13.5, palette["muted"]))
    return "\n".join(out + ["</svg>"])


def _runs(markup: str) -> list:
    """Splits figure markup into (text, italic, bold, shift) runs.

    The shift is 1 for a raised run, -1 for a lowered one and 0 otherwise.
    """
    runs, text, italic, bold, shift, i = [], "", False, False, 0, 0
    while i < len(markup):
        if markup.startswith("**", i):
            runs.append((text, italic, bold, shift))
            text, bold, i = "", not bold, i + 2
        elif markup[i] == "*":
            runs.append((text, italic, bold, shift))
            text, italic, i = "", not italic, i + 1
        elif markup.startswith(("^{", "_{"), i):
            runs.append((text, italic, bold, shift))
            text, shift, i = "", 1 if markup[i] == "^" else -1, i + 2
        elif markup[i] == "}" and shift:
            runs.append((text, italic, bold, shift))
            text, shift, i = "", 0, i + 1
        else:
            text, i = text + markup[i], i + 1
    runs.append((text, italic, bold, shift))
    return [run for run in runs if run[0]]


def _text_width(markup: str, size: float, bold: bool = False) -> float:
    """The width ``markup`` takes at ``size`` px."""
    total = 0.0
    for text, _, run_bold, shift in _runs(markup):
        advances = _ADVANCE[bold != run_bold]
        em = sum(
            advances[ord(ch) -
                     32] if " " <= ch <= "~" else _ADVANCE_SYMBOLS.get(ch, 600)
            for ch in text)
        total += em / 1000 * size * (_SCRIPT if shift else 1)
    return total


def _rich(x, y, markup, size, fill, bold=False, anchor="middle"):
    """A text element from markup: italic, bold and shifted runs as tspans."""
    spans, offset = [], 0.0
    for text, italic, run_bold, shift in _runs(markup):
        attrs = []
        target = {1: -0.4, -1: 0.22, 0: 0.0}[shift] * size
        if target != offset:
            attrs.append(f'dy="{target - offset:.2f}"')
            offset = target
        if shift:
            attrs.append(f'font-size="{size * _SCRIPT:.2f}"')
        if italic:
            attrs.append('font-style="italic"')
        if run_bold:  # **x** flips the element's weight
            attrs.append(f'font-weight="{400 if bold else 600}"')
        body = html.escape(text, quote=False)
        joined = " ".join(attrs)
        spans.append(f"<tspan {joined}>{body}</tspan>" if attrs else body)
    weight = ' font-weight="600"' if bold else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}"'
            f' font-size="{size}" fill="{fill}"{weight}'
            f' text-anchor="{anchor}">{"".join(spans)}</text>')


def _rounded(points, radius):
    """Path data through ``points``, the corners rounded by ``radius``."""
    d = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
    for (x0, y0), (x1, y1), (x2, y2) in zip(points, points[1:], points[2:]):
        before = math.hypot(x1 - x0, y1 - y0)
        after = math.hypot(x2 - x1, y2 - y1)
        r = min(radius, before / 2, after / 2)
        ax, ay = x1 - (x1 - x0) / before * r, y1 - (y1 - y0) / before * r
        bx, by = x1 + (x2 - x1) / after * r, y1 + (y2 - y1) / after * r
        d += f" L{ax:.1f},{ay:.1f} Q{x1:.1f},{y1:.1f} {bx:.1f},{by:.1f}"
    return d + f" L{points[-1][0]:.1f},{points[-1][1]:.1f}"


def _arrow(points, color, dash=None, radius=10.0):
    """A line through ``points`` whose last point is the tip of its head."""
    length, width = 8.5, 6.5
    (x1, y1), (x2, y2) = points[-2], points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    cos, sin = math.cos(angle), math.sin(angle)
    # the line stops inside the head, so its end never shows past the tip
    body = list(
        points[:-1]) + [(x2 - 0.62 * length * cos, y2 - 0.62 * length * sin)]
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    head = [(0, 0), (-length, width / 2), (-0.72 * length, 0),
            (-length, -width / 2)]
    tip = " L".join(f"{x2 + u * cos - v * sin:.1f},{y2 + u * sin + v * cos:.1f}"
                    for u, v in head)
    return (f'<path d="{_rounded(body, radius)}" fill="none" stroke="{color}"'
            f' stroke-width="1.4" stroke-linecap="round"'
            f' stroke-linejoin="round"{dashes}/>\n'
            f'<path d="M{tip} Z" fill="{color}"/>')


def _node(x, y, r, palette, concat=False):
    """A circle holding a plus (the sum) or two bars (the concatenation),
    drawn rather than typeset."""
    k = r * 0.52
    mark = (f"M{x - k / 2:.1f},{y - k:.1f} V{y + k:.1f}"
            f" M{x + k / 2:.1f},{y - k:.1f} V{y + k:.1f}" if concat else
            f"M{x - k:.1f},{y:.1f} H{x + k:.1f} M{x:.1f},{y - k:.1f}"
            f" V{y + k:.1f}")
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{palette["bg"]}"'
            f' stroke="{palette["line"]}" stroke-width="1.4"/>\n'
            f'<path d="{mark}" stroke="{palette["line"]}" stroke-width="1.4"'
            ' stroke-linecap="round"/>')


def _dot(x, y, r, palette):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}"'
            f' fill="{palette["line"]}"/>')


def _stage(x, top, w, h, title, lines, fill, palette):
    """A filled box with its title and parameters centred in it."""
    out = [
        f'<rect x="{x:.1f}" y="{top:.1f}" width="{w:.1f}" height="{h}" rx="8"'
        f' fill="{fill}"/>'
    ]
    y = top + h / 2 - _LINE_PITCH * len(lines) / 2 + 5
    out.append(_rich(x + w / 2, y, title, 17.5, palette["ink"], bold=True))
    for k, line in enumerate(lines):
        out.append(
            _rich(x + w / 2, y + 24 + _LINE_PITCH * k, line, 15,
                  palette["ink2"]))
    return out


def _cell(x, mid, w, name, detail, fill, palette):
    """A small filled box, 52 px high around ``mid``: a name and one detail."""
    return [
        f'<rect x="{x:.1f}" y="{mid - 26:.1f}" width="{w:.1f}" height="52"'
        f' rx="7" fill="{fill}"/>',
        _rich(x + w / 2, mid - 3, name, 15.5, palette["ink"], bold=True),
        _rich(x + w / 2, mid + 16, detail, 13.5, palette["ink2"]),
    ]


def _cell_width(cells) -> float:
    return max(
        max(_text_width(name, 15.5, bold=True), _text_width(detail, 13.5))
        for name, detail in cells) + 30


def _block_rows(x, y, palette):
    """The two block types as rows of residual sub-layers; (x, y) is the left
    edge and the first row's centre line."""
    out = []
    box_w = _cell_width(
        cell for _, _, sublayers, _ in ARCH_BLOCK_TYPES for cell in sublayers)
    for row, (tag, kind, sublayers, settings) in enumerate(ARCH_BLOCK_TYPES):
        mid = y + 78 * row
        out.append(_rich(x + 18, mid + 2, tag, 21, palette["ink"], bold=True))
        out.append(_rich(x + 18, mid + 20, kind, 12.5, palette["ink3"]))
        bx = x + 84
        out.append(_arrow([(x + 48, mid), (bx, mid)], palette["line"]))
        for k, (name, role) in enumerate(sublayers):
            out.extend(
                _cell(bx, mid, box_w, name, role, palette["net"], palette))
            px = bx + box_w + 28
            out.append(
                _arrow([(bx + box_w, mid), (px - 11, mid)], palette["line"]))
            # the residual: the sub-layer's input goes around it to the sum
            fork = bx - 18
            over = mid - 26 - 13
            out.append(_dot(fork, mid, 2.6, palette))
            out.append(
                _arrow([(fork, mid), (fork, over), (px, over), (px, mid - 11)],
                       palette["line"],
                       radius=7))
            out.append(_node(px, mid, 11, palette))
            nxt = px + 11 + (34 if k == 0 else 30)
            out.append(_arrow([(px + 11, mid), (nxt, mid)], palette["line"]))
            bx = nxt
        out.append(
            _rich(bx + 12,
                  mid + 5,
                  settings,
                  14.5,
                  palette["ink2"],
                  anchor="start"))
    return out


def architecture_svg(theme: str) -> str:
    """The network as a layer diagram: the pipeline to the speech outputs,
    the turn branch they condition, then the insides of the two block types."""
    p = ARCH_PALETTES[theme]
    left, right = ARCH_MARGIN, ARCH_WIDTH - ARCH_MARGIN
    top, h, stub = 66, 116, 54
    mid = top + h / 2
    (speech, speech_lines, speech_outputs), (turn, turn_lines,
                                             turn_outputs) = ARCH_BRANCHES
    widths = [
        max([_text_width(title, 17.5, bold=True)] +
            [_text_width(line, 15)
             for line in lines]) + 24
        for title, lines in [stage[:2] for stage in ARCH_STAGES] +
        [(speech, speech_lines), (turn, turn_lines)]
    ]
    branch_w = max(widths[3:])
    out_w = _cell_width(speech_outputs + turn_outputs)
    # input stub, three stages, the branches, their outputs; the stub sits
    # flush against its stage and the other gaps share what is left
    items = [stub] + widths[:3] + [branch_w, out_w]
    gap = (right - left - sum(items)) / 4
    xs, x = [], left
    for i, w in enumerate(items):
        xs.append(x)
        x += w + (0 if i == 0 else gap)
    out = []
    out.append(_arrow([(left, mid), (left + stub, mid)], p["line"]))
    out.append(_rich(left + stub / 2 - 4, mid - 10, ARCH_INPUT[0], 19,
                     p["ink"]))
    out.append(
        _rich(left + stub / 2 - 4, mid + 24, ARCH_INPUT[1], 13.5, p["ink3"]))
    for j, (title, lines, role) in enumerate(ARCH_STAGES, start=1):
        out.extend(_stage(xs[j], top, items[j], h, title, lines, p[role], p))
        x1, x2 = xs[j] + items[j], xs[j + 1]
        out.append(_arrow([(x1 + 1, mid), (x2 - 1, mid)], p["line"]))
        out.append(
            _rich((x1 + x2) / 2, top - 16, ARCH_LINKS[j - 1], 16, p["ink"]))
    # the speech branch and its two readouts
    bx, ox = xs[4], xs[5]
    out.extend(_stage(bx, top, branch_w, h, speech, speech_lines, p["head"], p))
    for k, (name, timing) in enumerate(speech_outputs):
        level = top + 26 + 64 * k
        out.append(
            _arrow([(bx + branch_w + 1, level), (ox - 1, level)], p["line"]))
        out.extend(_cell(ox, level, out_w, name, timing, p["head"], p))
    # the turn branch: the trunk's features beside the embedded previous
    # on-time speech probability
    turn_top, turn_h = top + h + 84, 76
    turn_mid = turn_top + turn_h / 2
    cx, trunk = bx - 34, xs[3] + items[3] / 2
    out.append(
        _arrow([(trunk, top + h), (trunk, turn_mid), (cx - 11, turn_mid)],
               p["line"]))
    out.append(
        _rich(trunk + 9,
              top + h + 26,
              ARCH_LINKS[2],
              16,
              p["ink"],
              anchor="start"))
    out.append(_node(cx, turn_mid, 11, p, concat=True))
    out.append(_rich(cx, turn_mid + 29, "concat", 12.5, p["ink3"]))
    out.append(_arrow([(cx + 11, turn_mid), (bx - 1, turn_mid)], p["line"]))
    out.extend(
        _stage(bx, turn_top, branch_w, turn_h, turn, turn_lines, p["head"], p))
    out.append(
        _arrow([(bx + branch_w + 1, turn_mid), (ox - 1, turn_mid)], p["line"]))
    out.extend(_cell(ox, turn_mid, out_w, *turn_outputs[0], p["head"], p))
    embed_w = _cell_width([ARCH_EMBEDDING])
    embed_x, embed_mid = cx - 22, top + h + 42
    feedback = ox + 28
    out.append(
        _arrow([(feedback, top + h), (feedback, embed_mid),
                (embed_x + embed_w + 1, embed_mid)],
               p["line"],
               dash="5 4"))
    out.extend(_cell(embed_x, embed_mid, embed_w, *ARCH_EMBEDDING, p["net"], p))
    out.append(
        _arrow([(cx, embed_mid + 26), (cx, turn_mid - 11)],
               p["line"],
               dash="5 4"))
    for k, line in enumerate(ARCH_TIMING):
        out.append(
            _rich(left,
                  top + h + 54 + 24 * k,
                  line,
                  15.5 if k == 0 else 14.5,
                  p["ink" if k == 0 else "ink2"],
                  bold=k == 0,
                  anchor="start"))
    # the block, opened
    panel = turn_top + turn_h + 40
    out.append(f'<rect x="{left}" y="{panel:.0f}" width="{right - left}"'
               f' height="214" rx="12" fill="{p["panel"]}"/>')
    heading = "Inside each trunk block"
    out.append(
        _rich(left + 22, panel + 32, heading, 15.5, p["ink"], True, "start"))
    out.append(
        _rich(left + 30 + _text_width(heading, 15.5, bold=True),
              panel + 32,
              "· two pre-norm residual steps",
              15.5,
              p["ink3"],
              anchor="start"))
    out.extend(_block_rows(left + 22, panel + 98, p))
    y = panel + 214 + 36
    for k, (lead, rest) in enumerate(ARCH_NOTES):
        out.append(
            _rich(left,
                  y + 23 * k,
                  f"**{lead}:** {rest}",
                  14.5,
                  p["ink2"],
                  anchor="start"))
    height = y + 23 * (len(ARCH_NOTES) - 1) + 22
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ARCH_WIDTH}"'
        f' height="{height:.0f}" viewBox="0 0 {ARCH_WIDTH} {height:.0f}"'
        ' role="img" aria-label="SAVA architecture">',
        f"<desc>{html.escape(ARCH_DESC)}</desc>",
        f'<rect width="100%" height="100%" fill="{p["bg"]}"/>',
    ] + out + ["</svg>"])


def main(argv=None):
    """Writes architecture figures and optionally the recorded comparisons."""
    parser = argparse.ArgumentParser(
        description="Draws the README's figures in both themes.")
    parser.add_argument("--results",
                        type=pathlib.Path,
                        default=pathlib.Path("bench/results"))
    parser.add_argument("--out",
                        type=pathlib.Path,
                        default=pathlib.Path("docs/assets"))
    parser.add_argument("--architecture-only", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {} if args.architecture_only else {
        task: benchmark_rows(args.results, task) for task in ("vad", "turn")
    }
    for theme in PALETTES:
        figures = {"architecture": architecture_svg(theme)}
        for task, values in rows.items():
            figures[task] = benchmark_svg(values, task, theme)
        for name, svg in figures.items():
            (args.out / f"{name}-{theme}.svg").write_text(svg + "\n",
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
