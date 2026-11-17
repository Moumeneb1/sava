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
"""The benchmark tables, rendered from ``bench/results/*.json`` as Markdown.
Every number quoted in the READMEs comes from here.

    python bench/scripts/tables.py [--results bench/results]
"""

import argparse
import json
import os

import common


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses this benchmark's command line without global flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=common.RESULTS, help="")
    return parser.parse_args(argv)


VAD_SETS = (
    ("tenvad", "TEN-VAD test"),
    ("voxconverse", "VoxConverse test, diarization labels"),
    ("voxconverse_curated", "VoxConverse test, curated"),
    ("vadsuite_aishell4", "AISHELL-4 test"),
    ("vadsuite_alimeeting", "AliMeeting test"),
    ("vadsuite_ava", "AVA-Speech"),
    ("vadsuite_earnings21", "Earnings-21"),
    ("vadsuite_libriparty", "LibriParty eval"),
    ("vadsuite_msdwild", "MSDWild"),
)
TURN_SETS = (("easyturn", "Easy Turn"), ("easyturn_anchor200",
                                         "Easy Turn, anchored +200 ms"),
             ("fastturn", "FastTurn"), ("fastturn_anchor200",
                                        "FastTurn, anchored +200 ms"))


def _load(name: str, results: str) -> dict | None:
    path = os.path.join(results, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


SYSTEM_TITLES = {
    "sava": "SAVA",
    "silero": "Silero VAD 6.2.1",
    "smart_turn": "smart-turn v3.2",
}


def system_title(system: str) -> str:
    return SYSTEM_TITLES.get(system, system)


def vad_table(results: str = common.RESULTS) -> list[str]:
    lines = [
        "| set | hours | SAVA AUC | Silero AUC | SAVA − Silero, 95% CI "
        "| SAVA F1 | Silero F1 |", "|---|---|---|---|---|---|---|"
    ]
    for name, title in VAD_SETS:
        result = _load(name, results)
        if result is None:
            continue
        sava = result["systems"]["sava"]
        silero = result["systems"].get("silero")
        if silero is None:
            lines.append(
                _row([
                    title, f"{result['hours']:.1f}", f"{sava['auc']:.4f}", "–",
                    "–", f"{sava['f1']:.3f}", "–"
                ]))
            continue
        ci = silero["sava_minus_this_auc"]
        mark = "*" if ci["low"] > 0 or ci["high"] < 0 else ""
        lines.append(
            _row([
                title, f"{result['hours']:.1f}", f"{sava['auc']:.4f}",
                f"{silero['auc']:.4f}",
                f"{sava['auc'] - silero['auc']:+.4f}{mark} "
                f"[{ci['low']:+.4f}, {ci['high']:+.4f}]", f"{sava['f1']:.3f}",
                f"{silero['f1']:.3f}"
            ]))
    return lines if len(lines) > 2 else []


def noise_table(results: str = common.RESULTS) -> list[str]:
    result = _load("vadsuite_esc50", results)
    if result is None:
        return []
    lines = [
        "| ESC-50 (no speech) | clips without a false trigger "
        "| clips without a frame above 0.5 | frames above 0.5 |",
        "|---|---|---|---|"
    ]
    for name, scores in result["systems"].items():
        lines.append(
            _row([
                system_title(name),
                f"{100 * scores['clips_without_a_false_trigger']:.1f}%",
                f"{100 * scores['clips_without_a_frame_above_threshold']:.1f}%",
                f"{100 * scores['frames_above_threshold']:.2f}%"
            ]))
    return lines


def pipecat_table(results: str = common.RESULTS) -> list[str]:
    result = _load("pipecat", results)
    if result is None:
        return []
    lines = [
        "| pipecat smart-turn test | accuracy at 0.5 | real | synthetic | AUC "
        "| best accuracy (threshold) |", "|---|---|---|---|---|---|"
    ]
    for name, scores in result["systems"].items():
        synthetic = scores["groups"]["synthetic"]
        lines.append(
            _row([
                system_title(name), f"{scores['overall']['accuracy']:.4f}",
                f"{synthetic['False']['accuracy']:.4f}",
                f"{synthetic['True']['accuracy']:.4f}", f"{scores['auc']:.4f}",
                f"{scores['sweep']['best_accuracy']:.4f} "
                f"({scores['sweep']['best_threshold']:.2f})"
            ]))
    return lines


def turnsets_table(results: str = common.RESULTS) -> list[str]:
    lines = []
    for name, title in TURN_SETS:
        result = _load(name, results)
        if result is None:
            continue
        lines += [
            f"**{title}** ({result['clips']} clips), share called ended at 0.5 "
            "and complete vs incomplete:", "",
            "| system | complete | incomplete | backchannel | wait "
            "| accuracy | miss "
            "| false alarm | AUC |", "|---|---|---|---|---|---|---|---|---|"
        ]
        for system, scores in result["systems"].items():
            shares = scores["share_ended"]["0.5"]
            binary = scores["complete_vs_incomplete"]["0.5"]["all"]
            lines.append(
                _row([system_title(system)] + [
                    f"{100 * shares[f'{s}/all']:.1f}%"
                    for s in ("complete", "incomplete", "backchannel", "wait")
                ] + [
                    f"{100 * binary['accuracy']:.1f}%",
                    f"{100 * binary['miss']:.1f}%",
                    f"{100 * binary['false_alarm']:.1f}%",
                    f"{scores['auc']['all']:.4f}"
                ]))
        lines.append("")
    return lines


def ami_table(results: str = common.RESULTS) -> list[str]:
    result = _load("ami", results)
    if result is None:
        return []
    counted = result["counted"]
    lines = [
        f"| AMI, {result['turn_ends']} real turn ends | " +
        " | ".join(f"fired by +{int(1000 * float(d))} ms (n={n})"
                   for d, n in counted.items()) + " | latency p50 / p90 |",
        "|---|" + "---|" * len(counted) + "---|",
    ]
    for name, scores in result["systems"].items():
        for threshold, s in scores["thresholds"].items():
            lines.append(
                _row([f"{system_title(name)}, threshold {threshold}"] + [
                    f"{100 * v:.1f}%" if v is not None else "–"
                    for v in s["fired_by"].values()
                ] + [
                    f"{s['latency_p50_ms']:.0f} / {s['latency_p90_ms']:.0f} ms"
                ]))
    return lines


EOT_SYSTEMS = (
    ("sava", "SAVA, full trace"),
    ("smart_turn", "smart-turn v3.2, full trace"),
    ("sava_at200ms", "SAVA, 0.2 s point"),
    ("smart_turn_at200ms", "smart-turn v3.2, 0.2 s point"),
)


def eot_cells(scores: dict) -> list[str]:
    head = scores["headline"]
    fixed = scores["policy_0.5"] or {}

    def pct(key):
        return f"{100 * head[key]['value']:.1f}%" if head[key] else "–"

    def ms(key):
        return f"{1000 * head[key]['value']:.0f}" if head[key] else "–"

    return [
        pct("cutoff_at_300ms"),
        pct("cutoff_at_600ms"),
        ms("latency_at_5pct"),
        ms("latency_at_10pct"),
        f"{100 * fixed['cutoff_rate']:.1f}%" if fixed else "–",
        f"{1000 * fixed['mean_latency']:.0f}" if fixed else "–",
        f"{scores['auc']:.4f}",
    ]


def eot_language_block(systems: dict, suffix: str, reading: str) -> list[str]:
    """Per-language rows of SAVA and smart-turn at one reading."""
    sava = systems.get("sava" + suffix, {}).get("languages", {})
    other = systems.get("smart_turn" + suffix, {}).get("languages", {})
    lines = [
        f"Per language, SAVA / smart-turn v3.2 ({reading}):",
        "",
        "| language | cut@300 ms | latency@5% (ms) | AUC |",
        "|---|---|---|---|",
    ]
    for language, scores in sava.items():
        cells = [eot_cells(scores)]
        if language in other:
            cells.append(eot_cells(other[language]))
        joined = [" / ".join(c[i] for c in cells) for i in (0, 2, 6)]
        lines.append(_row([language] + joined))
    return lines


def eot_table(results: str = common.RESULTS) -> list[str]:
    result = _load("eotbench", results)
    if result is None:
        return []
    systems = result["systems"]
    lines = [
        "| eot-bench, 14 languages pooled | cut@300 ms | cut@600 ms "
        "| latency@5% (ms) "
        "| latency@10% (ms) | policy 0.5: cut "
        "| policy 0.5: latency (ms) | AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, title in EOT_SYSTEMS:
        if key in systems and "pooled" in systems[key]:
            lines.append(_row([title] + eot_cells(systems[key]["pooled"])))
    for suffix, reading in (("", "full trace"), ("_at200ms", "0.2 s point")):
        lines += [""] + eot_language_block(systems, suffix, reading)
    return lines


def render(results: str = common.RESULTS) -> str:
    sections = [("Voice activity, frame AUC", vad_table(results)),
                ("Noise", noise_table(results)),
                ("End of turn, pipecat", pipecat_table(results)),
                ("End of turn, four states", turnsets_table(results)),
                ("End of turn, eot-bench", eot_table(results)),
                ("End of turn, AMI", ami_table(results))]
    out = []
    for title, lines in sections:
        if lines:
            out += [f"### {title}", ""] + lines + [""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(render(args.results))


if __name__ == "__main__":
    main()
