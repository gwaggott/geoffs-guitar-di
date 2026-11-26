"""Check the built instrument in sfizz: every sample referenced exists, and each way of
playing it picks the sample it should.

Usage:
  python3 tools/check.py

Renders short MIDI sequences through instrument/*.sfz with sfizz_render
(sfzrender.py) and names the sample that sounds at
each note-on: the best normalised correlation over 0.2 s against the candidate
samples at that key (picked medium, hammer-on, pull-off, palm mute medium and hard),
each pitched as its region plays it; and that a chord on the keyboard file with the
legato pedal up is the sum of its notes rendered alone (no note turned legato by
the others). Prints each case and exits 1 on any mismatch.
"""

import os
import re
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfzrender as R  # noqa: E402
from build import KEYSWITCH, LEGATO_CC, NAME, OUT, StringRegions, keyboard_strings  # noqa: E402
from guitar import STRINGS, load, tune  # noqa: E402

TMP = os.path.join(ROOT, "build", "check")


def note(t, n, vel, dur, ch=1):
    return [(t, (0x90 | (ch - 1), n, vel)), (t + dur, (0x80 | (ch - 1), n, 0))]


def cc(t, num, val, ch=1):
    return (t, (0xB0 | (ch - 1), num, val))


def samples_exist():
    missing = []
    for base, _, files in os.walk(OUT):
        for f in files:
            if f.endswith(".sfz"):
                text = open(os.path.join(base, f)).read()
                root = os.path.join(base, re.search(r"default_path=(\S+)", text).group(1))
                missing += [s for s in set(re.findall(r"sample=(\S+)", text)) if not os.path.exists(os.path.join(root, s))]
    return missing


def candidates(regions, string, key):
    """{take: the sample row and region volume the key plays for that take}."""
    r = regions[string]
    out = {}
    for take in ("med", "ho", "po", "mute_med", "mute_hard"):
        for lo, hi, vol, s in r.take(take):
            if lo <= key <= hi:
                out[take] = s
    return out


def pitched(s, key):
    y = sf.read(os.path.join(OUT, "Samples", s["file"]), dtype="float64")[0]
    ratio = 2 ** ((key - s["pitch"]) / 12 + tune(s) / 1200)
    return np.interp(np.arange(0, (len(y) - 1) / ratio, 1.0) * ratio, np.arange(len(y)), y)


def which(x, sr, t0, cands, key):
    n, lag = int(0.2 * sr), int(0.0003 * sr)
    seg = x[int(t0 * sr) - lag:int(t0 * sr) + n + lag]
    best = {}
    for take, s in cands.items():
        y = pitched(s, key)[:n]
        best[take] = max(float(np.dot(seg[k:k + len(y)], y) /
                               (np.linalg.norm(seg[k:k + len(y)]) * np.linalg.norm(y) + 1e-20)) for k in range(2 * lag))
    top = max(best, key=best.get)
    return top, best[top]


def run(case, sfz, events, checks, regions, end):
    os.makedirs(TMP, exist_ok=True)
    sr, x, _ = R.render(sfz, events, os.path.join(TMP, case + ".wav"), end)
    bad = 0
    for t, string, key, want, what in checks:
        got, score = which(x, sr, t, candidates(regions, string, key), key)
        ok = got == want
        bad += not ok
        print("  %-4s %-44s plays %-9s (%.3f)" % ("ok" if ok else "FAIL", what, got, score))
    return bad


def main():
    samples = load()
    regions = {s: StringRegions(samples, s) for s in STRINGS}
    missing = samples_exist()
    print("samples referenced and missing: %d" % len(missing))
    bad = len(missing)
    kb = os.path.join(OUT, NAME + ".sfz")
    on = keyboard_strings()
    string_of = {k: s for s, keys in on.items() for k in keys}

    print("keyboard file: a chord, pedal up, against its notes rendered one at a time")
    chord = [48, 52, 55, 60, 64]
    os.makedirs(TMP, exist_ok=True)
    sr, x, _ = R.render(kb, sum((note(0.5, k, 70, 1.5) for k in chord), []), os.path.join(TMP, "chord.wav"), 2.5)
    parts = [R.render(kb, note(0.5, k, 70, 1.5), os.path.join(TMP, "chord-%d.wav" % k), 2.5)[1] for k in chord]
    n = min([len(x)] + [len(y) for y in parts])
    alone = sum(y[:n] for y in parts)
    diff = 20 * np.log10(np.sqrt(np.mean((x[:n] - alone[:n]) ** 2)) / np.sqrt(np.mean(x[:n] ** 2)) + 1e-12)
    ok = diff < -40     # measured -53 dB (rounding in the renders); with the pedal down, legato samples: +2 dB
    bad += not ok
    print("  %-4s chord %s minus its notes alone: %.0f dB" % ("ok" if ok else "FAIL", chord, diff))

    print("keyboard file: D string 52 -> 54 -> 52, pedal down, then up")
    D = 3
    ev = [cc(0.3, LEGATO_CC, 127)] + note(0.5, 52, 80, 0.31) + note(0.8, 54, 56, 0.31) + note(1.1, 52, 56, 0.8)
    ev += [cc(2.3, LEGATO_CC, 0)] + note(2.5, 52, 80, 0.31) + note(2.8, 54, 56, 0.8)
    checks = [(0.5, D, 52, "med", "first note, pedal down"), (0.8, D, 54, "ho", "52 -> 54 overlapping, pedal down"),
              (1.1, D, 52, "po", "54 -> 52 overlapping, pedal down"), (2.8, D, 54, "med", "52 -> 54 overlapping, pedal up")]
    bad += run("keyboard-legato", kb, ev, checks, regions, 4.0)

    print("keyboard file: keyswitches on the G string (57)")
    G = 4
    ev, checks = [], []
    for t, sw, vel, want in ((0.5, "palm", 56, "mute_med"), (1.5, "palm", 110, "mute_hard"), (2.5, "dead", 110, "mute_hard"),
                             (3.5, "open", 70, "med")):
        ev += note(t - 0.2, KEYSWITCH[sw], 100, 0.05) + note(t, 57, vel, 0.6)
        checks.append((t, G, 57, want, "%s, velocity %d" % (sw, vel)))
    bad += run("keyboard-keyswitch", kb, ev, checks, regions, 4.5)

    print("per-string file (D): automatic legato and CC74")
    ps = os.path.join(OUT, "per-string", "%s-4-d.sfz" % NAME)
    ev = note(0.5, 52, 80, 0.31) + note(0.8, 54, 56, 0.31) + note(1.1, 52, 56, 0.8)
    ev += note(2.5, 54, 70, 0.5) + [cc(3.3, 74, 80)] + note(3.5, 54, 56, 0.5) + [cc(4.3, 74, 127)]
    checks = [(0.8, D, 54, "ho", "52 -> 54 overlapping"), (1.1, D, 52, "po", "54 -> 52 overlapping"),
              (2.5, D, 54, "med", "54 alone"), (3.5, D, 54, "mute_med", "54 with CC74 80")]
    bad += run("per-string", ps, ev, checks, regions, 4.5)
    print("all ok" if not bad else "%d problems" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
