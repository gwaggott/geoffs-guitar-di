"""Build Geoff's Guitar DI: the SFZ files, from the samples and their measurements.

Usage:
  python3 tools/build.py [--source DIR] [--zip]

Reads each sample's measurements from tools/measurements.json (guitar.py) and
writes into instrument/:

  geoffs-guitar-di.sfz                   one file for keyboards (below)
  per-string/geoffs-guitar-di-<n>-<string>.sfz
                                         one file per string, string 1 = high E,
                                         for MIDI guitars and MPE (one instance each)

The samples, instrument/Samples/<String>_fNN_<take>.flac (24-bit FLAC, 48 kHz,
mono), come from the release zip. --source DIR makes them instead from the
sliced recordings (WAV files of the same names), as the maintainer does.

Both versions play the same regions: velocity layers soft 1-42, medium 43-84,
hard 85-127 for picked notes, medium 1-84 and hard 85-127 for palm mutes; each
recorded position covers the frets next to it; tune from each sample's measured
pitch, volume from each string and layer's median level under one velocity curve.
They differ in how a note picks its articulation and string:

  per string   CC74 at note-on: 96-127 open, 64-95 palm mute, 0-63 dead; legato
               is automatic: a note-on while the string's last note is held plays
               the hammer-on take if that note was lower, the pull-off take if
               higher (trigger=legato with sw_previous).
  keyboard     keyswitches C1 (24) open, C#1 (25) palm mute, D1 (26) dead, open
               by default; each key plays on one string (the highest whose open
               note is at or below it: the frets nearest the nut, the high E
               above its 4th fret), one note at a time per string; legato only
               while the legato pedal (CC68) is down, as one instance counts held
               notes across all strings, so chords would otherwise play legato.

--zip also writes dist/geoffs-guitar-di-<date>.zip (instrument/, README.md, LICENSE).
"""

import argparse
import datetime
import os
import sys
import zipfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guitar import (CAL_DB, CC74, DEAD_ENVELOPE, DEAD_TAKE, LAYERS, LEGATO_SPAN,  # noqa: E402
                    LEGATO_TAKES, LEVEL_DB, LOWEST_NOTE, MUTE_LAYERS, PEAK_MAX_DB, STRINGS,
                    coverage, load, tune, velcurve)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

NAME = "geoffs-guitar-di"
TITLE = "Geoff's Guitar DI"
OUT = os.path.join(ROOT, "instrument")
TOP_FRET = 22
KEYSWITCH = {"open": 24, "palm": 25, "dead": 26}
LEGATO_CC = 68              # the MIDI legato footswitch
# per-string file order and names: string 1 = high E, as MIDI guitars number them
GUITAR_ORDER = [6, 5, 4, 3, 2, 1]
FILE_STRING = {"HighE": "high-e", "B": "b", "G": "g", "D": "d", "A": "a", "LowE": "low-e"}


class StringRegions:
    """One string's regions, as (lokey, hikey, extra opcodes, sample row)."""

    def __init__(self, samples, string):
        self.string = string
        self.sname, self.open = STRINGS[string]
        self.mine = [s for s in samples if s["string"] == string]
        self.low = LOWEST_NOTE if self.sname == "LowE" else self.open
        self.high = self.open + TOP_FRET

    def take(self, take):
        """[(lokey, hikey, volume, row)] for one take, each position over its frets."""
        rows = sorted((s for s in self.mine if s["take"] == take), key=lambda s: s["fret"])
        if not rows:
            return []
        median = float(np.median([s["level_db"] for s in rows]))
        cover = coverage([s["fret"] for s in rows])
        out = []
        for s in rows:
            lo, hi = cover[s["fret"]]
            lo_key = LOWEST_NOTE if s["fret"] == 0 and self.sname == "LowE" else self.open + lo
            vol = min(LEVEL_DB - CAL_DB - median, PEAK_MAX_DB - CAL_DB - s["peak_db"])
            out.append((lo_key, self.open + hi, vol, s))
        return out

    def legato(self, keys=None):
        """[(key, last key, volume, row)]: the hammer-on take if the last key is lower,
        the pull-off take if higher, else the other one."""
        by_take = {}
        for take in LEGATO_TAKES:
            by_take[take] = {k: (vol, s) for lo, hi, vol, s in self.take(take) for k in range(lo, hi + 1)}
        out = []
        for key in keys or range(self.low, self.high + 1):
            for last in range(key - LEGATO_SPAN, key + LEGATO_SPAN + 1):
                if last == key or not 0 <= last <= 127:
                    continue
                order = ("ho", "po") if last < key else ("po", "ho")
                take = next((t for t in order if key in by_take[t]), None)
                if take is not None:
                    out.append((key, last) + by_take[take][key])
        return out


def region(lo, hi, vol, s, extra=""):
    keys = "key=%d" % lo if lo == hi else "lokey=%d hikey=%d" % (lo, hi)
    return "<region> %s pitch_keycenter=%d%s volume=%.2f tune=%.1f sample=%s" % (
        keys, s["pitch"], extra, vol, tune(s), s["file"])


def clip(lo, hi, keys):
    """The part of lo..hi inside keys (a range), or None."""
    a, b = max(lo, keys.start), min(hi, keys.stop - 1)
    return (a, b) if a <= b else None


def picked(r, keys, extra=""):
    lines = []
    for take, (lv, hv) in LAYERS.items():
        for lo, hi, vol, s in r.take(take):
            c = clip(lo, hi, keys)
            if c:
                lines.append(region(*c, vol, s, " lovel=%d hivel=%d%s" % (lv, hv, extra)))
    return lines


def muted(r, keys):
    lines = []
    for take, (lv, hv) in MUTE_LAYERS.items():
        for lo, hi, vol, s in r.take(take):
            c = clip(lo, hi, keys)
            if c:
                lines.append(region(*c, vol, s, " lovel=%d hivel=%d" % (lv, hv)))
    return lines


def dead(r, keys):
    return [region(*c, vol, s) for lo, hi, vol, s in r.take(DEAD_TAKE) if (c := clip(lo, hi, keys))]


def legato(r, keys):
    return ["<region> key=%d sw_previous=%d pitch_keycenter=%d volume=%.2f tune=%.1f sample=%s" % (
        key, last, s["pitch"], vol, tune(s), s["file"]) for key, last, vol, s in r.legato(keys)]


def header(comment, default_path):
    return ["// %s: %s" % (TITLE, comment),
            "// Samples and SFZ files: CC BY 4.0, Geoffrey Waggott (LICENSE; no credit",
            "// needed for music made with them). Built by tools/build.py.", "",
            "<control>", "default_path=%s" % default_path, "set_cc74=127", "",
            "<global>", "bend_up=1200 bend_down=-1200", velcurve(),
            "ampeg_attack=0 ampeg_release=0.2 loop_mode=no_loop"]


def per_string(r, n):
    keys = range(r.low, r.high + 1)
    lines = header("string %d (%s) alone, for one sampler per string" % (n, r.sname), "../Samples/")
    lines[-1] += " group=1 polyphony=1"
    lines += ["", "// open: picked when no other note is held, else legato",
              "<group> locc74=%d hicc74=%d trigger=first" % CC74["open"]] + picked(r, keys)
    lines += ["", "<group> locc74=%d hicc74=%d trigger=legato" % CC74["open"]] + legato(r, keys)
    lines += ["", "// palm mute", "<group> locc74=%d hicc74=%d" % CC74["palm"]] + muted(r, keys)
    lines += ["", "// dead: a stand-in, the hard palm mutes cut short",
              "<group> locc74=%d hicc74=%d %s" % (CC74["dead"] + (DEAD_ENVELOPE,))] + dead(r, keys)
    return lines


def keyboard_strings():
    """{string: range of keys it plays} on the keyboard file."""
    opens = sorted((o, s) for s, (_, o) in STRINGS.items())
    out = {}
    for k, (o, s) in enumerate(opens):
        lo = LOWEST_NOTE if k == 0 else o
        hi = opens[k + 1][0] - 1 if k + 1 < len(opens) else o + TOP_FRET
        out[s] = range(lo, hi + 1)
    return out


def keyboard(regions):
    lines = header("one file for keyboards", "Samples/")
    lines[-1] += " sw_lokey=%d sw_hikey=%d sw_default=%d" % (KEYSWITCH["open"], KEYSWITCH["dead"], KEYSWITCH["open"])
    lines += ["// keyswitches: C1 open, C#1 palm mute, D1 dead; legato while CC68 (legato pedal) is down"]
    for k, (string, keys) in enumerate(sorted(keyboard_strings().items(), key=lambda kv: kv[1].start)):
        r = regions[string]
        lines += ["", "// %s string: keys %d-%d" % (r.sname, keys.start, keys.stop - 1),
                  "<master> group=%d polyphony=1" % (k + 1),
                  "<group> sw_last=%d hicc%d=63" % (KEYSWITCH["open"], LEGATO_CC)] + picked(r, keys)
        lines += ["<group> sw_last=%d locc%d=64 trigger=first" % (KEYSWITCH["open"], LEGATO_CC)] + picked(r, keys)
        lines += ["<group> sw_last=%d locc%d=64 trigger=legato" % (KEYSWITCH["open"], LEGATO_CC)] + legato(r, keys)
        lines += ["<group> sw_last=%d" % KEYSWITCH["palm"]] + muted(r, keys)
        lines += ["<group> sw_last=%d %s" % (KEYSWITCH["dead"], DEAD_ENVELOPE)] + dead(r, keys)
    return lines


def write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return sum(1 for line in lines if line.startswith("<region>"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="make the FLAC samples from the WAV files in DIR")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    samples = load()
    sdir = os.path.join(OUT, "Samples")
    os.makedirs(sdir, exist_ok=True)
    wanted = {s["file"] for s in samples}
    if a.source:
        for s in samples:
            dst = os.path.join(sdir, s["file"])
            src = os.path.join(a.source, s["file"][:-5] + ".wav")
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                x, sr = sf.read(src, dtype="int32")
                sf.write(dst, x, sr, format="FLAC", subtype="PCM_24")
        for f in os.listdir(sdir):
            if f not in wanted:
                os.remove(os.path.join(sdir, f))
    missing = sorted(wanted - set(os.listdir(sdir)))
    if missing:
        sys.exit("%d samples missing from instrument/Samples/ (e.g. %s): unpack the release zip there, "
                 "or give --source" % (len(missing), missing[0]))
    regions = {s: StringRegions(samples, s) for s in STRINGS}
    for n, string in enumerate(GUITAR_ORDER, 1):
        name = "%s-%d-%s.sfz" % (NAME, n, FILE_STRING[STRINGS[string][0]])
        print("%-34s %4d regions" % ("per-string/" + name, write(os.path.join(OUT, "per-string", name),
                                                                 per_string(regions[string], n))))
    print("%-34s %4d regions" % (NAME + ".sfz", write(os.path.join(OUT, NAME + ".sfz"), keyboard(regions))))
    size = sum(os.path.getsize(os.path.join(sdir, f)) for f in wanted)
    print("%d samples, %.0f MB of FLAC" % (len(wanted), size / 1e6))
    if a.zip:
        os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
        zpath = os.path.join(ROOT, "dist", "%s-%s.zip" % (NAME, datetime.date.today().isoformat()))
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for base, _, files in os.walk(OUT):
                for f in sorted(files):
                    p = os.path.join(base, f)
                    z.write(p, os.path.join(NAME, os.path.relpath(p, OUT)))
            for f in ("README.md", "LICENSE"):
                z.write(os.path.join(ROOT, f), os.path.join(NAME, f))
        print("wrote", zpath)


if __name__ == "__main__":
    main()
