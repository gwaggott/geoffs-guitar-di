"""The guitar and how its samples map to regions: shared by build.py and check.py.

measurements.json has one row per sample in instrument/Samples/: its string,
fret, MIDI pitch and take, its level and peak (dBFS) and its tuning in cents,
measured on the whole note (cents) and late in it (cents_late).
"""

import json
import os

MEASUREMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements.json")

# {string: (name, open-string MIDI note)}, string 1 = low E
STRINGS = {1: ("LowE", 40), 2: ("A", 45), 3: ("D", 50), 4: ("G", 55), 5: ("B", 59), 6: ("HighE", 64)}
LOWEST_NOTE = 38        # low E covers 38-39 by pitching its open-string sample down

LAYERS = {"soft": (1, 42), "med": (43, 84), "hard": (85, 127)}     # picked: take -> velocities
MUTE_LAYERS = {"mute_med": (1, 84), "mute_hard": (85, 127)}       # palm mutes
LEGATO_TAKES = ("ho", "po")     # hammer-on, pull-off
LEGATO_SPAN = 12        # semitones from the last note that a legato note-on can come
DEAD_TAKE = "mute_hard"         # dead notes, for now: the hard palm mutes cut short
DEAD_ENVELOPE = "ampeg_hold=0.02 ampeg_decay=0.03 ampeg_sustain=0"
CC74 = {"open": (96, 127), "palm": (64, 95), "dead": (0, 63)}     # per-string files

# Region volumes: each string and take's median sample plays at LEVEL_DB, as
# sfizz renders it at full velocity (sfizz renders a sample CAL_DB above its
# level in the file), with no sample's peak above PEAK_MAX_DB.
LEVEL_DB = -20.3
CAL_DB = -11.52
PEAK_MAX_DB = -1.0


def load():
    with open(MEASUREMENTS) as f:
        return json.load(f)["samples"]


def velcurve():
    g = lambda v: 1 - 0.97 * (1 - v / 127)
    return "amp_veltrack=100 amp_velcurve_0=%.4f amp_velcurve_95=1 amp_velcurve_127=1" % (g(0) / g(95))


def coverage(frets, top=22):
    """{recorded fret: (lowest fret, highest fret)} it plays."""
    frets = sorted(frets)
    out = {}
    for k, f in enumerate(frets):
        lo = 0 if f == 0 else (frets[k - 1] + f) // 2 + 1 if k else f
        hi = f if f == 0 else (f + frets[k + 1]) // 2 if k + 1 < len(frets) else top
        if k and frets[k - 1] == 0:
            lo = 1
        out[f] = (lo, hi)
    return out


def tune(s):
    """The region's tune (cents): the sample's measured pitch error, undone."""
    c = s["cents_late"] if s.get("cents_late") is not None else s["cents"]
    return -c if c is not None and abs(c) < 50 else 0.0
