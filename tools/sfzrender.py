"""Render MIDI events through an .sfz file with sfizz_render, for check.py.

  render(sfz, events, wav, end_time)   -> (sample rate, mono float array, seconds taken)

events are (seconds, (status, data1, data2)) tuples. They are written to a
Standard MIDI File next to the WAV: format 0, 960 PPQ and a tempo of 3000 BPM, so
one tick is one sample at 48 kHz and event times are sample-accurate.

sfizz_render comes with sfizz (https://github.com/sfztools/sfizz); this uses
$SFIZZ_RENDER, else sfizz_render on the PATH. It runs in its own process group
with a timeout, and the group is killed if it is still alive afterwards.
"""

import os
import shutil
import signal
import struct
import subprocess
import time

import soundfile as sf

PPQ = 960
TEMPO_US = 20000    # 3000 BPM: 48000 ticks per second, one tick per sample at 48 kHz


def sfizz_render():
    path = os.environ.get("SFIZZ_RENDER") or shutil.which("sfizz_render")
    if not path:
        raise SystemExit("sfizz_render not found: install sfizz, or set SFIZZ_RENDER to its path")
    return path


def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def write_mid(path, events, end_time):
    """Format-0 SMF. Times are rounded to ticks."""
    ticks_per_s = PPQ * 1e6 / TEMPO_US
    evs = sorted(((int(round(t * ticks_per_s)), i, msg) for i, (t, msg) in enumerate(events)))
    body = b"\x00\xff\x51\x03" + struct.pack(">I", TEMPO_US)[1:]
    last = 0
    for tick, _, msg in evs:
        body += _vlq(tick - last) + bytes(msg)
        last = tick
    end_tick = max(last, int(round(end_time * ticks_per_s)))
    body += _vlq(end_tick - last) + b"\xff\x2f\x00"
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, PPQ))
        f.write(b"MTrk" + struct.pack(">I", len(body)) + body)


def run(sfz, mid, wav, timeout=300):
    """Run sfizz_render; returns (seconds taken, return code, output text)."""
    if os.path.exists(wav):
        os.remove(wav)
    cmd = [sfizz_render(), "--sfz", sfz, "--midi", mid, "--wav", wav, "-s", "48000"]
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        out = b"timeout"
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
    return time.monotonic() - t0, p.returncode, out.decode(errors="replace")


def render(sfz, events, wav, end_time, timeout=300):
    mid = os.path.splitext(wav)[0] + ".mid"
    write_mid(mid, events, end_time)
    secs, rc, out = run(sfz, mid, wav, timeout)
    if rc != 0 or not os.path.exists(wav):
        raise RuntimeError("sfizz_render failed (%s): %s" % (rc, out[-300:]))
    y, sr = sf.read(wav, dtype="float64", always_2d=True)
    return sr, y[:, 0], secs     # sfizz writes stereo; a centred mono sample gives identical channels
