"""Reproduces the synthetic-music numbers in the README (no real data needed; needs the test helpers, librosa and ffmpeg).

    python scripts/synthetic_eval.py            # about 10 minutes
"""
from __future__ import annotations

import collections
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import synth  # noqa: E402
from musicleak import scan  # noqa: E402

REFS, UNREL, REPS, LENGTHS = 25, 40, 3, (10, 20, 30)


def main():
    d = Path(tempfile.mkdtemp())
    (d / "ref").mkdir()
    (d / "q").mkdir()
    songs = [synth.make_song(s, 60.0) for s in range(100, 100 + REFS)]
    for i, x in enumerate(songs):
        synth.write(d / "ref" / f"r{i:02d}.wav", x)
    for i in range(UNREL):
        synth.write(d / "q" / f"unrel{i:02d}.wav", synth.make_song(900 + i, 60.0))
    edits = {"none": None, "reverb": synth.reverb, "mp3_64k": lambda x: synth.to_mp3_and_back(x, kbps=64),
             "pitch+2 (phase vocoder)": lambda x: synth.pitch_shift(x, 2), "tempo+6% (phase vocoder)": lambda x: synth.tempo_change(x, 1.06)}
    truth = {}
    for en, e in edits.items():
        for L in LENGTHS:
            for r in range(REPS):
                k = (len(truth) * 7) % REFS
                q, _ = synth.excerpt_in_new_song(songs[k], 5.0 + r, L, 5000 + len(truth), seconds=60.0, at_s=12.0 + 3 * r, edit=e)
                name = f"copy_{len(truth):03d}.wav"
                synth.write(d / "q" / name, q)
                truth[name] = (en, L, f"r{k:02d}.wav")
    t = time.time()
    rep = scan.run(d / "q", d / "ref")
    res = {r.name: r for r in rep.results}
    fa = collections.Counter(res[f"unrel{i:02d}.wav"].verdict() for i in range(UNREL))
    print(f"scan {(time.time() - t) / 60:.1f} min; levels {rep.calibration.single:.2f}/{rep.calibration.chain:.2f} ({rep.calibration.source})")
    print(f"unrelated songs ({UNREL}): FAIL {fa['FAIL']}, WARN {fa['WARN']}, PASS {fa['PASS']}")
    print("| edit | " + " | ".join(f"{L} s FAIL / flagged" for L in LENGTHS) + " |")
    for en in edits:
        row = []
        for L in LENGTHS:
            names = [n for n, v in truth.items() if v[0] == en and v[1] == L]
            fail = sum(1 for n in names if res[n].verdict() == "FAIL" and any(m.ref == truth[n][2] for m in res[n].matches))
            flag = sum(1 for n in names if any(m.ref == truth[n][2] for m in res[n].matches))
            row.append(f"{fail}/{len(names)} / {flag}/{len(names)}")
        print(f"| {en} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
