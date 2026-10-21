"""Real-music check for musicleak, using MUSDB18-HQ (or any folder of full songs).

It answers the questions that synthetic tests cannot:
  1. FALSE ALARMS: do unrelated real songs get flagged? (queries taken from songs that are NOT in the reference set)
  2. DETECTION: are planted copies found, by kind of edit and by copied length? (pitch, tempo, reverb, MP3, mixed with other music)
  3. DETAIL: when found, are the reported transposition and tempo right?

    python scripts/real_data_check.py --musdb ~/MUSDB18HQ --out real_check

MUSDB18-HQ layout: <root>/train/<song>/mixture.wav and <root>/test/<song>/mixture.wav. Any folder works if you pass
--reference-dir and --unrelated-dir (folders of audio files) instead. Your files are never modified.

Optional extras for the edits: librosa (pitch and tempo changes) and ffmpeg (MP3). Edits whose tool is missing are skipped and listed.
To compare with MiRA (pip install mira-sim), the script also writes out/reference (links) and out/queries (WAV) so you can run
    mira real_check/reference real_check/queries --eval_name musdb
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from musicleak import scan
    from musicleak.features import FS, load_mono
except ImportError:
    sys.exit("musicleak is not installed (pip install --pre musicleak, or pip install -e . from a checkout)")

try:
    import librosa
except ImportError:
    librosa = None
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def mp3(x, kbps=64):
    d = Path(tempfile.mkdtemp())
    sf.write(str(d / "a.wav"), x, FS)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "a.wav"), "-b:a", f"{kbps}k", str(d / "a.mp3")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "a.mp3"), "-ar", str(FS), str(d / "b.wav")], check=True)
    y, _ = sf.read(str(d / "b.wav"), dtype="float32")
    shutil.rmtree(d, ignore_errors=True)
    return y if y.ndim == 1 else y.mean(axis=1)


def reverb(x, rt60=0.8, wet=0.35, seed=0):
    from scipy.signal import fftconvolve
    n = int(rt60 * FS)
    ir = np.random.RandomState(seed).randn(n) * np.exp(-6.9 * np.arange(n) / n)
    ir /= np.abs(ir).sum() / 4
    y = (1 - wet) * x + wet * fftconvolve(x, ir)[:len(x)]
    return (y / max(1e-9, np.abs(y).max()) * 0.8).astype(np.float32)


# name -> (function piece -> (edited piece, planted transposition, planted tempo ratio), needs)
def edits(unrelated_piece):
    E = {"none": (lambda p: (p, 0, 1.0), None),
         "reverb": (lambda p: (reverb(p), 0, 1.0), None),
         "mp3_64k": (lambda p: (mp3(p), 0, 1.0), "ffmpeg"),
         "pitch+2": (lambda p: (librosa.effects.pitch_shift(p, sr=FS, n_steps=2), 2, 1.0), "librosa"),
         "pitch-3": (lambda p: (librosa.effects.pitch_shift(p, sr=FS, n_steps=-3), -3, 1.0), "librosa"),
         "tempo+6%": (lambda p: (librosa.effects.time_stretch(p, rate=1.06), 0, 1.06), "librosa"),
         "tempo-6%": (lambda p: (librosa.effects.time_stretch(p, rate=0.94), 0, 0.94), "librosa"),
         "mixed with other music": (lambda p: (p + unrelated_piece(len(p)), 0, 1.0), None)}
    return E


def norm(x, peak=0.8):
    return (x / max(1e-9, float(np.abs(x).max())) * peak).astype(np.float32)


def find_songs(a):
    if a.reference_dir:
        return sorted(p for p in Path(a.reference_dir).rglob("*") if p.suffix.lower() in scan.AUDIO_EXT), \
               sorted(p for p in Path(a.unrelated_dir).rglob("*") if p.suffix.lower() in scan.AUDIO_EXT)
    root = Path(a.musdb)
    ref = sorted((root / "train").glob("*/mixture.wav"))
    unrel = sorted((root / "test").glob("*/mixture.wav"))
    return ref, unrel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--musdb", type=Path)
    ap.add_argument("--reference-dir", type=Path)
    ap.add_argument("--unrelated-dir", type=Path)
    ap.add_argument("--refs", type=int, default=40, help="reference songs (default 40)")
    ap.add_argument("--unrelated", type=int, default=40, help="unrelated query clips for the false-alarm rate (default 40)")
    ap.add_argument("--reps", type=int, default=2, help="planted copies per edit and length (default 2)")
    ap.add_argument("--lengths", default="10,20,30", help="copied lengths in seconds (default 10,20,30)")
    ap.add_argument("--host-gain", type=float, default=0.15, help="loudness of the unrelated song the copy is hidden in (default 0.15; 0 = the copy alone, which shows the matcher without mixing)")
    ap.add_argument("--clip-s", type=float, default=60.0, help="length of each query clip (default 60)")
    ap.add_argument("--out", type=Path, default=Path("real_check"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.musdb and not (a.reference_dir and a.unrelated_dir):
        sys.exit("give --musdb, or both --reference-dir and --unrelated-dir")
    ref_files, unrel_files = find_songs(a)
    if len(ref_files) < 10 or len(unrel_files) < 10:
        sys.exit(f"need at least 10 reference and 10 unrelated songs (found {len(ref_files)} and {len(unrel_files)})")
    rng = np.random.RandomState(a.seed)
    ref_files = [ref_files[i] for i in rng.permutation(len(ref_files))[:a.refs]]
    unrel_files = [unrel_files[i] for i in rng.permutation(len(unrel_files))]
    lengths = [int(v) for v in a.lengths.split(",")]
    out = a.out
    for d in ("reference", "queries"):
        shutil.rmtree(out / d, ignore_errors=True)
        (out / d).mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(ref_files):                                    # a folder of the references, for scanning (and for MiRA)
        dst = out / "reference" / f"ref{i:03d}.wav"
        try:
            os.symlink(Path(p).resolve(), dst)
        except OSError:
            shutil.copy(p, dst)

    def clip(p, length_s, start_frac=None):
        x = load_mono(p)
        n = int(length_s * FS)
        s = int((len(x) - n) * (rng.rand() if start_frac is None else start_frac)) if len(x) > n else 0
        return norm(x[s:s + n])

    notes, truth, where = [], {}, {}
    # --- unrelated clips
    pool = itertools.cycle(unrel_files)                                  # small sets: songs are reused instead of running out
    unrel_names = []
    for i in range(a.unrelated):
        p = next(pool)
        name = f"unrelated_{i:03d}.wav"
        sf.write(str(out / "queries" / name), clip(p, a.clip_s), FS)
        unrel_names.append(name)
    other = lambda n: 0.7 * clip(next(pool), n / FS)
    E = edits(other)
    skipped = []
    ok_edits = {}
    for name, (fn, needs) in E.items():
        if needs == "librosa" and librosa is None or needs == "ffmpeg" and not HAVE_FFMPEG:
            skipped.append(f"{name} (needs {needs})")
        else:
            ok_edits[name] = fn
    # --- planted copies
    for name, fn in ok_edits.items():
        for L in lengths:
            for r in range(a.reps):
                src = ref_files[rng.randint(len(ref_files))]
                ri = [i for i, q in enumerate(ref_files) if q == src][0]
                x = load_mono(src)
                start = int(rng.uniform(15, max(16, len(x) / FS - L - 15)) * FS) if len(x) / FS > L + 31 else \
                    int(rng.uniform(0, max(1, len(x) - int(L * FS))))
                piece = x[start:start + int(L * FS)]
                if len(piece) < int(L * FS):
                    piece = np.pad(piece, (0, int(L * FS) - len(piece)))
                edited, trans, tempo = fn(piece.astype(np.float32))
                host = clip(next(pool), a.clip_s)
                at = int(rng.uniform(5, max(6, a.clip_s - L * 1.15 - 5)) * FS)
                q = a.host_gain * host
                end = min(len(q), at + len(edited))
                q[at:end] += norm(edited, 0.8)[:end - at]
                qname = f"copy_{len(truth):03d}.wav"
                sf.write(str(out / "queries" / qname), norm(q), FS)
                truth[qname] = (name, L, f"ref{ri:03d}.wav", trans, tempo)
                where[qname] = {"edit": name, "length_s": L, "reference": f"ref{ri:03d}.wav", "source_song": Path(src).name,
                                "transposition": trans, "tempo": tempo, "at_s": at / FS, "ref_start_s": start / FS,
                                "copy_len_s": len(edited) / FS}
    # --- scan
    t0 = time.time()
    rep = scan.run(out / "queries", out / "reference", progress=scan.stderr_progress)
    secs = time.time() - t0
    res = {r.name: r for r in rep.results}
    L_ = ["# musicleak: real-music check", "",
          f"{len(ref_files)} reference songs, {len(unrel_names)} unrelated clips, {len(truth)} planted copies, clips of {a.clip_s:.0f} s. "
          f"Scan took {secs / 60:.1f} minutes.", ""]
    c = rep.calibration
    L_ += [f"Unrelated-song level measured on {c.n} windows ({c.source}): lone-window level {c.single:.2f}, confirmed level {c.chain:.2f}.", ""]
    if skipped:
        L_ += ["Edits skipped (tool missing): " + ", ".join(skipped), ""]
    fa = collections.Counter(res[n].verdict() for n in unrel_names)
    L_ += ["## 1. False alarms on unrelated real songs", "",
           f"- FAIL (a confirmed match): **{fa['FAIL']}/{len(unrel_names)}**",
           f"- WARN (a lone window above every unrelated window seen): **{fa['WARN']}/{len(unrel_names)}**",
           f"- PASS: {fa['PASS']}/{len(unrel_names)}", "",
           "FAIL here is a false alarm unless the two songs really share material (covers, samples, the same backing track). "
           "Look at those pairs in `report`.", ""]
    L_ += ["## 2. Planted copies found (right reference song)", "",
           "| edit | " + " | ".join(f"{L} s: FAIL / flagged" for L in lengths) + " |", "|---|" + "---|" * len(lengths)]
    detail = collections.defaultdict(list)
    for edit in ok_edits:
        row = []
        for L in lengths:
            names = [n for n, t in truth.items() if t[0] == edit and t[1] == L]
            fail = sum(1 for n in names if res[n].verdict() == "FAIL" and any(m.ref == truth[n][2] for m in res[n].matches))
            flag = sum(1 for n in names if any(m.ref == truth[n][2] for m in res[n].matches))
            row.append(f"{fail}/{len(names)} / {flag}/{len(names)}")
            for n in names:
                for m in res[n].matches:
                    if m.ref == truth[n][2]:
                        detail[edit].append((m.transposition == truth[n][3], abs(m.tempo - truth[n][4]) <= 0.07))
                        break
        L_.append(f"| {edit} | " + " | ".join(row) + " |")
    L_ += ["", "'FAIL' = confirmed match to the right reference. 'flagged' = FAIL or WARN pointing at the right reference.", "",
           "## 3. When found, was the description right?", "", "| edit | transposition right | tempo right (within 0.07) |", "|---|---|---|"]
    for edit, v in detail.items():
        L_.append(f"| {edit} | {sum(x for x, _ in v)}/{len(v)} | {sum(y for _, y in v)}/{len(v)} |")
    wrong = [n for n, t in truth.items() if res[n].matches and not any(m.ref == t[2] for m in res[n].matches)]
    L_ += ["", f"Copies where a match was reported but to the WRONG reference song: {len(wrong)}", "",
           "## Send this file back", "", "Also tell me how long it took and which edits (if any) were skipped."]
    (out / "results.md").write_text("\n".join(L_), encoding="utf-8")
    (out / "truth.json").write_text(json.dumps(where, indent=1), encoding="utf-8")        # ground truth of every planted copy
    from musicleak.report import write_reports
    write_reports(rep, out / "report")
    print("\n".join(L_))
    print(f"\nwritten to {out / 'results.md'}")


if __name__ == "__main__":
    main()
