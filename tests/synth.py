"""Synthetic music with known structure, so tests have known answers.

A song is made from a seed: a key, a tempo, a chord progression, a bass line, a random-walk melody and noise hi-hats. Two songs
with different seeds are unrelated (use `same_chords` to make unrelated songs that share a chord loop, as real songs do)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
MAJOR = np.array([0, 2, 4, 5, 7, 9, 11])
PROGRESSIONS = [[0, 5, 3, 4], [0, 3, 4, 0], [5, 3, 0, 4], [0, 4, 5, 3], [1, 4, 0, 0], [0, 2, 3, 4], [5, 4, 3, 4], [0, 5, 1, 4]]


def _tone(freq, dur, amp, harmonics=(1.0, 0.5, 0.25, 0.12)):
    t = np.arange(int(dur * SR)) / SR
    y = sum(a * np.sin(2 * np.pi * freq * (i + 1) * t) for i, a in enumerate(harmonics))
    env = np.minimum(1.0, np.minimum(t / 0.01, (dur - t) / 0.05 + 0.0)).clip(0, 1) ** 0.7
    return amp * y * env / sum(harmonics)


def make_song(seed: int, seconds: float = 60.0, same_chords: int | None = None, key: int | None = None, bpm: float | None = None) -> np.ndarray:
    """A mono float32 song. The same seed with another `key` or `bpm` is the same song transposed or at another tempo (a copy with a
    known edit and no processing artefacts). `same_chords` picks a chord progression by index, to make unrelated songs that share chords."""
    rng = np.random.RandomState(seed)
    k0, b0 = int(rng.randint(0, 12)), float(rng.uniform(80, 150))     # always drawn, so the same seed gives the same song in any key or tempo
    key = k0 if key is None else key
    bpm = b0 if bpm is None else bpm
    beat = 60.0 / bpm
    drawn = [0] + [int(v) for v in rng.randint(0, 7, size=int(rng.choice([3, 5, 7])))]   # a fresh random progression
    prog = drawn if same_chords is None else PROGRESSIONS[same_chords]
    n = int(seconds * SR)
    y = np.zeros(n + SR * 8)
    root_hz = lambda semis: 440.0 * 2 ** ((semis + key - 9) / 12.0)
    bar = 4 * beat
    nbars = int(seconds / bar) + 1
    deg = 0
    for b in range(nbars):
        d = prog[b % len(prog)]
        triad = [MAJOR[(d + k) % 7] + 12 * ((d + k) // 7) for k in (0, 2, 4)]
        t0 = int(b * bar * SR)
        for s in triad:
            tone = _tone(root_hz(s - 12), bar, 0.18)
            y[t0:t0 + len(tone)] += tone
        for q in range(4):                                               # bass on every beat
            tone = _tone(root_hz(MAJOR[d] - 24), beat * 0.9, 0.35, (1.0, 0.4))
            tt = int((b * bar + q * beat) * SR)
            y[tt:tt + len(tone)] += tone
        pos = 0.0
        while pos < bar - 1e-6:                                          # melody: random walk, random note lengths
            ln = float(rng.choice([0.5, 0.5, 1.0, 1.0, 1.5, 2.0])) * beat
            ln = min(ln, bar - pos)
            deg = int(np.clip(deg + rng.choice([-2, -1, -1, 0, 1, 1, 2]), -3, 10))
            semis = MAJOR[deg % 7] + 12 * (deg // 7)
            tone = _tone(root_hz(semis + 12), ln, 0.30)
            tt = int((b * bar + pos) * SR)
            y[tt:tt + len(tone)] += tone
            pos += ln
        for q in range(8):                                               # hi-hats
            hh = rng.randn(int(0.04 * SR)) * np.exp(-np.arange(int(0.04 * SR)) / (0.008 * SR)) * 0.08
            tt = int((b * bar + q * beat / 2) * SR)
            y[tt:tt + len(hh)] += hh
    y = y[:n]
    y /= max(1e-9, np.abs(y).max()) / 0.8
    return y.astype(np.float32)


def write(path, x, sr=SR):
    sf.write(str(path), x, sr)
    return Path(path)


def to_mp3_and_back(x, sr=SR, kbps=64, workdir=None):
    """Round trip through ffmpeg's MP3 encoder at the given bitrate (needs ffmpeg)."""
    import tempfile
    d = Path(workdir or tempfile.mkdtemp())
    a, m, b = d / "a.wav", d / "a.mp3", d / "b.wav"
    sf.write(str(a), x, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(a), "-b:a", f"{kbps}k", str(m)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(m), "-ar", str(sr), str(b)], check=True)
    y, _ = sf.read(str(b), dtype="float32")
    return y if y.ndim == 1 else y.mean(axis=1)


def reverb(x, sr=SR, rt60=0.8, wet=0.35, seed=0):
    from scipy.signal import fftconvolve
    rng = np.random.RandomState(seed)
    n = int(rt60 * sr)
    ir = rng.randn(n) * np.exp(-6.9 * np.arange(n) / n)
    ir /= np.abs(ir).sum() / 4
    y = (1 - wet) * x + wet * fftconvolve(x, ir)[:len(x)]
    return (y / max(1e-9, np.abs(y).max()) * 0.8).astype(np.float32)


def pitch_shift(x, semitones, sr=SR):
    import librosa
    return librosa.effects.pitch_shift(x.astype(np.float32), sr=sr, n_steps=semitones).astype(np.float32)


def tempo_change(x, rate, sr=SR):
    import librosa
    return librosa.effects.time_stretch(x.astype(np.float32), rate=rate).astype(np.float32)


def excerpt_in_new_song(src, start_s, length_s, host_seed, seconds=60.0, at_s=20.0, edit=None, sr=SR):
    """A new unrelated song with `length_s` seconds copied from `src` (after `edit`) pasted in at `at_s`. Returns (audio, copied_range)."""
    host = make_song(host_seed, seconds)
    piece = src[int(start_s * sr):int((start_s + length_s) * sr)].copy()
    if edit is not None:
        piece = edit(piece)
    a = int(at_s * sr)
    fade = int(0.02 * sr)
    piece[:fade] *= np.linspace(0, 1, fade)
    piece[-fade:] *= np.linspace(1, 0, fade)
    out = host.copy()
    out[a:a + len(piece)] = 0.15 * out[a:a + len(piece)] + piece[:len(out) - a]
    return out, (at_s, at_s + len(piece) / sr)
