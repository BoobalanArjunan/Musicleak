"""Audio to a pitch-class ("chroma") sequence: 12 numbers per frame, 10 frames a second.

Chroma keeps which notes are sounding and throws away most of the timbre. That is why a copy survives MP3 compression, reverb,
EQ and re-mixing. A pitch shift becomes a rotation of the 12 numbers, so we can look for it. Whole-semitone shifts are covered;
shifts that fall between semitones (a detuned copy) are only partly covered (see the README limits)."""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import convolve1d
from scipy.signal import resample_poly

FS = 22050                    # every file is converted to this rate
N_FFT = 4096
HOP = 2205                    # 10 frames per second
FPS = FS / HOP
FMIN, FMAX = 65.0, 2100.0     # C2 to about C7
EDGE = 80.0                   # half-width (Hz) of the soft edge around MELODY_FROM
MELODY_FROM = 330.0           # second band: from here up (melody and upper chord notes), so songs that only share a chord loop differ
BANDS = ((FMIN, FMAX), (MELODY_FROM - EDGE, FMAX))   # two overlapping bands, each scored on its own; a copy must match in both
SMOOTH_FRAMES = 5             # about half a second: melody notes and chords, not the exact attack
SHARPEN = 4                   # raise chroma to this power: keeps the strong notes, drops the haze that makes unrelated songs look alike
SILENCE_REL = 0.03            # a frame quieter than this share of the loudest frames carries no information


class MusicleakError(Exception):
    pass


def load_mono(path: Path) -> np.ndarray:
    """Read any file soundfile can read (wav, flac, ogg, and mp3 on recent builds), mix to mono, resample to FS."""
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - soundfile raises several types
        raise MusicleakError(f"cannot read {Path(path).name}: {e}") from None
    x = x.mean(axis=1)
    if sr != FS:
        g = gcd(int(sr), FS)
        x = resample_poly(x, FS // g, int(sr) // g).astype(np.float32)
    return x


_bin_class_cache: dict = {}


def _bin_classes(lo: float, hi: float):
    key = (lo, hi)
    if key not in _bin_class_cache:
        f = np.fft.rfftfreq(N_FFT, 1.0 / FS)
        ok = (f >= lo) & (f < hi)
        cls = np.full(len(f), -1)
        cls[ok] = np.round(12 * np.log2(f[ok] / 440.0) + 9).astype(int) % 12     # pitch class, C = 0
        _bin_class_cache[key] = cls
    return _bin_class_cache[key]


def chroma(x: np.ndarray) -> np.ndarray:
    """(24, frames): 12 pitch classes for the full range, then 12 for the melody range. Each band's frames are scaled to length 1;
    frames where a band is silent are all zero in that band."""
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    pad = N_FFT // 2
    x = np.pad(x, (pad, pad), mode="reflect")
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    win = np.hanning(N_FFT).astype(np.float32)
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=1)).astype(np.float32)             # (frames, bins)
    f = np.fft.rfftfreq(N_FFT, 1.0 / FS)
    ref = max(float(np.percentile(spec[:, (f >= FMIN) & (f < FMAX)].sum(axis=1), 95)), 1e-9)
    comp = np.log1p(5000.0 * spec / ref)
    tap = np.clip((f - (MELODY_FROM - EDGE)) / (2 * EDGE), 0.0, 1.0)                 # soft edge, so a transposed note does not jump across it
    k = np.hanning(SMOOTH_FRAMES + 2)[1:-1]
    k /= k.sum()
    bands = []
    for bi, (lo, hi) in enumerate(BANDS):
        cls = _bin_classes(lo, hi)
        w = tap if bi == 1 else np.ones_like(f)
        wc = comp * w[None, :]
        band = np.stack([wc[:, cls == c].sum(axis=1) for c in range(12)])          # (12, frames)
        if SMOOTH_FRAMES > 1:
            band = convolve1d(band, k, axis=1, mode="nearest")
        band = band / np.maximum(np.linalg.norm(band, axis=0), 1e-9)
        band = band ** SHARPEN
        en = (spec * w[None, :])[:, (f >= lo) & (f < hi)].sum(axis=1)                 # loudness before log compression, which would lift hiss
        quiet = en < SILENCE_REL * max(float(np.percentile(en, 95)), 1e-9)
        band = band / np.maximum(np.linalg.norm(band, axis=0), 1e-9)
        band[:, quiet] = 0.0
        bands.append(band)
    return np.concatenate(bands, axis=0).astype(np.float32)
