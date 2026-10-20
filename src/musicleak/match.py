"""Finding where a query song's windows appear in reference songs.

Each query window (8 s, every 4 s) is compared with every position of every reference song, in all 12 transpositions and at a few
tempo ratios, using a normalised correlation of the chroma sequences (like a Pearson correlation: 1 = same shape, 0 = unrelated).
Centring removes the part of the score that only says "same key / similar chord colours"."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FPS

WIN_S = 8.0
HOP_S = 4.0
RATES = (0.88, 0.94, 1.0, 1.06, 1.12)      # how much longer (> 1) or shorter the reference stretch is than the query window


@dataclass
class Hit:
    score: float          # best normalised correlation against this reference
    ref_frame: int        # where in the reference the window starts (frames)
    shift: int            # transposition in semitones (query heard this many semitones up from the reference: see `transposition`)
    rate: float           # tempo ratio


def _rfft_rows(a: np.ndarray, m: int) -> np.ndarray:
    return np.fft.rfft(a, m, axis=1)


class RefIndex:
    """Precomputed transforms for one reference song, reused for every query window."""

    def __init__(self, C: np.ndarray):
        self.C = C
        self.n = C.shape[1]
        self.groups = C.shape[0] // 12
        C64 = C.astype(np.float64)
        self.cs1 = np.concatenate([np.zeros((C.shape[0], 1)), np.cumsum(C64, axis=1)], axis=1)
        self.cs2 = np.concatenate([np.zeros((self.groups, 1)), np.cumsum((C64 ** 2).reshape(self.groups, 12, -1).sum(axis=1), axis=1)], axis=1)
        self._fft: dict[int, np.ndarray] = {}

    def fft(self, m: int) -> np.ndarray:
        if m not in self._fft:
            self._fft[m] = _rfft_rows(self.C, m)
        return self._fft[m]

    def window_stats(self, L: int):
        """For every start t and band: the centred energy of the window, shape (bands, t)."""
        t = self.n - L + 1
        s1 = self.cs1[:, L:L + t] - self.cs1[:, :t]                                  # (12 * bands, t)
        s2 = self.cs2[:, L:L + t] - self.cs2[:, :t]                                  # (bands, t)
        var = s2 - (s1 ** 2).reshape(self.groups, 12, t).sum(axis=1) / L
        return np.maximum(var, 0.0)


COMBINE = "min"           # how the bands' scores are combined: "min" (every band must match), "mean"
BAND_MIN_ENERGY = 1e-3    # a query band with less centred energy than this (per frame) is silent and is left out of the score


def best_in_ref(q: np.ndarray, ref: RefIndex, rates=RATES) -> Hit | None:
    """Best match of query window `q` (12 * bands, L) anywhere in `ref`, over transpositions and tempo ratios."""
    best: Hit | None = None
    L0 = q.shape[1]
    G = ref.groups
    idx = (np.arange(12)[:, None] + np.arange(12)[None, :])                           # [s, k] -> channel (k + s) mod 12 (+12 for the doubled array)
    for r in rates:
        L = int(round(L0 * r))
        if ref.n < L + 1 or L < 8:
            continue
        if L == L0:
            qq = q
        else:
            src = np.linspace(0, L0 - 1, L)
            qq = np.stack([np.interp(src, np.arange(L0), row) for row in q])
        qc = qq - qq.mean(axis=1, keepdims=True)
        m = 1 << int(np.ceil(np.log2(ref.n + L)))
        fr_all = ref.fft(m)
        var = ref.window_stats(L)                                                      # (G, positions)
        maps = []
        for g in range(G):
            qg = qc[12 * g:12 * g + 12]
            qn = float(np.sqrt((qg ** 2).sum()))
            if qn ** 2 / L < BAND_MIN_ENERGY:
                continue                                                               # this band is silent in the query window
            fq = _rfft_rows(qg[:, ::-1], m)
            fr = fr_all[12 * g:12 * g + 12]
            fr2 = np.concatenate([fr, fr], axis=0)
            prod = np.einsum("kb,skb->sb", fq, fr2[idx])                               # (12 shifts, bins)
            corr = np.fft.irfft(prod, m, axis=1)[:, L - 1:ref.n]
            den = qn * np.sqrt(var[g])
            ok = var[g] > 0.05 * max(float(np.percentile(var[g], 90)), 1e-9)           # skip near-silent reference stretches
            maps.append(np.where(ok[None, :], corr / np.maximum(den, 1e-9)[None, :], -1.0))
        if not maps:
            return None
        sc = np.min(maps, axis=0) if COMBINE == "min" else np.mean(maps, axis=0)
        s_, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
        if best is None or sc[s_, j] > best.score:
            best = Hit(float(sc[s_, j]), int(j), int(s_), r)
    return best


def transposition(shift: int) -> int:
    """`shift` is how far the REFERENCE channels were rotated to line up with the query: the query is `shift` semitones
    below the reference, folded into -5..+6."""
    s = (-shift) % 12
    return s - 12 if s > 6 else s
