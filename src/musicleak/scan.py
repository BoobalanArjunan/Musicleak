"""Running the comparison: build the reference library, calibrate what an 'unrelated' score looks like, scan query songs."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FPS, MusicleakError, chroma, load_mono
from .match import HOP_S, WIN_S, Hit, RefIndex, best_in_ref, transposition

AUDIO_EXT = {".wav", ".flac", ".ogg", ".opus", ".mp3", ".aif", ".aiff"}
SINGLE_PCT = 99.0            # a lone window above this share of unrelated windows (and above all we saw) gives a WARN
CHAIN_PCT = 95.0             # consistent windows above this, the first and last not overlapping (about 2 windows of audio), give a FAIL
MIN_REFS_TO_CALIBRATE = 8    # fewer reference songs than this and the unrelated level cannot be measured from them
FALLBACK = (0.90, 0.85)      # (single, chain) used when nothing can be calibrated; flagged in the report
CALIB_WINDOWS = 150
MAX_GAP_WINDOWS = 3          # a chain may skip up to this many windows that scored low (sustained pads, a damaged stretch) between two that matched
CHAIN_TOL_S = 1.5            # consecutive windows must point to places in the reference this close to where a steady copy would


@dataclass
class Library:
    names: list[str]
    refs: list[RefIndex]
    paths: list[Path]


@dataclass
class Calibration:
    single: float
    chain: float
    n: int
    source: str                 # "control", "references" or "fallback"

    @property
    def calibrated(self) -> bool:
        return self.source != "fallback"


@dataclass
class Match:
    kind: str                   # "confirmed" (a chain of windows) or "possible" (one window)
    ref: str
    q_start: float
    q_end: float
    r_start: float
    r_end: float
    transposition: int          # semitones
    tempo: float                # reference length / query length (1.0 = same speed)
    score: float


@dataclass
class QueryResult:
    name: str
    duration_s: float
    matches: list[Match] = field(default_factory=list)
    error: str | None = None

    def matched_s(self) -> float:
        iv = sorted((m.q_start, m.q_end) for m in self.matches if m.kind == "confirmed")
        tot, end = 0.0, -1.0
        for a, b in iv:
            a = max(a, end)
            if b > a:
                tot += b - a
                end = b
        return tot

    def verdict(self) -> str:
        if self.error:
            return "FAIL"
        if any(m.kind == "confirmed" for m in self.matches):
            return "FAIL"
        return "WARN" if self.matches else "PASS"


@dataclass
class Report:
    results: list[QueryResult]
    calibration: Calibration
    notes: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    def verdict(self) -> str:
        v = {r.verdict() for r in self.results}
        return "FAIL" if "FAIL" in v else "WARN" if "WARN" in v else "PASS"


def find_audio(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise MusicleakError(f"{path} is not a file or a folder")
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise MusicleakError(f"no audio files found in {path}")
    return files


def load_chroma(path: Path, cache: Path | None = None) -> np.ndarray:
    key = None
    if cache is not None:
        st = Path(path).stat()
        key = cache / (hashlib.sha1(f"{Path(path).resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:20] + ".npy")
        if key.is_file():
            try:
                return np.load(key)
            except Exception:  # noqa: BLE001 - a damaged cache file is simply rebuilt
                pass
    C = chroma(load_mono(path))
    if key is not None:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(key, C)
    return C


def build_library(folder: Path, cache: Path | None = None, notes: list[str] | None = None, progress=None) -> Library:
    files = find_audio(folder)
    root = Path(folder) if Path(folder).is_dir() else None
    names, refs, paths = [], [], []
    for i, p in enumerate(files, 1):
        if progress:
            progress("reference", i, len(files), p.name)
        try:
            C = load_chroma(p, cache)
        except MusicleakError as e:
            if notes is not None:
                notes.append(str(e))
            continue
        if C.shape[1] < int(WIN_S * FPS) + 2:
            if notes is not None:
                notes.append(f"{p.name}: shorter than {WIN_S:.0f} s, left out of the reference set")
            continue
        names.append(p.relative_to(root).as_posix() if root else p.name)
        refs.append(RefIndex(C))
        paths.append(p.resolve())
    if not refs:
        raise MusicleakError("no usable reference songs")
    return Library(names, refs, paths)


def window_starts(n_frames: int, win: int, hop: int) -> list[int]:
    if n_frames < win:
        return []
    s = list(range(0, n_frames - win + 1, hop))
    if s[-1] != n_frames - win and (n_frames - win - s[-1]) > hop // 2:
        s.append(n_frames - win)
    return s


def top_score(win: np.ndarray, lib: Library, skip: set[int] = frozenset()) -> tuple[float, int, Hit | None]:
    best, bi, bh = -1.0, -1, None
    for i, r in enumerate(lib.refs):
        if i in skip:
            continue
        h = best_in_ref(win, r)
        if h is not None and h.score > best:
            best, bi, bh = h.score, i, h
    return best, bi, bh


def calibrate(lib: Library, control: Library | None, win_s: float, hop_s: float, seed: int = 0, progress=None,
              n_windows: int = CALIB_WINDOWS) -> Calibration:
    """What does the best match of an UNRELATED window look like? Measured on the user's own songs:
    windows of the control songs (if given) or of the reference songs themselves (each against all the OTHER references)."""
    rng = np.random.RandomState(seed)
    src, srcname = (control, "control") if control is not None else (lib, "references")
    if control is None and len(lib.refs) < MIN_REFS_TO_CALIBRATE:
        return Calibration(FALLBACK[0], FALLBACK[1], 0, "fallback")
    win = int(win_s * FPS)
    cand = []
    for si, r in enumerate(src.refs):
        for w0 in window_starts(r.n, win, int(hop_s * FPS)):
            cand.append((si, w0))
    if not cand:
        return Calibration(FALLBACK[0], FALLBACK[1], 0, "fallback")
    pick = [cand[i] for i in rng.permutation(len(cand))[:n_windows]]
    scores = []
    for k, (si, w0) in enumerate(pick, 1):
        if progress:
            progress("calibrating", k, len(pick), "")
        w = src.refs[si].C[:, w0:w0 + win]
        s, _, _ = top_score(w, lib, skip={si} if control is None else frozenset())
        if s > -1:
            scores.append(s)
    if len(scores) < 20:
        return Calibration(FALLBACK[0], FALLBACK[1], len(scores), "fallback")
    a = np.asarray(scores)
    single = max(float(np.percentile(a, SINGLE_PCT)), float(a.max()))                   # at least as high as any unrelated window we saw
    return Calibration(single, float(np.percentile(a, CHAIN_PCT)), len(a), srcname)


def _consistent(h1: tuple[int, Hit, int], h2: tuple[int, Hit, int], hop_frames: int) -> bool:
    i1, a, _ = h1
    i2, b, _ = h2
    if i1 != i2 or a.shift != b.shift:
        return False
    rate = (a.rate + b.rate) / 2
    return abs((b.ref_frame - a.ref_frame) - hop_frames * rate) <= CHAIN_TOL_S * FPS


def scan_query(name: str, C: np.ndarray, lib: Library, cal: Calibration, win_s: float, hop_s: float, skip: set[int] = frozenset()) -> QueryResult:
    win, hop = int(win_s * FPS), int(hop_s * FPS)
    res = QueryResult(name, C.shape[1] / FPS)
    seq: list[tuple[int, int, Hit] | None] = []                  # per window: (window start, ref index, hit)
    for w0 in window_starts(C.shape[1], win, hop):
        s, bi, h = top_score(C[:, w0:w0 + win], lib, skip)
        seq.append((w0, bi, h) if h is not None and s >= cal.chain else None)
    used = set()
    k = 0
    while k < len(seq):                                           # chains of consistent windows above the chain level; low-scoring windows in between may be skipped
        if seq[k] is None:
            k += 1
            continue
        chain = [k]
        j = k + 1
        while j < len(seq) and j - chain[-1] - 1 <= MAX_GAP_WINDOWS:
            if seq[j] is not None and _consistent((seq[chain[-1]][1], seq[chain[-1]][2], 0), (seq[j][1], seq[j][2], 0), seq[j][0] - seq[chain[-1]][0]):
                chain.append(j)
            j += 1
        # Confirmed if non-overlapping ends (>= win) OR a consistent chain of >= 2 windows with strong evidence (at least one window >= cal.single)
        is_confirmed = seq[chain[-1]][0] - seq[k][0] >= win
        if is_confirmed:
            res.matches.append(_make_match("confirmed", [seq[i] for i in chain], lib, win))
            used.update(chain)
        k = chain[-1] + 1
    # Adaptive refinement: for isolated high-confidence windows (>= cal.single) not chained on the coarse grid,
    # probe a half-step offset specifically against that reference to verify consistent temporal tracking
    for k in range(len(seq)):
        if seq[k] is not None and k not in used and seq[k][2].score >= cal.single:
            w0, bi, h = seq[k]
            confirmed_sub = False
            for probe_w0 in [w0 + hop // 2, w0 - hop // 2]:
                if 0 <= probe_w0 and probe_w0 + win <= C.shape[1]:
                    h_probe = best_in_ref(C[:, probe_w0:probe_w0 + win], lib.refs[bi])
                    if h_probe is not None and h_probe.score >= cal.chain:
                        h1 = (bi, h, 0) if w0 < probe_w0 else (bi, h_probe, 0)
                        h2 = (bi, h_probe, 0) if probe_w0 > w0 else (bi, h, 0)
                        dt = abs(probe_w0 - w0)
                        if _consistent(h1, h2, dt):
                            sub_chain = [(min(w0, probe_w0), bi, h if w0 < probe_w0 else h_probe),
                                         (max(w0, probe_w0), bi, h_probe if probe_w0 > w0 else h)]
                            res.matches.append(_make_match("confirmed", sub_chain, lib, win))
                            used.add(k)
                            confirmed_sub = True
                            break
            if not confirmed_sub:
                res.matches.append(_make_match("possible", [seq[k]], lib, win))
    res.matches.sort(key=lambda m: m.q_start)
    return res


def _make_match(kind: str, ws, lib: Library, win: int) -> Match:
    w0, bi, h0 = ws[0]
    w1, _, h1 = ws[-1]
    rate = float(np.mean([h.rate for _, _, h in ws]))
    return Match(kind, lib.names[bi], w0 / FPS, (w1 + win) / FPS, h0.ref_frame / FPS, (h1.ref_frame + win * h1.rate) / FPS,
                 transposition(h0.shift), rate, float(max(h.score for _, _, h in ws)))


def run(query: Path, reference: Path, control: Path | None = None, *, window_s: float = WIN_S, hop_s: float = HOP_S,
        cache: Path | None = None, max_files: int | None = None, progress=None, calibration_windows: int = CALIB_WINDOWS) -> Report:
    if window_s < 5 or hop_s <= 0 or hop_s > window_s:
        raise MusicleakError("--window must be at least 5 seconds")
    notes: list[str] = []
    lib = build_library(reference, cache, notes, progress)
    ctl = build_library(control, cache, notes, progress) if control else None
    cal = calibrate(lib, ctl, window_s, hop_s, progress=progress, n_windows=calibration_windows)
    if not cal.calibrated:
        notes.append(f"fewer than {MIN_REFS_TO_CALIBRATE} reference songs and no --control folder, so the 'unrelated' level could "
                     f"not be measured: fixed levels {FALLBACK[0]}/{FALLBACK[1]} were used and may be wrong for your music")
    qfiles = find_audio(query)
    if max_files and len(qfiles) > max_files:
        notes.append(f"only the first {max_files} of {len(qfiles)} query files were checked (--max-files)")
        qfiles = qfiles[:max_files]
    qroot = Path(query) if Path(query).is_dir() else None
    out: list[QueryResult] = []
    for i, p in enumerate(qfiles, 1):
        if progress:
            progress("query", i, len(qfiles), p.name)
        name = p.relative_to(qroot).as_posix() if qroot else p.name
        skip = {j for j, rp in enumerate(lib.paths) if rp == p.resolve()}
        if skip:
            notes.append(f"{name}: is also in the reference folder; compared with the other reference songs only")
        try:
            C = load_chroma(p, cache)
            if C.shape[1] < int(window_s * FPS):
                out.append(QueryResult(name, C.shape[1] / FPS, error=f"shorter than the {window_s:.0f} s window"))
                continue
            out.append(scan_query(name, C, lib, cal, window_s, hop_s, skip))
        except MusicleakError as e:
            out.append(QueryResult(name, 0.0, error=str(e)))
    return Report(out, cal, notes, {"window_s": window_s, "hop_s": hop_s, "references": len(lib.refs)})


def stderr_progress(stage: str, i: int, n: int, name: str):
    if n > 1 and (i == n or i % 10 == 0 or stage == "query"):
        print(f"[{stage} {i}/{n}] {name}", file=sys.stderr)
