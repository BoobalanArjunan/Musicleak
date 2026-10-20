import numpy as np
import pytest

import synth
from conftest import song
from musicleak import match
from musicleak.features import FPS, chroma
from musicleak.match import RefIndex, best_in_ref, transposition

L = int(10 * FPS)
SEED = 7
_rng = np.random.RandomState(SEED)
KEY0, BPM0 = int(_rng.randint(0, 12)), float(_rng.uniform(80, 150))     # what make_song(SEED) drew


@pytest.fixture(scope="module")
def ref():
    return RefIndex(chroma(song(SEED)))


def test_a_window_of_the_reference_matches_itself_exactly(ref):
    h = best_in_ref(ref.C[:, 150:150 + L], ref)
    assert h.score > 0.999 and h.ref_frame == 150 and h.shift == 0 and h.rate == 1.0


@pytest.mark.parametrize("semis,min_score", [(1, 0.9), (3, 0.9), (-2, 0.9), (-4, 0.8), (5, 0.7), (-5, 0.5), (6, 0.7)])
def test_a_transposed_copy_is_found_with_the_right_sign(ref, semis, min_score):
    # measured: the score falls as the shift grows (band edges move), but stays far above unrelated songs (about 0.4 at most)
    up = chroma(synth.make_song(SEED, 40.0, key=(KEY0 + semis) % 12))
    h = best_in_ref(up[:, 150:150 + L], ref)
    assert h.score > min_score and transposition(h.shift) == semis and abs(h.ref_frame - 150) <= 3


def test_transposition_folds_into_a_range_and_has_the_documented_sign():
    assert [transposition(s) for s in range(12)] == [0, -1, -2, -3, -4, -5, 6, 5, 4, 3, 2, 1]


@pytest.mark.parametrize("ratio", [0.94, 1.06, 1.12])
def test_a_copy_at_another_tempo_is_found_and_the_ratio_reported(ref, ratio):
    fast = chroma(synth.make_song(SEED, 40.0, bpm=BPM0 * ratio))
    w0 = 100
    h = best_in_ref(fast[:, w0:w0 + L], ref)
    assert h.score > 0.85 and h.rate == pytest.approx(ratio, abs=0.07)
    assert abs(h.ref_frame - w0 * ratio) < 15                                  # lands where that part of the song is (1.5 s)


def test_unrelated_song_scores_far_below_a_copy(ref):
    scores = [best_in_ref(chroma(song(500 + i))[:, 100:100 + L], ref).score for i in range(6)]
    assert max(scores) < 0.8


def test_the_two_bands_stop_a_shared_chord_loop_from_looking_like_a_copy():
    a = RefIndex(chroma(song(21, same_chords=0, key=2, bpm=100)))
    b = chroma(song(22, same_chords=0, key=2, bpm=100))
    assert best_in_ref(b[:, 100:100 + L], a).score < 0.8                       # same key, chords and tempo, different melody


def test_silent_query_window_gives_no_hit(ref):
    assert best_in_ref(np.zeros((24, L), dtype=np.float32), ref) is None


def test_reference_shorter_than_the_window_gives_no_hit():
    short = RefIndex(chroma(song(9, 6.0)))
    assert best_in_ref(chroma(song(9))[:, :L], short) is None


def test_silent_reference_stretches_are_not_matched():
    x = np.concatenate([song(SEED, 12.0), np.zeros(22050 * 20, dtype=np.float32)])
    r = RefIndex(chroma(x))
    h = best_in_ref(chroma(song(SEED, 12.0))[:, :L], r)
    assert h.score > 0.95 and h.ref_frame < 30


def test_one_band_silent_in_the_query_is_left_out_of_the_score(ref):
    q = ref.C[:, 150:150 + L].copy()
    q[12:] = 0.0                                                                # no melody-range content at all
    h = best_in_ref(q, ref)
    assert h is not None and h.score > 0.95


def test_mean_combination_is_never_below_min(ref):
    q = chroma(song(8))[:, 100:100 + L]
    saved = match.COMBINE
    try:
        match.COMBINE = "min"
        low = best_in_ref(q, ref).score
        match.COMBINE = "mean"
        mean = best_in_ref(q, ref).score
    finally:
        match.COMBINE = saved
    assert mean >= low - 1e-9


def test_both_bands_must_match_a_strong_match_in_one_band_alone_is_not_enough():
    assert match.COMBINE == "min"                                               # the shipped default
    rng = np.random.RandomState(0)
    n = 600
    A = np.abs(rng.randn(12, n)).astype(np.float32) ** 3
    B1 = np.abs(rng.randn(12, n)).astype(np.float32) ** 3
    B2 = np.abs(rng.randn(12, n)).astype(np.float32) ** 3
    def unit(M):
        return M / np.linalg.norm(M, axis=0, keepdims=True)
    ref = RefIndex(np.concatenate([unit(A), unit(B1)]).astype(np.float32))
    q = np.concatenate([unit(A)[:, 200:200 + L], unit(B2)[:, 200:200 + L]]).astype(np.float32)     # band 1 identical, band 2 unrelated
    assert best_in_ref(q, ref).score < 0.5                                     # a mean of the bands would give about 0.7


def test_low_contrast_reference_stretches_are_not_matched():
    Cb = chroma(song(31, 40.0))[:, :300]
    m = Cb.mean(axis=1, keepdims=True)
    squashed = (m + 0.03 * (Cb - m)).astype(np.float32)                         # the same shape at 3% of the contrast
    ref = RefIndex(np.concatenate([chroma(song(30, 40.0))[:, :300], squashed], axis=1))
    assert best_in_ref(Cb[:, 100:100 + L], ref).score < 0.8                      # scale-free correlation would give about 1.0 there
