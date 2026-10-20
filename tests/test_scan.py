import numpy as np
import pytest

import synth
from conftest import have_ffmpeg, have_librosa, song
from musicleak import scan
from musicleak.features import FPS, MusicleakError, chroma
from musicleak.scan import Calibration, Match, QueryResult, Report, build_library, calibrate, scan_query, window_starts

SEED = 103
_rng = np.random.RandomState(SEED)
KEY0, BPM0 = int(_rng.randint(0, 12)), float(_rng.uniform(80, 150))


def scan_audio(x, library, name="q"):
    d, lib, cal = library
    return scan_query(name, chroma(x), lib, cal, 10.0, 5.0)


def with_copy(src, length, host_seed=900, start=5.0, at=12.0):
    return synth.excerpt_in_new_song(src, start, length, host_seed, seconds=40.0, at_s=at)


def test_window_starts_cover_the_song_and_end_on_the_last_frame():
    s = window_starts(440, 100, 50)                       # the tail after the last regular window is longer than half a hop: one more window
    assert s[0] == 0 and s[-1] == 340 and all(b - a <= 50 for a, b in zip(s, s[1:]))
    assert window_starts(405, 100, 50)[-1] == 300         # a tail of 5 frames is not worth a window
    assert window_starts(99, 100, 50) == []


def test_calibration_measures_unrelated_songs_on_the_users_own_references(library):
    _, lib, cal = library
    assert cal.calibrated and cal.source == "references" and cal.n >= 40
    assert 0.2 < cal.chain <= cal.single < 0.95


def test_too_few_references_fall_back_to_fixed_levels_and_say_so(tmp_path):
    for i in range(3):
        synth.write(tmp_path / f"r{i}.wav", song(200 + i))
    lib = build_library(tmp_path)
    cal = calibrate(lib, None, 10.0, 5.0)
    assert not cal.calibrated and (cal.single, cal.chain) == scan.FALLBACK


def test_a_control_folder_is_used_when_given(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "c").mkdir()
    for i in range(3):
        synth.write(tmp_path / "r" / f"r{i}.wav", song(200 + i))
        synth.write(tmp_path / "c" / f"c{i}.wav", song(300 + i))
    cal = calibrate(build_library(tmp_path / "r"), build_library(tmp_path / "c"), 10.0, 5.0, n_windows=20)
    assert cal.source == "control" and cal.calibrated and cal.n >= 20


def test_short_and_unreadable_reference_files_are_left_out_with_a_note(tmp_path):
    synth.write(tmp_path / "ok.wav", song(1))
    synth.write(tmp_path / "short.wav", song(2, 4.0))
    (tmp_path / "bad.wav").write_bytes(b"junk")
    notes = []
    lib = build_library(tmp_path, notes=notes)
    assert lib.names == ["ok.wav"] and any("short.wav" in n for n in notes) and any("bad.wav" in n for n in notes)


def test_no_usable_reference_is_an_error(tmp_path):
    synth.write(tmp_path / "short.wav", song(2, 4.0))
    with pytest.raises(MusicleakError, match="no usable reference"):
        build_library(tmp_path)


def test_a_plain_copy_is_a_confirmed_match_with_the_right_places(library):
    x, (a, b) = with_copy(song(SEED), 30.0)
    r = scan_audio(x, library)
    m = [m for m in r.matches if m.kind == "confirmed"]
    assert r.verdict() == "FAIL" and len(m) == 1
    assert m[0].ref == "ref03.wav" and m[0].transposition == 0 and m[0].tempo == pytest.approx(1.0, abs=0.07)
    assert abs(m[0].q_start - a) <= 6 and abs(m[0].q_end - b) <= 6
    assert abs(m[0].r_start - 5.0) <= 6 and 15 <= r.matched_s() <= 36


def test_a_transposed_and_faster_copy_is_found_and_described(library):
    src = song(SEED, key=(KEY0 + 3) % 12, bpm=BPM0 * 1.06)                  # the same song, 3 semitones up and 6% faster
    x, _ = with_copy(src, 28.0, start=6.0)
    r = scan_audio(x, library)
    conf = [m for m in r.matches if m.kind == "confirmed"]
    assert r.verdict() == "FAIL" and conf and conf[0].ref == "ref03.wav"
    assert conf[0].transposition == 3 and conf[0].tempo == pytest.approx(1.06, abs=0.07)


def test_a_copy_with_reverb_is_found(library):
    x, _ = with_copy(synth.reverb(song(SEED)), 30.0)
    assert scan_audio(x, library).verdict() == "FAIL"


@pytest.mark.skipif(not have_ffmpeg(), reason="needs ffmpeg")
def test_a_copy_through_low_bitrate_mp3_is_found(library):
    x, _ = with_copy(song(SEED), 30.0)
    assert scan_audio(synth.to_mp3_and_back(x, kbps=64), library).verdict() == "FAIL"


@pytest.mark.skipif(not have_librosa(), reason="needs librosa")
def test_a_pitch_shifted_copy_made_by_a_phase_vocoder_is_found(library):
    x, _ = with_copy(synth.pitch_shift(song(SEED), 2), 30.0)
    r = scan_audio(x, library)
    assert r.verdict() == "FAIL" and any(m.transposition == 2 for m in r.matches)


def test_a_short_copy_is_flagged_but_only_as_possible_or_confirmed_never_missed(library):
    x, _ = with_copy(song(SEED), 12.0)
    r = scan_audio(x, library)
    assert r.verdict() in ("WARN", "FAIL") and any(m.ref == "ref03.wav" for m in r.matches)


def test_unrelated_songs_are_not_failed(library):
    verdicts = [scan_audio(song(700 + i), library, f"u{i}").verdict() for i in range(8)]
    assert "FAIL" not in verdicts


def test_a_song_that_only_shares_a_chord_loop_with_a_reference_is_not_failed(tmp_path):
    for i in range(9):
        synth.write(tmp_path / f"r{i}.wav", song(40 + i, same_chords=0, key=2, bpm=100))
    lib = build_library(tmp_path)
    cal = calibrate(lib, None, 10.0, 5.0, n_windows=60)
    r = scan_query("u", chroma(song(99, same_chords=0, key=2, bpm=100)), lib, cal, 10.0, 5.0)
    assert r.verdict() != "FAIL"


def test_a_reference_song_checked_against_the_other_references_only(tmp_path):
    for i in range(3):
        synth.write(tmp_path / f"r{i}.wav", song(60 + i))
    rep = scan.run(tmp_path / "r1.wav", tmp_path)
    assert rep.verdict() == "PASS" and any("also in the reference folder" in n for n in rep.notes)


def test_query_shorter_than_the_window_is_reported_not_crashed(tmp_path, library):
    d, _, _ = library
    synth.write(tmp_path / "tiny.wav", song(5, 6.0))
    rep = scan.run(tmp_path, d, calibration_windows=20)
    assert rep.results[0].error and rep.verdict() == "FAIL"


def test_verdicts_and_matched_seconds_from_matches():
    def mm(kind, a, b):
        return Match(kind, "r", a, b, 0, 1, 0, 1.0, 0.9)
    assert QueryResult("x", 100.0).verdict() == "PASS"
    assert QueryResult("x", 100.0, [mm("possible", 0, 10)]).verdict() == "WARN"
    assert QueryResult("x", 100.0, [mm("possible", 0, 10), mm("confirmed", 20, 40)]).verdict() == "FAIL"
    assert QueryResult("x", 100.0, error="boom").verdict() == "FAIL"
    assert QueryResult("x", 100.0, [mm("confirmed", 0, 20), mm("confirmed", 10, 30), mm("possible", 50, 60)]).matched_s() == 30.0
    r = Report([QueryResult("a", 1.0), QueryResult("b", 1.0, [mm("possible", 0, 1)])], Calibration(0.8, 0.7, 10, "references"))
    assert r.verdict() == "WARN"


def test_the_feature_cache_gives_the_same_features_and_is_reused(tmp_path):
    synth.write(tmp_path / "a.wav", song(1, 15.0))
    cache = tmp_path / "cache"
    c1 = scan.load_chroma(tmp_path / "a.wav", cache)
    assert len(list(cache.glob("*.npy"))) == 1
    c2 = scan.load_chroma(tmp_path / "a.wav", cache)
    assert np.array_equal(c1, c2)
    synth.write(tmp_path / "a.wav", song(2, 15.0))                            # the file changed: a new cache entry, new features
    assert not np.array_equal(scan.load_chroma(tmp_path / "a.wav", cache), c1) and len(list(cache.glob("*.npy"))) == 2


def test_run_checks_its_settings(tmp_path):
    synth.write(tmp_path / "a.wav", song(1, 15.0))
    with pytest.raises(MusicleakError, match="window"):
        scan.run(tmp_path, tmp_path, window_s=2.0)
    with pytest.raises(MusicleakError, match="not a file or a folder"):
        scan.find_audio(tmp_path / "missing")


def test_calibration_levels_are_the_95th_and_99th_percentiles_and_never_below_the_highest(monkeypatch, tmp_path):
    for i in range(9):
        synth.write(tmp_path / f"r{i}.wav", song(400 + i, 40.0))
    lib = build_library(tmp_path)
    scores = iter(np.linspace(0.0, 1.0, 200).tolist() * 2)
    monkeypatch.setattr(scan, "top_score", lambda *a, **k: (next(scores), 0, None))
    cal = calibrate(lib, None, 10.0, 5.0, n_windows=60)
    got = np.linspace(0.0, 1.0, 200)[:cal.n]
    assert cal.chain == pytest.approx(np.percentile(got, 95)) and cal.single == pytest.approx(got.max())
    assert cal.single >= np.percentile(got, 99)


def test_windows_chain_only_if_same_song_same_transposition_and_a_steady_place():
    from musicleak.match import Hit
    hop = 50
    a = (0, Hit(0.9, 100, 3, 1.0), 0)
    assert scan._consistent(a, (0, Hit(0.9, 150, 3, 1.0), 0), hop)               # steady copy: next window 5 s further on
    assert not scan._consistent(a, (1, Hit(0.9, 150, 3, 1.0), 0), hop)           # another reference song
    assert not scan._consistent(a, (0, Hit(0.9, 150, 4, 1.0), 0), hop)           # another transposition
    assert not scan._consistent(a, (0, Hit(0.9, 400, 3, 1.0), 0), hop)           # somewhere else in the song
    assert scan._consistent(a, (0, Hit(0.9, 150 + 14, 3, 1.0), 0), hop)          # within the tolerance
    assert not scan._consistent(a, (0, Hit(0.9, 150 + 20, 3, 1.0), 0), hop)
    assert scan._consistent((0, Hit(0.9, 100, 3, 1.12), 0), (0, Hit(0.9, 156, 3, 1.12), 0), hop)
