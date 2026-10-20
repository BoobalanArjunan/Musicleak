import numpy as np
import pytest
import soundfile as sf

from conftest import song
from musicleak.features import FPS, FS, MusicleakError, chroma, load_mono


def tone(hz, sec=3.0):
    t = np.arange(int(sec * FS)) / FS
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_ten_frames_a_second_and_two_bands():
    C = chroma(tone(440, 5.0))
    assert C.shape[0] == 24 and abs(C.shape[1] / 5.0 - 10.0) < 0.5 and FPS == pytest.approx(10.0)


@pytest.mark.parametrize("hz,pc", [(440.0, 9), (261.63, 0), (392.0, 7), (493.88, 11)])
def test_a_pure_note_lights_its_pitch_class(hz, pc):
    C = chroma(tone(hz))
    assert int(np.argmax(C[:12, 15])) == pc


def test_transposing_a_note_rotates_the_pitch_classes():
    a, b = chroma(tone(440.0)), chroma(tone(440.0 * 2 ** (3 / 12)))
    assert int(np.argmax(b[:12, 15])) == (int(np.argmax(a[:12, 15])) + 3) % 12


def test_silence_gives_all_zero_frames():
    x = np.zeros(FS * 4, dtype=np.float32)
    x[FS:2 * FS] = tone(440.0, 1.0)
    C = chroma(x)
    assert np.all(C[:, 30:38] == 0.0) and C[:, 15].any()


def test_active_frames_are_scaled_to_length_one_per_band():
    C = chroma(song(1, 10.0))
    for b in (0, 12):
        norms = np.linalg.norm(C[b:b + 12], axis=0)
        assert np.all((np.abs(norms - 1.0) < 1e-4) | (norms == 0.0)) and (norms > 0).mean() > 0.9


def test_chroma_is_deterministic_and_short_input_is_padded():
    x = song(2, 6.0)
    assert np.array_equal(chroma(x), chroma(x))
    assert chroma(x[:500]).shape[0] == 24


def test_loud_and_quiet_versions_give_the_same_features():
    x = song(3, 8.0)
    assert np.allclose(chroma(x), chroma(x * 0.1), atol=1e-3)


def test_load_resamples_and_mixes_stereo(tmp_path):
    x = song(4, 5.0)
    sf.write(str(tmp_path / "a.wav"), np.stack([x, 0.5 * x], axis=1), 44100)    # the file says 44.1 kHz
    y = load_mono(tmp_path / "a.wav")
    assert abs(len(y) - len(x) * FS / 44100) < 5
    sf.write(str(tmp_path / "b.wav"), np.stack([x, x], axis=1), FS)
    assert np.allclose(load_mono(tmp_path / "b.wav"), x, atol=1e-3)


def test_unreadable_file_is_a_clear_error(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"not audio at all")
    with pytest.raises(MusicleakError, match="bad.wav"):
        load_mono(tmp_path / "bad.wav")


def test_hiss_between_notes_is_treated_as_silence_not_as_a_chord():
    t = np.arange(FS) / FS
    x = np.concatenate([0.5 * np.sin(2 * np.pi * 440 * t), 1e-4 * np.random.RandomState(0).randn(3 * FS)]).astype(np.float32)
    C = chroma(x)
    assert not C[:, 15:35].any() and C[:, 5].any()


def test_sharpening_keeps_a_weaker_note_well_below_a_stronger_one():
    t = np.arange(FS * 4) / FS
    x = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 659.26 * t)).astype(np.float32)
    v = np.sort(chroma(x)[:12, 15])[::-1]
    assert v[1] / v[0] < 0.8                                 # measured 0.72 with sharpening, 0.92 without


def test_features_change_gradually_when_the_note_changes():
    t = np.arange(FS * 4) / FS
    x = (0.5 * np.where(t < 2, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 523.25 * t))).astype(np.float32)
    C = chroma(x)[:12]
    assert np.linalg.norm(np.diff(C, axis=1), axis=0).max() < 0.8     # measured 0.68 smoothed, 0.96 unsmoothed


def test_stereo_is_the_average_of_the_channels(tmp_path):
    x = song(4, 3.0)
    sf.write(str(tmp_path / "l.wav"), np.stack([x, np.zeros_like(x)], axis=1), FS)
    assert np.allclose(load_mono(tmp_path / "l.wav"), 0.5 * x, atol=1e-3)
