import functools
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import synth  # noqa: E402

SECONDS = 40.0


@functools.lru_cache(maxsize=None)
def song(seed: int, seconds: float = SECONDS, **kw):
    return synth.make_song(seed, seconds, **kw)


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def have_librosa() -> bool:
    try:
        import librosa  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """Ten unrelated reference songs on disk, with the library and the unrelated-song level measured once."""
    from musicleak.scan import build_library, calibrate
    d = tmp_path_factory.mktemp("ref")
    for i in range(10):
        synth.write(d / f"ref{i:02d}.wav", song(100 + i))
    lib = build_library(d)
    return d, lib, calibrate(lib, None, 10.0, 5.0, n_windows=60)
