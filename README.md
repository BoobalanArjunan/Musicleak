# musicleak

**Does your music model copy its training songs?** musicleak compares generated (or any query) songs with a reference set, such as the
training data, and reports *where* a stretch matches, *which* reference song, *how far* it was shifted in pitch and tempo, and *how sure*
it is. It survives pitch shifts, tempo changes, reverb and MP3 compression, which exact-match tools do not.

> **Status: alpha (0.1.0a1).** Built and tested on synthetic music. It has **not yet been measured on real songs**; do that first
> with [docs/REAL_DATA_CHECK.md](docs/REAL_DATA_CHECK.md) and treat its verdicts as a first pass until then.
> Read [Limits](#limits).

## Why

[MiRA](https://github.com/roserbatlleroca/mira) (ISMIR 2024) is the existing open tool for this question. It returns similarity
distances; it was evaluated on synthetic clips only, with no tests against pitch, tempo or reverb changes, and you choose the cut-off yourself.
musicleak aims at the same question with: a **verdict** from a chance level measured on your own songs, **pitch and tempo search** with the
shift reported, **any format** soundfile reads, and **no model downloads** (plain signal processing, MIT licence).

## Install and use

```bash
pip install --pre musicleak
musicleak check generated_songs/ --reference training_songs/ --out report/ --html report/musicleak_report.html
```

Options: `--control DIR` (songs known to be unrelated, to measure chance; default: the reference songs against each other),
`--window 10` (seconds), `--cache DIR` (keep computed features), `--json`, `--max-files N`.
Exit codes: 0 pass or warn, 1 a confirmed match, 2 usage error. Writes `report.txt`, `report.json`, `matches.csv`, and interactive visual HTML with `--html`.

Example line: `FAIL gen_042.wav (3:10): 32 s matched (17% of the song)` then
`confirmed 1:05-1:37 matches train_0311.flac 0:44-1:16, +3 semitones, tempo x1.06, score 0.84`.


### Python API (Use in Code)

You can run MusicLeak directly in your Python ML workflows or data evaluation pipelines:

```python
from pathlib import Path
from musicleak import run, report_to_html, write_html_report

# Run audit against training reference songs
rep = run(
    query=Path("generated_songs/"),
    reference=Path("training_songs/"),
    window_s=10.0
)

print(f"Verdict: {rep.verdict()}")  # PASS, WARN, or FAIL

# Export interactive visual HTML report
write_html_report(rep, Path("report/audit_dashboard.html"))
```

## How it works (plain words)

1. Each song becomes 12 numbers every tenth of a second: how strong each note name (C, C#, D ...) is. This keeps the notes and drops most
   of the sound quality, so compression, reverb and EQ barely change it. It is computed twice: over the whole range, and over the
   melody range only, so two songs that share just a chord loop do not look alike.
2. Each 10-second window of a query song is slid along every reference song, in all 12 transpositions and at five tempo ratios (x0.88 to x1.12),
   using a correlation (1 = same shape, 0 = unrelated). A match must hold in **both** note maps.
3. **What counts as a match is measured, not guessed:** the best score of unrelated windows (the control songs, or the references against each
   other) sets the levels. A **FAIL** needs consistent windows (same song, same transposition, steady place) whose ends do not overlap,
   about 20 s of copied audio. A **WARN** is a lone window above every unrelated window seen.

## Measured on synthetic music (reproduce with `scripts/synthetic_eval.py`)

25 reference songs and 40 unrelated 60 s songs; copies planted into unrelated songs, 3 per cell. The 'none' row is the plain copy.
"FAIL / flagged" = confirmed / confirmed-or-warned, pointing at the right reference.

| edit | 10 s | 20 s | 30 s |
|---|---|---|---|
| none | 0/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 |
| reverb | 0/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 |
| MP3 64 kbps | 0/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 |
| pitch +2 (phase vocoder) | 0/3 / 2/3 | 3/3 / 3/3 | 3/3 / 3/3 |
| tempo +6% (phase vocoder) | 0/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 |

Unrelated songs: **0/40 FAIL**, **7/40 WARN** (a lone window above every unrelated window seen: expected now and then; WARN does not change the exit code).
Transposition and tempo were reported correctly for the planted copies in the test suite (up to 3 semitones scores above 0.9; 5 or 6
semitones still score well above unrelated songs but lower, 0.5 to 0.8).
Time: about 4 minutes for 40 + 15 query songs against 25 references on one CPU.

## Limits

- **Synthetic only so far.** The synthetic songs are simpler than real music. Real false-alarm and detection rates are unknown until you run the real-data check.
- **It finds the same notes, not the same recording.** A cover, a re-recording or a song with the same melody over a new backing also matches.
  Drum loops, samples without pitch and speech are not matched at all (no pitch content).
- **Same chords and melody are not copying in law.** This is a screening tool, not legal evidence.
- **Short copies:** about 20 s are needed for a FAIL; 10 to 12 s copies are usually WARN. A WARN on its own is weak.
- **Detuned copies** (a shift that falls between semitones) are only partly covered. Large shifts (5 to 6 semitones) score lower.
- **Heavy mixing:** a copy mixed at equal level with other music is harder (check the real-data table).
- **Scale:** comparing is exhaustive, about 12 ms per window per reference song; 1,000 references and a 4-minute song take around 10 minutes. Use `--cache`.
- **Few references:** with fewer than 8 reference songs and no `--control`, chance cannot be measured and fixed levels are used (the report says so).
- **If your reference set contains covers or duplicates of each other,** the measured chance level is too high and matches are missed; clean the set first (splitsafe-style near-duplicate check).
- Formats: whatever soundfile reads (wav, flac, ogg; mp3 on recent builds). Everything is mixed to mono at 22.05 kHz.

## Tests

`pip install -e ".[dev]"` then `pytest -q` (65 tests, about a minute; needs ffmpeg and librosa for the MP3 and phase-vocoder tests, which skip if missing).
A deliberate-bug check (37 planted bugs in the code) is caught by the suite. CI runs on Linux, macOS and Windows.

## Related work

MiRA (Batlle-Roca et al., ISMIR 2024) for replication assessment; commercial services (audio fingerprinting, lyrics screening) match against rights-holder catalogues instead of your own training set.

Licence: MIT.
