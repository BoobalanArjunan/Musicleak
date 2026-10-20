# MusicLeak — Project Summary & Checkpoint

> **Product:** MusicLeak (`v0.1.0a1`)  
> **Parent Suite:** ComposerMind Labs Audio Engineering Suite  
> **Author & Lead Architect:** Boobalan Arjunan  
> **Repository:** [https://github.com/BoobalanArjunan/Musicleak.git](https://github.com/BoobalanArjunan/Musicleak.git)  
> **Status:** Production-Ready Alpha (`0.1.0a1`), 67/67 Passing Tests, Twine-Verified Packages  

---

## 1. What MusicLeak Does

MusicLeak detects **training-data memorization, replication, and copyright leakage** in generative AI music models (such as MusicLM, Stable Audio, Suno, Udio, or custom generative pipelines).

- **Core Problem:** AI music generation models can memorize entire melodies, chord sequences, and rhythmic hooks from their training audio and reproduce them almost verbatim, creating severe copyright infringement and model governance liabilities.
- **Why Naive Tools Fail:** Simple acoustic fingerprinting (Shazam/Chromaprint) fails whenever generative models alter the key (pitch shift), alter the tempo, add reverb, or compress to MP3. Deep neural similarity models (like ISMIR 2024 MiRA) require heavy weights, arbitrary threshold guesses, and cannot pinpoint the exact matching time window.
- **The MusicLeak Innovation:**
  - **Dual-Layer Note Mapping:** Computes 12-dimensional harmonic chroma profiles across the entire audio spectrum *and* specifically over the melody band (to prevent false alarms on generic chord progressions).
  - **Empirical Null Distribution Calibration:** Automatically measures the chance baseline by cross-correlating unrelated reference tracks, giving calibrated `PASS`, `WARN`, or `FAIL` verdicts with mathematical confidence.
  - **Invariance:** Detects matches despite pitch transposition, tempo drift (0.88x–1.12x), reverb, and MP3 lossy compression.
  - **Exact Forensic Auditing:** Outputs the exact query time window, matching reference track, reference timestamp, semitone offset, tempo ratio, and similarity score.
  - **Zero Model Weights:** Pure mathematical signal processing (NumPy, SciPy, SoundFile). Fast, lightweight, MIT licensed.

---

## 2. Validation & Verification Status

| Gate | Status | Details |
|---|---|---|
| **Unit Test Suite** | **PASS (67/67)** | 67 tests passing in 37.08s (`pytest -q`). |
| **Interactive HTML Report** | **PASS** | Glassmorphic visual dashboard with live filtering, audio timeline segments, match table, and JSON/CSV export. |
| **Packaging & Distribution** | **PASS** | Wheel (`musicleak-0.1.0a1-py3-none-any.whl`) & sdist built; `twine check` PASSED. |
| **GitHub Remote** | **PASS** | Initialized and pushed to `https://github.com/BoobalanArjunan/Musicleak.git` (`main`). |
| **Interactive Demo** | **PASS** | Sample demo suite generated in `demo/` with `musicleak_audit_demo.html`. |

---

## 3. CLI & Python API Usage

### Command Line Interface
```bash
# Basic run with text, JSON, and CSV output
musicleak check generated_songs/ --reference training_songs/ --out report/

# Generate an interactive visual HTML dashboard
musicleak check generated_songs/ --reference training_songs/ --out report/ --html report/musicleak_report.html
```

### Programmatic Python API
```python
from pathlib import Path
from musicleak import run, report_to_html, write_html_report

# Run audit in ML ingestion or training evaluation loop
report = run(
    query=Path("generated_tracks/"),
    reference=Path("training_catalog/"),
    window_s=10.0
)

print("Audit Verdict:", report.verdict())  # PASS, WARN, or FAIL

# Export visual HTML report
write_html_report(report, Path("report/musicleak_audit.html"))
```
