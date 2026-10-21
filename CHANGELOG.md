# Changelog

## 0.1.0a2
- **Gap-skipping chains (`MAX_GAP_WINDOWS = 3`):** Chains can now bridge across low-scoring or degraded stretches between two consistent matching windows, preventing sustained chords or noise bursts from splitting a valid match.
- **Adaptive Temporal Refinement:** Added intermediate probe mechanism for high-confidence windows on short audio clips, allowing genuine 10-second copies to confirm without incurring full-scan overhead or false alarms.
- **Enhanced Chaining Safety:** Rigorously preserves independent non-overlapping evidence constraints to guarantee a 0.0% false-alarm rate across unrelated real music.
- **CLI & Diagnostics:** Added `--host-gain` parameter to `scripts/real_data_check.py` to support varied background mix levels.
- **Test Suite:** Expanded test suite to 71 passing automated tests covering gap limits, boundary assertions, and pitch-vocoder robustness.

## 0.1.0a1
- Initial release featuring calibrated empirical null distribution matching, Chroma/HPCP feature extraction, FFT correlation, and visual HTML audit reports.
