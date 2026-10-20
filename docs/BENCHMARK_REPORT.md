# MusicLeak Technical Report: Preliminary Synthetic Evaluation & Methodology

**Author:** Boobalan Arjunan  
**Affiliation:** ComposerMind Labs  
**Date:** October 20, 2026  
**Software Release:** MusicLeak `v0.1.0a1` (Alpha)  
**Status:** Synthetic Pilot Verification (Pre-Real Data Benchmark)  
**Repository:** [https://github.com/BoobalanArjunan/Musicleak.git](https://github.com/BoobalanArjunan/Musicleak.git)  

---

### Executive Summary & Methodological Context

As generative music models approach widespread deployment, detecting training-data memorization and potential copyright infringement is an active area of research. Existing open academic tools, notably MiRA (*Musical Replication Assessment*, Batlle-Roca et al., ISMIR 2024), explore continuous embedding distances for replication assessment.

**MusicLeak** proposes an alternative signal-processing approach using dual-layer harmonic chroma, circular cross-correlation across 12 transpositions, multi-scale tempo alignment, and empirical null-distribution calibration to output discrete `PASS`, `WARN`, or `FAIL` verdicts with timestamped match intervals.

This document summarizes **preliminary synthetic benchmarks** conducted during initial alpha development. **These results are pilot observations on synthetic signals and do not constitute definitive proof of superiority on real-world music.** Full empirical claims require comprehensive evaluation on large real-world corpora (such as MUSDB18-HQ) comparing against MiRA across equivalent detection tasks at calibrated false-alarm thresholds.

---

### 1. Synthetic Stress-Test Protocol (Controlled Pilot)

The initial validation was executed using `scripts/synthetic_eval.py` to verify whether the algorithm functions under ideal mathematical transformations:

* **Reference Set:** 25 synthesized musical pieces (60.0s each) with distinct harmonic progressions.
* **Unrelated Control Set:** 40 independent synthesized musical pieces (60.0s each) to measure baseline false-positive tendencies.
* **Planted Copies (45 tracks):** Excerpts of 10s, 20s, and 30s spliced into unrelated host songs under 5 conditions (3 trials per cell):
  1. `none`: Unaltered copy.
  2. `reverb`: Synthetic room reverberation.
  3. `mp3_64k`: 64 kbps MP3 compression.
  4. `pitch+2`: Pitch transposition of +2 semitones via phase vocoder.
  5. `tempo+6%`: Tempo change of +6% speed via phase vocoder.

#### Observed Results on Synthetic Data:

| Condition | 10s Clip (FAIL / Flagged) | 20s Clip (FAIL / Flagged) | 30s Clip (FAIL / Flagged) | Notes |
|---|:---:|:---:|:---:|---|
| **Unaltered Copy (`none`)** | 0/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 | Full recovery on $\ge 20$s |
| **Room Reverb (`reverb`)** | 0/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 | Reverb did not disrupt chroma correlation |
| **MP3 64 kbps (`mp3_64k`)** | 0/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 | High-frequency cutoff preserved pitch classes |
| **Pitch Shift +2 Semitones (`pitch+2`)** | 0/3 / 2/3 | 3/3 / 3/3 | 3/3 / 3/3 | Detected across 12-semitone rotation |
| **Tempo Change +6% Speed (`tempo+6%`)** | 0/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 | Aligned via multi-scale tempo grid |

* **Unrelated Control Observations (40 tracks):** 0 `FAIL` verdicts, 7 `WARN` verdicts (single-window chance spikes, which do not alter the exit code), and 33 `PASS` verdicts.
* **Calibration Thresholds:** Empirical null distribution calculated at 0.69 (single window) and 0.60 (chained window).

---

### 2. Pilot Demonstration with MiRA (Small-Sample Observation)

A preliminary pilot run was conducted using MiRA's CLAP module (`mira-sim`) on a small demo set (8 reference tracks vs. 1 clean track and 1 copied track with pitch/tempo shift):

```csv
songA,songB,clap_score
catalog_song_04,ai_gen_memorized_track,0.9582   (Actual copied track)
catalog_song_04,ai_gen_original_ambient,0.9614   (Unrelated track in similar genre)
catalog_song_05,ai_gen_memorized_track,0.9585
catalog_song_02,ai_gen_original_ambient,0.9604
```

#### Objective Analysis of this Observation:
1. **Metric Specialization:** CLAP embeddings compute general acoustic and stylistic similarity. When evaluating tracks within the same genre (e.g. ambient), CLAP naturally produces tight clusters of high scores (~0.93–0.96). This behavior is expected for a genre-similarity metric and does not mean MiRA is invalid as a research tool.
2. **Task Distinction:** MusicLeak evaluates melodic note sequences, producing a `FAIL` verdict specifically for `catalog_song_04` and `PASS` for the unrelated track. However, because this test was conducted on only **two test files** and evaluated only one of MiRA's available metrics (`clap`), **it is a demo observation, not a statistical benchmark.**
3. **Security Note on Checkpoint Loading:** Running `mira-sim` on PyTorch >= 2.6 required setting `weights_only=False` due to legacy NumPy scalars in the checkpoint. While safe for the known official LAION checkpoint, this should not be done on unverified third-party checkpoints due to arbitrary code execution risks.

---

### 3. Rigorous Roadmap for Peer-Reviewed Validation (Tasks 3–8)

To establish fair, scientific, and peer-review-grade claims for future publication (such as ISMIR 2027), the following experimental protocol is established:

1. **Real-Corpus Benchmark (MUSDB18-HQ):**
   - Run `scripts/real_data_check.py` on the complete MUSDB18-HQ dataset (full songs across diverse human performances, mixing styles, and vocal/instrumental arrangements).
   - Measure true false-alarm rates on hundreds of un-related commercial tracks.
   - Measure empirical catch rates across varying copy lengths (5s, 10s, 15s, 20s, 30s) and noise/bleed conditions.
2. **Head-to-Head Evaluation Against Full MiRA Suite:**
   - Run MiRA's full metric suite (`clap`, `coverid`, `defnet`, `kld`) on the exact same MUSDB18-HQ evaluation splits.
   - Compare performance using standard statistical measures:
     - Area Under the ROC Curve (AUC-ROC)
     - Recall at a fixed false-alarm rate (e.g., FAR = 1% and 5%)
     - Mean Reciprocal Rank (MRR) for candidate identification.
3. **Transparent Reporting of Failure Modes:**
   - Fully document scenarios where MusicLeak struggles, such as non-tonal percussion loops, unpitched audio, extreme timbral masking, or microtonal/detuned audio between standard semitones.

---

### Conclusion

MusicLeak's dual-layer chroma and empirical calibration demonstrate solid preliminary behavior on synthetic test cases. The project is actively following the rigorous MUSDB18-HQ real-data benchmark protocol to produce verified, reproducible metrics before making comparative claims in academic or public domains.
