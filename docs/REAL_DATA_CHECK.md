# Real-music check for musicleak (about 30 to 60 minutes, mostly waiting)

The tests use synthetic music. This check uses real songs (MUSDB18-HQ) and measures: false alarms on unrelated songs, how many
planted copies are found (by edit and by length), and whether the reported pitch and tempo are right.

## Tasks
1. Install: `pip install -e ".[dev]"` from the repo. Check ffmpeg is installed (`ffmpeg -version`; on Mac `brew install ffmpeg`).
2. Run the tests: `pytest -q` (expect 65 passed; a few take a minute).
3. Have MUSDB18-HQ unpacked (the same folder you used for mixforge and stemcheck). It needs `train/` and `test/` folders.
4. Run: `python scripts/real_data_check.py --musdb ~/MUSDB18HQ --out real_check`
   (about 10 to 20 minutes; add `--refs 20 --unrelated 20 --reps 1` for a quick first try).
5. Send me `real_check/results.md`.
6. Optional but valuable: compare with MiRA on the same files. `pip install mira-sim`, then
   `mira real_check/reference real_check/queries --eval_name musdb` and send what it prints.
7. Optional: open two or three `real_check/queries/copy_*.wav` next to the song it was copied from, and tell me whether the reported
   times in `real_check/report/report.txt` are where you hear the match.

## How I use the numbers
- False alarms above a few percent: the chance level is measured too low for real music; I raise the confirmed level or lengthen the window.
- Detection low for one edit: I look at that edit's scores and fix or document the limit. The README table will contain only your measured numbers.
- Wrong reference song or wrong pitch/tempo: I fix the description logic before anything is published.
