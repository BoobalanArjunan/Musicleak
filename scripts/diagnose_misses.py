"""Why was a planted copy missed? Run after scripts/real_data_check.py (it saves truth.json).

    python scripts/diagnose_misses.py real_check_big

For every planted copy it looks at each window that lies inside the copied part and asks:
  * how high did the TRUE reference song score (compared with the 'chain' and 'single' levels)?
  * did the best match point at the RIGHT place in that song (a loop that repeats can point to a wrong, equal-looking place)?
  * did a DIFFERENT reference song score higher?
  * how flat is the music in the window (low 'movement' = sustained pads/drones, where chroma matching is weak)?
Writes <dir>/diagnosis.md and diagnosis.csv. Your audio is not changed.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from musicleak import scan
    from musicleak.features import FPS
    from musicleak.match import best_in_ref
except ImportError:
    sys.exit("musicleak is not installed (pip install -e . from the checkout)")

POS_TOL_S = 2.0


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    d = Path(sys.argv[1])
    truth = json.loads((d / "truth.json").read_text(encoding="utf-8"))
    rep = json.loads((d / "report" / "report.json").read_text(encoding="utf-8"))
    chain, single = rep["calibration"]["chain"], rep["calibration"]["single"]
    lib = scan.build_library(d / "reference")
    idx = {n: i for i, n in enumerate(lib.names)}
    win, hop = int(8 * FPS), int(4 * FPS)
    rows = []
    for k, (q, t) in enumerate(sorted(truth.items()), 1):
        print(f"  {k}/{len(truth)} {q}", flush=True)
        C = scan.load_chroma(d / "queries" / q)
        ti = idx[t["reference"]]
        a0, a1 = t["at_s"] * FPS, (t["at_s"] + t["copy_len_s"]) * FPS
        for w0 in scan.window_starts(C.shape[1], win, hop):
            inside = (min(w0 + win, a1) - max(w0, a0)) / win
            if inside < 0.9:
                continue
            w = C[:, w0:w0 + win]
            h = best_in_ref(w, lib.refs[ti])
            top, bi, _ = scan.top_score(w, lib)
            exp = t["ref_start_s"] * FPS + (w0 - a0) * t["tempo"]
            rows.append({"query": q, "edit": t["edit"], "length_s": t["length_s"], "window_start_s": round(w0 / FPS, 1),
                         "true_ref_score": round(h.score, 3) if h else None,
                         "position_error_s": round(abs(h.ref_frame - exp) / FPS, 1) if h else None,
                         "best_ref_is_true": bi == ti, "top_score": round(top, 3),
                         "movement": round(float(np.mean(np.std(w, axis=1))), 4)})
    with open(d / "diagnosis.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def reason(r):
        if r["true_ref_score"] is None or r["true_ref_score"] < chain:
            return "score too low (below the chain level)"
        if r["position_error_s"] > POS_TOL_S:
            return "scored high but at the WRONG place in the song (repeating material?)"
        if not r["best_ref_is_true"]:
            return "another reference song scored higher"
        return "found correctly"
    L = ["# Why windows inside planted copies were (not) found", "",
         f"{len(rows)} windows inside {len(truth)} planted copies. chain level {chain:.2f}, single level {single:.2f}.", "",
         "| reason | windows | share |", "|---|---|---|"]
    c = Counter(reason(r) for r in rows)
    L += [f"| {k} | {v} | {100 * v / len(rows):.0f}% |" for k, v in c.most_common()]
    L += ["", "## By edit", "", "| edit | windows | found correctly | score too low | wrong place | other song higher |", "|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[r["edit"]].append(reason(r))
    for e, rs in by.items():
        cc = Counter(rs)
        L.append(f"| {e} | {len(rs)} | {cc['found correctly']} | {cc['score too low (below the chain level)']} | "
                 f"{cc['scored high but at the WRONG place in the song (repeating material?)']} | {cc['another reference song scored higher']} |")
    low = [r["movement"] for r in rows if reason(r).startswith("score too low")]
    ok = [r["movement"] for r in rows if reason(r) == "found correctly"]
    if low and ok:
        L += ["", f"Music movement (higher = more changing notes): median {np.median(ok):.4f} where found, {np.median(low):.4f} where the score was too low."]
    ex = [r for r in rows if r["edit"] == "none"]
    L += ["", "## Unedited copies, window by window (these should all be found)", "",
          "| query | window start s | true-ref score | position error s | best ref is true | movement |", "|---|---|---|---|---|---|"]
    L += [f"| {r['query']} | {r['window_start_s']} | {r['true_ref_score']} | {r['position_error_s']} | {r['best_ref_is_true']} | {r['movement']} |" for r in ex]
    (d / "diagnosis.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
