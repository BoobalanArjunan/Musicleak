from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._version import __version__
from .features import MusicleakError
from .report import report_to_dict, report_to_text, write_reports, write_html_report
from .scan import run, stderr_progress


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="musicleak", description="Find stretches of generated (or any query) music that match songs in a reference set, "
                                                                "even after pitch, tempo, reverb or MP3 changes.")
    p.add_argument("--version", action="version", version=f"musicleak {__version__}")
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("check", help="compare query songs with a reference set",
                       description="QUERY is a song or a folder of songs (for example your model's outputs). --reference is the folder "
                                   "to compare against (for example the training songs).")
    c.add_argument("query", type=Path)
    c.add_argument("--reference", type=Path, required=True, help="folder (or file) of reference songs")
    c.add_argument("--control", type=Path, help="folder of songs known to be unrelated to the references; used to measure what chance "
                                                 "looks like (default: the reference songs against each other)")
    c.add_argument("--out", type=Path, help="write report.txt, report.json and matches.csv into this folder")
    c.add_argument("--json", action="store_true", help="print JSON instead of text")
    c.add_argument("--html", type=Path, help="write interactive visual HTML report")
    c.add_argument("--window", type=float, default=10.0, metavar="SECONDS", help="comparison window (default 10; at least 5)")
    c.add_argument("--cache", type=Path, help="folder to keep computed features, so repeat runs are fast")
    c.add_argument("--max-files", type=int, help="check only the first N query files")
    return p


def main(argv=None) -> int:
    p = _parser()
    a = p.parse_args(argv)
    if a.cmd != "check":
        p.print_help()
        return 2
    try:
        rep = run(a.query, a.reference, a.control, window_s=a.window, hop_s=a.window / 2, cache=a.cache, max_files=a.max_files,
                  progress=stderr_progress)
        if a.out:
            write_reports(rep, a.out)
        if a.html:
            write_html_report(rep, a.html)
    except MusicleakError as e:
        print(f"musicleak: {e}", file=sys.stderr)
        return 2
    if a.json:
        import json
        print(json.dumps(report_to_dict(rep), indent=2))
    else:
        print(report_to_text(rep))
        if a.out:
            print(f"\nreport files written to {a.out}")
    return 1 if rep.verdict() == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
