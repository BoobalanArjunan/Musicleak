"""musicleak: does generated music copy songs from a reference set (such as the training data)?"""
from ._version import __version__
from .features import MusicleakError
from .scan import Calibration, Match, QueryResult, Report, run

from .report import report_to_html, write_html_report

__all__ = ["__version__", "MusicleakError", "Calibration", "Match", "QueryResult", "Report", "run", "report_to_html", "write_html_report"]
