import csv
import json

import pytest

import synth
from conftest import song
from musicleak.cli import main
from musicleak.report import report_to_dict, report_to_text
from musicleak import scan

SEED = 103


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A reference folder (10 songs), a query folder with one copy of ref03 and one unrelated song."""
    root = tmp_path_factory.mktemp("world")
    (root / "ref").mkdir()
    (root / "q").mkdir()
    for i in range(10):
        synth.write(root / "ref" / f"ref{i:02d}.wav", song(100 + i))
    x, _ = synth.excerpt_in_new_song(song(SEED), 5.0, 30.0, 900, seconds=40.0, at_s=12.0)
    synth.write(root / "q" / "copy.wav", x)
    synth.write(root / "q" / "fresh.wav", song(777))
    return root


@pytest.fixture(scope="module")
def report(world):
    return scan.run(world / "q", world / "ref", calibration_windows=60)


def test_text_report_names_files_matches_and_explains(report):
    t = report_to_text(report)
    assert "FAIL copy.wav" in t and "confirmed" in t and "ref03.wav" in t and "same key" in t
    assert "PASS fresh.wav" in t and "Chroma finds the same notes" in t
    assert "NOT calibrated" not in t and "measured on" in t


def test_json_report_has_everything_a_script_needs(report):
    d = report_to_dict(report)
    assert d["verdict"] == "FAIL" and d["calibration"]["source"] == "references"
    f = {x["name"]: x for x in d["files"]}
    assert f["copy.wav"]["verdict"] == "FAIL" and f["copy.wav"]["matches"][0]["reference"] == "ref03.wav"
    assert f["copy.wav"]["matched_s"] > 15 and f["fresh.wav"]["matches"] == []
    json.dumps(d)


def test_cli_exit_codes_and_out_files(world, tmp_path, capsys):
    rc = main(["check", str(world / "q" / "copy.wav"), "--reference", str(world / "ref"), "--out", str(tmp_path / "o")])
    assert rc == 1 and "FAIL" in capsys.readouterr().out
    assert {p.name for p in (tmp_path / "o").iterdir()} == {"report.txt", "report.json", "matches.csv"}
    rows = list(csv.DictReader(open(tmp_path / "o" / "matches.csv")))
    assert rows and rows[0]["reference"] == "ref03.wav" and rows[0]["kind"] == "confirmed" and float(rows[0]["score"]) > 0.5
    assert main(["check", str(world / "q" / "fresh.wav"), "--reference", str(world / "ref")]) == 0


def test_cli_json_output_parses(world, capsys):
    main(["check", str(world / "q" / "fresh.wav"), "--reference", str(world / "ref"), "--json"])
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"


def test_cli_usage_errors_exit_2(world, tmp_path, capsys):
    assert main([]) == 2
    assert main(["check", str(tmp_path / "nothing"), "--reference", str(world / "ref")]) == 2
    assert "musicleak:" in capsys.readouterr().err
    assert main(["check", str(world / "q"), "--reference", str(tmp_path / "missing")]) == 2
    assert main(["check", str(world / "q"), "--reference", str(world / "ref"), "--window", "1"]) == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "musicleak 0.1.0a1" in capsys.readouterr().out


def test_text_report_gives_seconds_and_share_of_the_song():
    from musicleak.scan import Calibration, Match, QueryResult, Report
    m = Match("confirmed", "ref.wav", 20.0, 40.0, 5.0, 25.0, -2, 1.0, 0.9)
    t = report_to_text(Report([QueryResult("song.wav", 200.0, [m])], Calibration(0.8, 0.7, 50, "references"), [], {"references": 3}))
    assert "FAIL song.wav (3:20): 20 s matched (10% of the song)" in t and "0:20-0:40 matches ref.wav 0:05-0:25, -2 semitones" in t


def test_html_report_generation(report):
    from musicleak.report import report_to_html
    html = report_to_html(report)
    assert '<!DOCTYPE html>' in html
    assert 'MusicLeak Audit Report' in html
    assert 'FAIL' in html
    assert 'copy.wav' in html
    assert 'downloadJSON' in html


def test_cli_html_flag(world, tmp_path):
    html_out = tmp_path / 'audit_report.html'
    rc = main(['check', str(world / 'q' / 'copy.wav'), '--reference', str(world / 'ref'), '--html', str(html_out)])
    assert rc == 1
    assert html_out.exists()
    content = html_out.read_text(encoding='utf-8')
    assert '<!DOCTYPE html>' in content
    assert 'ref03.wav' in content
