import csv
import argparse
import sys
import subprocess
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tbench import orchestrate
from tbench.data import files_for


def test_partial_tier_cannot_be_mislabeled_as_six_months(tmp_path):
    (tmp_path / "yellow_tripdata_2024-01.parquet").touch()
    assert len(files_for("1m", str(tmp_path))) == 1
    with pytest.raises(FileNotFoundError, match="Incomplete 6m"):
        files_for("6m", str(tmp_path))


def test_complete_tier_excludes_unrelated_files(tmp_path):
    for year in (2023, 2024, 2025):
        for month in range(1, 13):
            (tmp_path / f"yellow_tripdata_{year}-{month:02d}.parquet").touch()
    selected = files_for("24m", str(tmp_path))
    assert len(selected) == 24 and not any("2025-" in path for path in selected)


def test_equivalent_null_results_compare_equal_and_integer_counts_remain_exact():
    assert orchestrate.same(pd.DataFrame({"v": [1.0, float("nan")]}), pd.DataFrame({"v": [1.0, float("nan")]}))
    assert not orchestrate.same(pd.DataFrame({"n": [1000000]}), pd.DataFrame({"n": [1000001]}))


def test_every_repetition_is_checked_and_mismatch_fails_cli(tmp_path, monkeypatch):
    output = tmp_path / "results.csv"
    monkeypatch.setattr(sys, "argv", ["bench", "--tiers", "1m", "--engines", "pandas", "--queries", "q1_groupby",
                                     "--reps", "2", "--out", str(output)])
    monkeypatch.setattr(orchestrate, "files_for", lambda tier: ["synthetic.parquet"])
    frames = {}
    calls = {}

    def measure(engine, tier, query, out, timeout):
        calls[engine] = calls.get(engine, 0) + 1
        value = 2 if engine == "pandas" and calls[engine] == 2 else 1
        frames[out] = pd.DataFrame({"n": [value]})
        return {"status": "ok", "seconds": 0.01, "peak_gb": 0.01, "result_rows": 1}

    monkeypatch.setattr(orchestrate, "measure", measure)
    monkeypatch.setattr(orchestrate.pd, "read_parquet", lambda path: frames[path])
    assert orchestrate.main() == 1
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[-1]["status"] == "mismatch"
    assert rows[-1]["runs"] == "1" and rows[-1]["agrees_with_duckdb"] == "False"
    assert "repetition 2" in rows[-1]["detail"]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_nonpositive_repetition_or_timeout_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        orchestrate.positive_int(value)


def test_report_does_not_call_a_mismatch_or_timeout_agreement():
    from tbench.report import comparison_summary

    frame = pd.DataFrame([
        {"engine": "duckdb", "tier": "1m", "query": "q1", "status": "ok", "agrees_with_duckdb": ""},
        {"engine": "pandas", "tier": "1m", "query": "q1", "status": "mismatch", "agrees_with_duckdb": False},
        {"engine": "spark", "tier": "1m", "query": "q1", "status": "timeout", "agrees_with_duckdb": None},
    ])
    summary = comparison_summary(frame)
    assert "2 case(s) failed" in summary and "All recorded cases passed" not in summary


def test_timeout_terminates_observed_children_and_reaps_parent(monkeypatch):
    killed = []

    class Process:
        pid = 123
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("synthetic benchmark", timeout)
            return "", ""

        def kill(self):
            killed.append("parent")

    child = SimpleNamespace(kill=lambda: killed.append("child"), is_running=lambda: True,
                            memory_info=lambda: SimpleNamespace(rss=1024))
    parent = SimpleNamespace(children=lambda recursive: [child], is_running=lambda: True,
                             memory_info=lambda: SimpleNamespace(rss=1024))
    process = Process()
    monkeypatch.setattr(orchestrate.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(orchestrate.psutil, "Process", lambda pid: parent)
    assert orchestrate.measure("spark", "1m", "q1_groupby", "synthetic.parquet", 1) == {"status": "timeout"}
    assert killed == ["child", "parent"] and process.calls == 2


@pytest.mark.parametrize('output', ['', '{}', '[]', '{broken',
                                    json.dumps({'engine': 'duckdb', 'tier': '1m', 'query': 'q1_groupby', 'seconds': float('nan')}),
                                    json.dumps({'engine': 'spark', 'tier': '1m', 'query': 'q1_groupby', 'seconds': 1})])
def test_malformed_worker_measurement_is_a_recorded_failure(monkeypatch, output):
    process = SimpleNamespace(pid=123, returncode=0, communicate=lambda timeout: (output, ''))
    parent = SimpleNamespace(children=lambda recursive: [], is_running=lambda: True,
                             memory_info=lambda: SimpleNamespace(rss=1024))
    monkeypatch.setattr(orchestrate.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(orchestrate.psutil, 'Process', lambda pid: parent)
    assert orchestrate.measure('duckdb', '1m', 'q1_groupby', 'synthetic.parquet', 1)['status'] == 'error'


def test_atomic_csv_checkpoint_preserves_previous_bytes_on_serialization_failure(tmp_path, monkeypatch):
    path = tmp_path / 'results.csv'
    old = b'tier,status\n1m,ok\n'
    path.write_bytes(old)
    real_writer = orchestrate.csv.DictWriter

    def broken_writer(stream, fieldnames):
        writer = real_writer(stream, fieldnames=fieldnames)

        def fail(rows):
            stream.write('partial broken row')
            raise OSError('disk failure')

        writer.writerows = fail
        return writer

    monkeypatch.setattr(orchestrate.csv, 'DictWriter', broken_writer)
    with pytest.raises(OSError, match='disk failure'):
        orchestrate.write_results(str(path), [{'tier': '6m', 'status': 'ok'}])
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('selection,value', [('tiers', '1m'), ('engines', 'duckdb'), ('queries', 'q1_groupby')])
def test_duplicate_matrix_selections_are_refused(selection, value, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['bench', f'--{selection}', value, value])
    with pytest.raises(SystemExit) as error:
        orchestrate.main()
    assert error.value.code == 2


def test_duplicate_or_nonfinite_cases_cannot_inflate_report_totals():
    from tbench.report import totals
    frame = pd.DataFrame([{'engine': 'duckdb', 'tier': '1m', 'query': f'q{i}',
                           'status': 'ok', 'median_s': 1.0, 'peak_gb': 1.0} for i in range(4)])
    assert totals(frame).total_s.iloc[0] == 4
    with pytest.raises(ValueError, match='Duplicate'):
        totals(pd.concat([frame, frame.iloc[[0]]]))
    frame.loc[0, 'median_s'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        totals(frame)


@pytest.mark.parametrize("fault", ["missing", "corrupt", "row_count", "missing_count"])
def test_unwritten_or_invalid_repetition_cannot_reuse_previous_result(tmp_path, monkeypatch, fault):
    from pathlib import Path

    output = tmp_path / "results.csv"
    monkeypatch.setattr(sys, "argv", ["bench", "--tiers", "1m", "--engines", "pandas", "--queries", "q1_groupby",
                                     "--reps", "2", "--out", str(output)])
    monkeypatch.setattr(orchestrate, "files_for", lambda tier: ["synthetic.parquet"])
    paths = []

    def worker(engine, tier, query, out, timeout):
        paths.append(Path(out))
        result = {"status": "ok", "seconds": 0.1, "peak_gb": 0.1, "result_rows": 1}
        if len(paths) == 1:
            pd.DataFrame({"n": [7]}).to_parquet(out)
        elif fault == "corrupt":
            Path(out).write_bytes(b"not parquet")
        elif fault == "row_count":
            pd.DataFrame({"n": [7, 8]}).to_parquet(out)
        elif fault == "missing_count":
            pd.DataFrame({"n": [7]}).to_parquet(out)
            result.pop("result_rows")
        return result

    monkeypatch.setattr(orchestrate, "measure", worker)
    assert orchestrate.main() == 1
    rows = list(csv.DictReader(output.open(newline="")))
    assert len(rows) == 1  # No engine is measured against an unvalidated DuckDB reference.
    assert rows[0]["status"] == "error" and rows[0]["runs"] == "1"
    assert "invalid result artifact" in rows[0]["detail"]
    assert paths[0] != paths[1]
    assert not any(path.exists() for path in paths)  # Temporary outputs are cleaned on failure.


def test_valid_workers_use_a_fresh_artifact_for_every_case_and_repetition(tmp_path, monkeypatch):
    from pathlib import Path

    output = tmp_path / "results.csv"
    monkeypatch.setattr(sys, "argv", ["bench", "--tiers", "1m", "--engines", "pandas", "--queries", "q1_groupby",
                                     "--reps", "2", "--out", str(output)])
    monkeypatch.setattr(orchestrate, "files_for", lambda tier: ["synthetic.parquet"])
    paths = []

    def worker(engine, tier, query, out, timeout):
        paths.append(Path(out))
        assert not Path(out).exists()
        pd.DataFrame({"n": [7]}).to_parquet(out)
        return {"status": "ok", "seconds": 0.1, "peak_gb": 0.1, "result_rows": 1}

    monkeypatch.setattr(orchestrate, "measure", worker)
    assert orchestrate.main() == 0
    rows = list(csv.DictReader(output.open(newline="")))
    assert len(rows) == 2 and all(row["runs"] == "2" for row in rows)
    assert rows[-1]["agrees_with_duckdb"] == "True"
    assert len(set(paths)) == 4 and not any(path.exists() for path in paths)
