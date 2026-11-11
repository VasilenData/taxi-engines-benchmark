"""Run the full matrix. Every measurement is a fresh subprocess; peak memory is sampled from the
whole process tree (so Spark's JVM is included). Each engine's answer is compared with DuckDB's."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import pandas as pd
import psutil

from .data import TIERS, files_for
from .engines import ENGINES, QUERIES, _tidy  # noqa: F401  (_tidy kept for symmetry with tests)

RESULTS = "results/results.csv"


def measure(engine: str, tier: str, query: str, out: str, timeout: int) -> dict:
    cmd = [sys.executable, "-m", "tbench.run_one", engine, tier, query, "--out", out]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = 0
    stop = threading.Event()

    def sample():
        nonlocal peak
        try:
            ps = psutil.Process(p.pid)
            while not stop.is_set():
                procs = [ps, *ps.children(recursive=True)]
                peak = max(peak, sum(x.memory_info().rss for x in procs if x.is_running()))
                time.sleep(0.05)
        except psutil.Error:
            pass

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        so, se = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Killing only Python leaves Spark's JVM running and contaminates later measurements.
        try:
            children = psutil.Process(p.pid).children(recursive=True)
        except psutil.Error:
            children = []
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        p.kill()
        p.communicate()
        stop.set()
        th.join(1)
        return {"status": "timeout"}
    stop.set()
    th.join(1)
    if p.returncode != 0:
        tail = "MemoryError" if "MemoryError" in se else se.strip().splitlines()[-1][:120] if se.strip() else "failed"
        return {"status": "error", "detail": tail}
    try:
        d = json.loads(so.strip().splitlines()[-1])
        seconds = d['seconds']
        if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError('invalid seconds')
        if type(d.get('result_rows')) is not int or d['result_rows'] < 0:
            raise ValueError('invalid result row count')
        if (d.get('engine'), d.get('tier'), d.get('query')) != (engine, tier, query):
            raise ValueError('worker identity does not match requested case')
    except (ValueError, TypeError, KeyError, IndexError):
        return {'status': 'error', 'detail': 'invalid worker measurement output'}
    d.update(status="ok", peak_gb=round(peak / 1e9, 2))
    return d


def write_results(path: str, rows: list[dict]) -> None:
    """Replace a complete CSV checkpoint; serialization failure preserves the old one."""
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                         dir=destination.parent, prefix='.benchmark-', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        try:
            pd.testing.assert_series_equal(a[c], b[c], check_dtype=False, check_names=False,
                                           check_exact=a[c].dtype.kind not in "fc", rtol=1e-6, atol=1e-6)
        except AssertionError:
            return False
    return True


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", choices=TIERS, default=TIERS)
    ap.add_argument("--engines", nargs="+", choices=list(ENGINES), default=list(ENGINES))
    ap.add_argument("--queries", nargs="+", choices=QUERIES, default=QUERIES)
    ap.add_argument("--reps", type=positive_int, default=3)
    ap.add_argument("--timeout", type=positive_int, default=1200)
    ap.add_argument("--out", default=RESULTS)
    a = ap.parse_args()
    for name in ('tiers', 'engines', 'queries'):
        values = getattr(a, name)
        if len(values) != len(set(values)):
            ap.error(f'--{name} must not contain duplicates')
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="taxi-benchmark-") as tmp:
        for tier in a.tiers:
            n_files = len(files_for(tier))
            for q in a.queries:
                reference = None
                for eng in ["duckdb"] + [e for e in a.engines if e != "duckdb"]:
                    times, peaks, agree, last = [], [], None, None
                    for r in range(a.reps):
                        # Every invocation gets a path that has never existed. A worker
                        # cannot accidentally reuse a previous repetition's valid file.
                        out = str(Path(tmp) / f"{eng}_{tier}_{q}_{r}.parquet")
                        m = measure(eng, tier, q, out, a.timeout)
                        last = m
                        if m["status"] != "ok":
                            break
                        try:
                            if type(m.get("result_rows")) is not int or m["result_rows"] < 0:
                                raise ValueError("worker result_rows must be a non-negative integer")
                            result = pd.read_parquet(out)
                            if len(result) != m["result_rows"]:
                                raise ValueError("artifact row count differs from worker measurement")
                        except (OSError, ValueError) as exc:
                            last = {"status": "error", "detail": f"invalid result artifact: {type(exc).__name__}"}
                            break
                        if eng == "duckdb" and r == 0:
                            reference = result
                        if reference is None or not same(result, reference):
                            agree = False
                            last = {"status": "mismatch", "detail": f"result mismatch on repetition {r + 1}"}
                            break
                        agree = True
                        times.append(m["seconds"])
                        peaks.append(m["peak_gb"])
                    row = {"tier": tier, "files": n_files, "query": q, "engine": eng, "status": last["status"],
                           "median_s": round(statistics.median(times), 2) if times else "",
                           "min_s": min(times) if times else "", "runs": len(times),
                           "peak_gb": max(peaks) if peaks else "", "agrees_with_duckdb": "" if eng == "duckdb" else agree,
                           "detail": last.get("detail", "")}
                    rows.append(row)
                    print(row, flush=True)
                    write_results(a.out, rows)
                    if eng == "duckdb" and row["status"] != "ok":
                        break  # no validated reference exists for the remaining engines
    return 0 if rows and all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
