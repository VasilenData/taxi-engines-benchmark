"""Run the full matrix. Every measurement is a fresh subprocess; peak memory is sampled from the
whole process tree (so Spark's JVM is included). Each engine's answer is compared with DuckDB's."""
from __future__ import annotations

import argparse
import csv
import json
import os
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
        p.kill()
        stop.set()
        return {"status": "timeout"}
    stop.set()
    th.join(1)
    if p.returncode != 0:
        tail = "MemoryError" if "MemoryError" in se else se.strip().splitlines()[-1][:120] if se.strip() else "failed"
        return {"status": "error", "detail": tail}
    d = json.loads(so.strip().splitlines()[-1])
    d.update(status="ok", peak_gb=round(peak / 1e9, 2))
    return d


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        if a[c].dtype.kind in "fc":
            if not ((a[c] - b[c]).abs() <= 1e-6 * (b[c].abs() + 1)).all():
                return False
        elif not (a[c].values == b[c].values).all():
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="*", default=TIERS)
    ap.add_argument("--engines", nargs="*", default=list(ENGINES))
    ap.add_argument("--queries", nargs="*", default=QUERIES)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out", default=RESULTS)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    tmp = tempfile.mkdtemp()
    for tier in a.tiers:
        n_files = len(files_for(tier))
        for q in a.queries:
            ref_path = f"{tmp}/ref_{tier}_{q}.parquet"
            for eng in ["duckdb"] + [e for e in a.engines if e != "duckdb"]:
                if eng not in a.engines and eng != "duckdb":
                    continue
                times, peaks, agree, last = [], [], None, None
                for r in range(a.reps):
                    out = ref_path if eng == "duckdb" else f"{tmp}/{eng}_{tier}_{q}.parquet"
                    m = measure(eng, tier, q, out, a.timeout)
                    last = m
                    if m["status"] != "ok":
                        break
                    times.append(m["seconds"])
                    peaks.append(m["peak_gb"])
                    if r == 0 and eng != "duckdb":
                        agree = same(pd.read_parquet(out), pd.read_parquet(ref_path))
                row = {"tier": tier, "files": n_files, "query": q, "engine": eng, "status": last["status"],
                       "median_s": round(statistics.median(times), 2) if times else "",
                       "min_s": min(times) if times else "", "runs": len(times),
                       "peak_gb": max(peaks) if peaks else "", "agrees_with_duckdb": "" if eng == "duckdb" else agree,
                       "detail": last.get("detail", "")}
                rows.append(row)
                print(row, flush=True)
                with open(a.out, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(row))
                    w.writeheader()
                    w.writerows(rows)


if __name__ == "__main__":
    main()
